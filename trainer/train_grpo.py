"""使用 GRPO / CISPO 对 MiniMind 进行在线强化学习训练。

建议先读（按数据流顺序）：
1. ``README.md`` 的 GRPO、CISPO 小节：先认识组内相对优势、概率比裁剪和 KL 约束。
2. ``model/model_minimind.py`` 中的 ``MiniMindForCausalLM.forward``：理解 logits
   为什么用位置 t 预测位置 t+1，以及可选 MoE 辅助损失从何而来。
3. ``dataset/lm_dataset.py`` 中的 ``RLAIFDataset``：理解数据集只提供 prompt，
   最后一条参考回复不参加训练；``thinking_ratio`` 会改变提示词的思考模式。
4. ``trainer/rollout_engine.py`` 中的 ``RolloutResult``、``TorchRolloutEngine`` 和
   ``SGLangRolloutEngine``：弄清在线生成、旧策略 log 概率和回答掩码的来源。
5. ``trainer/trainer_utils.py`` 中的 ``init_model``、``LMForRewardModel``、
   ``lm_checkpoint`` 和 ``SkipBatchSampler``：理解初始权重、奖励模型与断点续训。

读完本文件后，推荐看 ``trainer/train_ppo.py``（用 Critic 估计优势的另一种在线 RL）、
``trainer/train_agent.py``（把组内优势扩展到多轮工具调用）和
``trainer/train_dpo.py``（使用离线偏好数据的对齐训练）。

阅读约定：B 为一个批次的 prompt 数，G 为每个 prompt 采样的回答数，P 为 prompt
补齐后的长度，R 为回答补齐后的长度，N=B*G。回答按“同一 prompt 的 G 条回答相邻”
排列。Policy 是正在训练的模型；Ref 是冻结的初始策略；Reward Model 给完整回答打分。
每批的流程是：在线采样 -> 奖励与组内优势 -> 逐 token 策略损失 -> 更新参数。
"""

import os
import sys

# 支持从 trainer/ 目录直接运行，并让绝对导入找到仓库根目录的 model、dataset。
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # 先导入 datasets，规避 Windows 上 pyarrow/torch 的 DLL 冲突（#771）。
import argparse
import math
import re
import gc
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model, LMForRewardModel
from trainer.rollout_engine import create_rollout_engine

# 训练库可能产生较多非关键告警；这里沿用仓库脚本的统一设置。
warnings.filterwarnings('ignore')


def rep_penalty(text, n=3, cap=0.5):
    """统计重复的 n-gram 并扣分，最高扣 ``cap``；用于回答正文的规则奖励。"""
    # 将英文单词和标点分别视为 token，并统一小写；这不是模型 tokenizer 的切分结果。
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    # 逐位置取长度为 n 的连续片段；短文本不足 n 个 token 时没有片段。
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    # 总片段数减去不同片段数就是重复量，最后把惩罚限制在 [0, cap]。
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0


def calculate_rewards(prompts, responses, reward_model):
    """按 prompt 分组，为 N=B*G 条完整回答计算标量奖励。

    奖励由长度/思考格式、回答正文重复惩罚和外部奖励模型分数相加得到。这里只评分，
    不对奖励模型反向传播。函数读取入口处解析出的全局 ``args``。
    """
    rewards = torch.zeros(len(responses), device=args.device)

    with torch.no_grad():
        reward_model_scores = []
        batch_size = len(prompts)

        # rollout_engine 保证同一 prompt 的 G 个回答连续排列，所以索引是 i*G+j。
        for i in range(batch_size):
            for j in range(args.num_generations):
                response_idx = i * args.num_generations + j
                response = responses[response_idx]
                prompt = prompts[i]

                # 从聊天模板字符串还原完整的 system/user/assistant 消息，供奖励模型使用。
                # 末尾尚未闭合的 assistant 生成前缀不会匹配，正好不作为历史回答传入。
                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]
                answer = response
                # 整条回答过短或过长扣分；这里的 len 是解码后字符串的字符数。
                rewards[response_idx] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
                if '</think>' in response:
                    # 有闭合标签时分开评估思考段与最终答案；仅给恰好一个闭合标签加分。
                    thinking_content, answer_content = response.split('</think>', 1)
                    rewards[response_idx] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
                    rewards[response_idx] += 0.25 if response.count('</think>') == 1 else -0.25
                    answer = answer_content.strip()
                # 重复惩罚和奖励模型只评估 answer；有思考段时不把思考内容混入正文。
                rewards[response_idx] -= rep_penalty(answer)

                score = reward_model.get_score(messages, answer)
                reward_model_scores.append(score)

        # LMForRewardModel.get_score 已将外部模型分数裁剪到 [-3, 3]。
        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards


def grpo_train_epoch(epoch, loader, iters, rollout_engine, ref_model, reward_model, start_step=0, wandb=None, use_sglang=False):
    """遍历一个 epoch：每批在线生成 G 个回答，再用组内相对奖励更新 Policy。

    ``start_step`` 是断点续训时已完成的批次数，用于延续日志/保存步数。
    ``use_sglang`` 为调用接口预留参数；具体引擎已封装在 ``rollout_engine`` 中。
    模型、分词器、优化器等由主入口初始化，在此通过模块级变量使用。
    """
    for step, batch in enumerate(loader, start=start_step + 1):
        # RLAIFDataset 的 answer 是空占位符；真实回答由当前策略在线生成。
        prompts = batch['prompt']  # 长度为 B 的字符串列表。
        # 左补齐使同批 prompt 的末尾对齐；不额外添加特殊 token，避免重复聊天模板标记。
        prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False,
                                  padding_side="left", add_special_tokens=False).to(args.device)
        if args.max_seq_len:
            # 只保留末尾 P 个 token，尽量保住最近的提问和 assistant 生成起点。
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]

        # 每个 prompt 采样 G 次；引擎同时记录采样时策略的逐 token log 概率（旧策略）。
        rollout_result = rollout_engine.rollout(
            prompt_ids=prompt_inputs["input_ids"],
            attention_mask=prompt_inputs["attention_mask"],
            num_generations=args.num_generations,
            max_new_tokens=args.max_gen_len,
            temperature=0.8,
        )
        outputs = rollout_result.output_ids             # [N, P+R]：prompt 与回答的 token。
        completion_ids = rollout_result.completion_ids  # [N, R]：只含回答，供 EOS 定位。
        completions = rollout_result.completions        # N 条解码后的回答字符串，供打分。
        # detach 防止梯度穿过采样过程；SGLang 情况下这些值来自外部服务。
        old_per_token_logps = rollout_result.per_token_logps.to(args.device).detach()
        # 每行真实回答在 outputs 中的起点；Torch 引擎各行相同，SGLang 可各不相同。
        prompt_lens = rollout_result.prompt_lens.to(args.device)
        # 构造前向计算所需的完整注意力掩码，并以引擎返回的掩码标记真实回答 token。
        full_mask = (outputs != tokenizer.pad_token_id).long()
        # 自回归模型在位置 t 的 logits 预测位置 t+1：第一个回答 token
        # 对应 prompt 最后一个位置的 logits，因此 logp_pos 从 prompt_lens-1 开始。
        logp_pos = prompt_lens.unsqueeze(1) - 1 + torch.arange(completion_ids.size(1), device=args.device).unsqueeze(0)
        full_mask.scatter_(1, logp_pos + 1, rollout_result.completion_mask.to(args.device, dtype=full_mask.dtype))

        # 每条完整回答只有一个标量奖励，随后会广播给这条回答的所有有效 token。
        rewards = calculate_rewards(prompts, completions, reward_model).to(args.device)  # [N]

        with autocast_ctx:
            # 当前 Policy 对同一批已生成序列重新打分；MoE 模型可能额外返回负载均衡损失。
            res = model(outputs, attention_mask=full_mask)
            aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
            # 先对词表维做 log_softmax，再取实际生成 token 的概率，最后选回答位置。
            # [:, :-1] 与 outputs[:, 1:] 错位对齐，结果形状为 [N, R]。
            per_token_logps = F.log_softmax(res.logits[:, :-1, :], dim=-1).gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)

        with torch.no_grad():
            # Ref 固定在 from_weight 指定的起始权重，用来限制 Policy 偏离原模型；不更新参数。
            ref_per_token_logps = F.log_softmax(ref_model(outputs, attention_mask=full_mask).logits[:, :-1, :], dim=-1).gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)

        # 可选打印原始提示词、每条生成文本和奖励，便于检查奖励是否符合直觉。
        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            for i in range(len(prompts)):
                Logger(f"[DEBUG] step={step}, sample[{i}]")
                Logger('-'*100)
                Logger(f"{'=' * 30} [DEBUG] sample[{i}] CONTEXT_BEGIN {'=' * 30}")
                Logger(prompts[i])
                Logger(f"{'=' * 31} [DEBUG] sample[{i}] CONTEXT_END {'=' * 31}")
                for j in range(args.num_generations):
                    idx = i * args.num_generations + j
                    Logger(f"{'=' * 28} [DEBUG] gen[{j}] RESPONSE_BEGIN {'=' * 28}")
                    Logger(completions[idx])
                    Logger(f"{'=' * 29} [DEBUG] gen[{j}] RESPONSE_END {'=' * 29}")
                    Logger(f"[DEBUG] gen[{j}] reward={rewards[idx].item():.4f}")
                Logger('='*100)

        # 同题 G 条回答组成一组：以组均值为 baseline，再按组内标准差归一化。
        # 优于同题其他回答的 advantage 为正，较差的为负；不需要另训 Critic。
        grouped_rewards = rewards.view(-1, args.num_generations)  # [B, G]
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)  # [N]
        std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)  # [N]
        # 1e-4 防止整组奖励相同时除零；此时该组优势都为 0，不提供策略项梯度。
        advantages = (rewards - mean_r) / (std_r + 1e-4)  # [N]

        # 回答的有效范围以引擎掩码为基础，再在首次 EOS 处截断（EOS 自身保留）。
        # 未出现 EOS 时 eos_idx 默认是最后一列，最终仍受 completion_pad_mask 限制。
        completion_pad_mask = rollout_result.completion_mask.to(args.device).bool()
        is_eos = (completion_ids == tokenizer.eos_token_id) & completion_pad_mask  # [N, R]
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1) - 1, dtype=torch.long, device=args.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        completion_mask = ((torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)) & completion_pad_mask).int()  # [N, R]

        # log π_ref - log π_policy 构成参考约束的逐 token KL 估计；下式非负且在两者
        # 相等时为 0。旧策略仅用于下面的采样重要性比，不是 Ref。
        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1  # [N, R]
        ratio = torch.exp(per_token_logps - old_per_token_logps)  # [N, R]
        if args.loss_type == "cispo":
            # CISPO 把裁剪后的 ratio 当作常数权重，乘当前 log 概率；即使比值触顶，
            # log 概率这一项仍可传梯度。这里只有上界 epsilon_high。
            clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
            per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl)
        else:
            # GRPO 使用 PPO 式双侧裁剪；min 选择更保守的策略收益。
            clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
            per_token_loss1 = ratio * advantages.unsqueeze(1)
            per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
            per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)
        # 每条回答先按有效 token 平均，再对 N 条回答平均；padding 不计入损失。
        # clamp 防止极端空回答造成分母为 0；MoE 辅助损失也参与总反向传播。
        policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)).mean()
        loss = (policy_loss + aux_loss) / args.accumulation_steps  # 标量；按累积步数缩放梯度。
        loss.backward()

        # 累积若干批次再更新一次；epoch 尾部即使不足设定步数也执行更新。
        if step % args.accumulation_steps == 0 or step == iters:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # 日志中的 Actor Loss 实际为缩放前的总 loss（策略项加可能存在的 MoE 项）。
        if step % args.log_interval == 0 or step == iters:
            policy_loss_val = loss.item() * args.accumulation_steps
            current_aux_loss = aux_loss.item()
            avg_reward_val = rewards.mean().item()
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()
            # 此处展示的是 log 概率差的平均值；优化时使用的是上面的非负 KL 估计。
            kl_ref_val = ((ref_per_token_logps - per_token_logps) * completion_mask).sum().item() / max(completion_mask.sum().item(), 1)
            advantages_mean_val = advantages.mean().item()
            advantages_std_val = advantages.std().item()
            current_lr = optimizer.param_groups[0]['lr']

            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                   f'Reward: {avg_reward_val:.4f}, KL_ref: {kl_ref_val:.4f}, '
                   f'Adv Std: {advantages_std_val:.4f}, Adv Mean: {advantages_mean_val:.4f}, '
                   f'Actor Loss: {policy_loss_val:.4f}, Avg Response Len: {avg_len_val:.2f}, Learning Rate: {current_lr:.8f}')

            if wandb and is_main_process():
                wandb.log({
                    "reward": avg_reward_val,
                    "kl_ref": kl_ref_val,
                    "advantages_std": advantages_std_val,
                    "advantages_mean": advantages_mean_val,
                    "policy_loss": policy_loss_val,
                    "avg_response_len": avg_len_val,
                    "learning_rate": current_lr
                })

        # 只有主进程写模型权重与完整续训断点，避免多进程同时写同一个文件。
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # 去掉 DDP / torch.compile 包装，保证保存的参数名与原始模型一致。
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scheduler=scheduler)
            model.train()
            del state_dict

        # 将更新后的 Policy 同步给采样引擎；Torch 引擎保存对象引用，SGLang 引擎
        # 通过共享目录与 HTTP 请求更新服务器权重。当前同步频率跟保存间隔一致。
        if step % args.save_interval == 0 or step == iters: rollout_engine.update_policy(model)

        # 及时释放本批大张量的引用，降低下一批生成时的显存压力。
        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
        del completions, rewards, grouped_rewards, mean_r, std_r, advantages, completion_mask, completion_pad_mask, prompt_lens, logp_pos


if __name__ == "__main__":
    # 命令行参数分成输出/优化、模型与数据、GRPO 损失、续训与采样引擎几组。
    # 路径默认值沿用 README 从 trainer/ 目录启动的约定；从其他目录启动需显式传路径。
    parser = argparse.ArgumentParser(description="MiniMind GRPO (Group Relative Policy Optimization)")
    # 输出文件名形如 grpo_768.pth；batch_size 是每批 prompt 数，而非回答数。
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='grpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    # 模型结构必须与 from_weight 指向的起始权重兼容；MoE 另有辅助损失和文件后缀。
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    # max_seq_len 截断 prompt，max_gen_len 限制新生成 token；配置中的总长度是两者之和。
    parser.add_argument('--max_seq_len', default=768, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1024, help="生成的最大长度")
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif.jsonl", help="RLAIF数据路径")
    # G=num_generations。G 至少应为 2，组内才有可比较的不同回答。
    parser.add_argument("--num_generations", type=int, default=6, help="每个prompt生成的样本数")
    # beta 控制偏离 Ref 的惩罚；默认选择 CISPO，改为 grpo 可运行 PPO 式裁剪目标。
    parser.add_argument("--beta", type=float, default=0.1, help="KL惩罚系数")
    parser.add_argument("--loss_type", type=str, default="cispo", choices=["grpo", "cispo"], help="loss类型")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GRPO的PPO clip epsilon")
    parser.add_argument("--epsilon_high", type=float, default=5.0, help="epsilon上界")
    # from_weight 指初始纯模型权重；from_resume 读取本次训练保存的完整续训断点。
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    # 这里期望奖励模型实现自定义 get_score 接口，详见 LMForRewardModel。
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-GRPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--debug_mode", action="store_true", help="是否打印训练调试采样")
    parser.add_argument("--debug_interval", type=int, default=20, help="debug模式下每隔多少step打印一次采样")
    # 数据集按此概率打开聊天模板中的思考模式，影响 prompt 而非损失公式。
    parser.add_argument("--thinking_ratio", type=float, default=0.9, help="按概率开启thinking（0.0~1.0）")
    # torch 在当前进程推理；sglang 从外部服务取回答并同步权重到共享目录。
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"], help="rollout引擎类型")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998", help="SGLang服务器URL")
    parser.add_argument("--sglang_model_path", type=str, default="../model", help="SGLang tokenizer路径")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_grpo", help="SGLang共享存储路径")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    # torchrun 启动时每个 rank 绑定各自 GPU；单进程模式下不创建进程组。
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    # 不同 rank 使用不同种子；epoch 内还会重设种子以重建采样顺序。
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查续训断点 ==========
    os.makedirs(args.save_dir, exist_ok=True)
    # 模型允许的序列长度要容纳截断后的 prompt 与最多 max_gen_len 个新 token。
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               max_seq_len=args.max_seq_len + args.max_gen_len, use_moe=bool(args.use_moe))
    # 不要求续训时断点必然存在；找不到时从 from_weight 正常开始。
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    # CPU 不进入 CUDA 自动混合精度上下文；GPU 上按 dtype 选择 BF16 或 FP16。
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配置实验记录 ==========
    # 这里的变量名叫 wandb，实际导入的是 SwanLab 的兼容接口；仅主进程记录。
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-GRPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 初始化模型和数据 ==========
    base_weight = args.from_weight
    # Policy 从 from_weight 指定的权重开始学习，是优化器唯一直接更新的语言模型。
    model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    # Ref 从同一初始权重创建，此后固定参数，只为 KL 约束提供基准概率。
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    # Reward Model 是独立的外部评分模型，不由此脚本训练。
    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
    # Rollout 引擎只负责 Policy 推理和记录采样概率，不计算训练损失。
    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine,
        policy_model=model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )
    # 数据集返回原始 prompt 字符串；DDP 时各 rank 由 DistributedSampler 分配样本。
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len, thinking_ratio=args.thinking_ratio)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # 只有 Policy 参数交给 AdamW。先计算每个 epoch 的批次数，再按实际参数更新次数
    # 设置余弦退火周期；梯度累积时一个优化器 step 对应多个数据批次。
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    total_optimizer_steps = math.ceil(iters / args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    
    # ========== 6. 从断点恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        # 除模型外恢复优化器和学习率调度器，否则续训初期的更新轨迹会改变。
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        # 各 rank 的 RoPE buffer 相同，无需 DDP 每次前向时重新广播。
        model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
    # 首批采样前同步 Policy。Torch 引擎接收模型对象；SGLang 引擎加载导出的权重。
    rollout_engine.update_policy(model)
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # 分布式采样器每轮换随机顺序；单进程则显式生成随机下标。
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        # 续训时跳过已完成的批次，并把日志/保存用的 step 编号接在断点之后。
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            grpo_train_epoch(epoch, loader, len(loader) + skip, rollout_engine, ref_model, reward_model, start_step, wandb, use_sglang = (args.rollout_engine == "sglang"))
        else:
            grpo_train_epoch(epoch, loader, len(loader), rollout_engine, ref_model, reward_model, 0, wandb, use_sglang = (args.rollout_engine == "sglang"))
    
    # ========== 9. 清理分布进程 ==========
    # 所有 rank 训练完成后再统一销毁进程组。
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
