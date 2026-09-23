"""使用 PPO 对 MiniMind 进行在线强化学习训练。

建议先读：
1. ``README.md`` 的 PPO 小节：了解 Actor、Critic、优势和裁剪目标的含义。
2. ``model/model_minimind.py`` 中的 ``MiniMindModel``、``MiniMindForCausalLM``：
   了解主干输出、语言模型头，以及“位置 t 预测位置 t+1”的对齐方式。
3. ``dataset/lm_dataset.py`` 中的 ``RLAIFDataset``：了解为什么数据集只提供 prompt，
   而回答必须由当前策略在线生成。
4. ``trainer/rollout_engine.py`` 中的 ``RolloutResult`` 和两个 rollout 引擎：
   了解生成结果、逐 token 对数概率、长度与掩码从何而来。
5. ``trainer/trainer_utils.py`` 中的 ``init_model``、``LMForRewardModel``、
   ``lm_checkpoint`` 和 ``SkipBatchSampler``：了解权重、奖励及断点续训。

读完本文件可继续看 ``trainer/train_grpo.py``（不训练 Critic 的组内相对优势）、
``trainer/train_dpo.py``（使用离线偏好对的训练），比较不同对齐方法的数据流。

阅读约定：B 是一批 prompt 的数量，P 是补齐后的 prompt 长度，R 是补齐后的回答长度。
Actor 是待训练的生成模型；Critic 预测每个生成位置的价值；Ref 是冻结的参考策略；
Reward Model 对完整回答打分。一次外层迭代先在线生成，再用同一批结果执行若干次 PPO 更新。
"""

import os
import sys

# 允许从 trainer/ 目录直接运行脚本，同时让绝对导入能找到仓库根目录下的模块。
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # 先导入 datasets，规避 Windows 上 pyarrow/torch 的 DLL 冲突（#771）。
import argparse
import math
import re
import warnings
import torch
import torch.distributed as dist
import torch.nn.functional as F
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model, LMForRewardModel
from trainer.rollout_engine import create_rollout_engine

warnings.filterwarnings('ignore')


def rep_penalty(text, n=3, cap=0.5):
    """按重复的 n-gram 数量扣分，最多扣 ``cap``；仅用于回答正文的启发式奖励。"""
    # 把英文单词和标点分别当作 token，统一小写后统计连续 n 个 token 的片段。
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    # len(grams)-len(set(grams)) 是重复片段数；短于 n 个 token 时不扣分。
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0


# Critic 复用语言模型主干，并新增“每个位置预测一个价值”的标量头。
class CriticModel(MiniMindForCausalLM):
    def __init__(self, params):
        super().__init__(params)
        # lm_head 不参与 forward，仅靠 tie_word_embeddings 与 embed_tokens 共享权重，解绑后 DDP 会报未使用参数
        self.value_head = nn.Linear(params.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        """返回 [B, 序列长度] 的价值；训练循环会取回答 token 前一位置的价值。"""
        # 直接调用主干，不走父类的词表投影；模型主干返回的首项是隐藏状态。
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = self.model.norm(outputs[0])
        # squeeze 只去掉末尾大小为 1 的价值维，不影响 batch 或时间维。
        values = self.value_head(hidden_states).squeeze(-1)
        return values


def calculate_rewards(prompts, responses, reward_model):
    """给每条完整回答一个标量奖励：[B]，由格式/长度规则与奖励模型分数相加。"""
    rewards = torch.zeros(len(responses), device=args.device)

    with torch.no_grad():
        reward_model_scores = []
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            # 从聊天模板生成的字符串中还原角色消息，供 LMForRewardModel.get_score 使用。
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [{"role": role, "content": content.strip()} for role, content in matches]
            answer = response
            # 长度、思考段闭合和重复度是简单规则分；不是由 Critic 预测的价值。
            rewards[i] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
            if '</think>' in response:
                thinking_content, answer_content = response.split('</think>', 1)
                rewards[i] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
                rewards[i] += 0.25 if response.count('</think>') == 1 else -0.25
                # 重复惩罚和奖励模型主要评估最终答案，不把思考内容当成答案正文。
                answer = answer_content.strip()
            rewards[i] -= rep_penalty(answer)

            score = reward_model.get_score(messages, answer)
            reward_model_scores.append(score)

        # 该分数由 trainer_utils.py 中的包装器限制在 [-3, 3]。
        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards


def ppo_train_epoch(epoch, loader, iters, rollout_engine, ref_model, actor_scheduler, critic_scheduler, reward_model, start_step=0, wandb=None):
    """遍历一个数据 epoch；每批先 rollout，再用 PPO 更新 Actor 和 Critic。

    这里的 ``step`` 是数据批次编号，不是优化器更新次数；后者还受 mini-batch、
    ``ppo_update_iters`` 和梯度累积影响。模型、优化器及配置在主程序中初始化。
    """
    actor_model.train()
    critic_model.train()
    grad_accum_step = 0

    for step, batch in enumerate(loader, start=start_step + 1):
        # 数据集不提供要模仿的答案；每次都由 rollout 引擎中的 Actor 在线采样。
        # SGLang 的服务权重按下方 update_policy 的时机同步，可能略滞后于训练中的 Actor。
        prompts = batch["prompt"]  # 长度为 B 的字符串列表
        # 左侧补齐让一批 prompt 的末尾对齐；截断保证输入不超过 prompt 长度上限。
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_len,
                        padding_side="left").to(args.device)  # input_ids、attention_mask 均为 [B, P]

        # 生成回答并记录采样时策略给各 token 的对数概率（PPO 的“旧策略”基准）。
        # num_generations=1 表示一个 prompt 对应一条回答；引擎可选本地 PyTorch 或 SGLang。
        rollout_result = rollout_engine.rollout(
            prompt_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            num_generations=1,
            max_new_tokens=args.max_gen_len,
            temperature=0.8,
        )
        gen_out = rollout_result.output_ids  # 完整 token 序列 [B, P+R]
        completion_ids = rollout_result.completion_ids  # 仅回答 token [B, R]
        prompt_lens = rollout_result.prompt_lens.to(args.device)  # 各行回答起点 [B]
        responses_text = rollout_result.completions  # 解码后的回答字符串
        old_resp_logp = rollout_result.per_token_logps.to(args.device)  # 采样策略的 log π_old [B, R]
        rewards = calculate_rewards(prompts, responses_text, reward_model)  # 每条回答一个分数 [B]

        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            for i in range(len(prompts)):
                Logger(f"[DEBUG] step={step}, sample[{i}]")
                Logger('-'*100)
                Logger(f"{'=' * 30} [DEBUG] sample[{i}] CONTEXT_BEGIN {'=' * 30}")
                Logger(prompts[i])
                Logger(f"{'=' * 31} [DEBUG] sample[{i}] CONTEXT_END {'=' * 31}")
                Logger(f"[DEBUG] prompt_len={prompt_lens[i].item()}, response_len={len(responses_text[i])}")
                Logger(f"{'=' * 28} [DEBUG] sample[{i}] RESPONSE_BEGIN {'=' * 28}")
                Logger(responses_text[i])
                Logger(f"{'=' * 29} [DEBUG] sample[{i}] RESPONSE_END {'=' * 29}")
                Logger(f"[DEBUG] reward={rewards[i].item():.4f}")
                Logger('='*100)

        # 奖励是序列级的，但 PPO/GAE 在回答 token 级别计算，需要先对齐位置与掩码。
        full_mask = (gen_out != tokenizer.pad_token_id).long()  # 整段前向时使用 [B, P+R]
        labels = gen_out[:, 1:].clone()  # 位置 t 的 logits 对应位置 t+1 的 token [B, P+R-1]
        B = len(prompts)
        resp_labels = completion_ids
        resp_idx = torch.arange(resp_labels.size(1), device=gen_out.device).unsqueeze(0)
        # 回答的第 0 个 token 由 prompt 最后一个位置预测，所以 logits 下标是 prompt_lens-1。
        # SGLang 去掉左侧 padding 后，各行 prompt_lens 可不同；这里逐行计算真实下标。
        logp_pos = prompt_lens.unsqueeze(1) - 1 + resp_idx
        resp_pad_mask = rollout_result.completion_mask.to(args.device).bool()
        # 使用引擎给出的有效回答位置修正注意力掩码，尤其适用于补齐后的 SGLang 返回值。
        full_mask.scatter_(1, logp_pos + 1, resp_pad_mask.to(full_mask.dtype))
        # 只训练首个 EOS 及其之前的 token；EOS 之后即使仍占有张量位置，也不参与损失。
        # 没有 EOS 时使用引擎的 completion_mask；空回答至少保留长度 1 以避免非法下标。
        resp_lengths = resp_pad_mask.sum(dim=1); valid_resp = resp_lengths > 0; eos_mask = resp_labels.eq(tokenizer.eos_token_id) & resp_pad_mask
        has_eos = eos_mask.any(dim=1); eos_pos = torch.argmax(eos_mask.int(), dim=1)
        resp_lengths = torch.where(has_eos, eos_pos + 1, resp_lengths).long().clamp(min=1)
        resp_policy_mask = ((resp_idx < resp_lengths.unsqueeze(1)) & resp_pad_mask).float()
        # 当前实现让 Actor 和 Critic 使用同一组有效回答位置。
        resp_value_mask = resp_policy_mask.clone()

        # 下面构造固定的 PPO 训练目标，不对旧价值、Ref 或奖励反向传播。
        with torch.no_grad():
            critic_for_rollout = critic_model.module if isinstance(critic_model, DistributedDataParallel) else critic_model
            values_seq = critic_for_rollout(input_ids=gen_out, attention_mask=full_mask)
            # V_t 对齐到预测回答 token 的位置，与 old_resp_logp 一一对应。
            old_resp_values = values_seq.gather(1, logp_pos) * resp_value_mask

            # Ref 从 SFT 起点加载并冻结，给 KL 正则项提供“不偏离起始策略太远”的基准。
            ref_resp_logp = F.log_softmax(ref_model(input_ids=gen_out, attention_mask=full_mask).logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)
            token_rewards = torch.zeros_like(old_resp_logp)
            # 外部奖励只放在回答的最后一个有效 token；空回答不写入奖励。
            last_idx = resp_lengths - 1  # [B]
            token_rewards[torch.arange(B, device=args.device)[valid_resp], last_idx[valid_resp]] += rewards[valid_resp]

            # 反向递推 GAE：δ_t = r_t + γ V_(t+1) - V_t；A_t = δ_t + γλ A_(t+1)。
            # 这里末尾的下一状态价值视为 0，之后用掩码忽略无效回答位置。
            gen_len = old_resp_values.size(1); lastgaelam = torch.zeros(B, device=args.device); advs_rev = []
            for t in reversed(range(gen_len)):
                nv = old_resp_values[:, t + 1] if t < gen_len - 1 else 0.0
                delta = token_rewards[:, t] + args.gamma * nv - old_resp_values[:, t]
                lastgaelam = delta + args.gamma * args.lam * lastgaelam
                advs_rev.append(lastgaelam)
            advantages = torch.stack(advs_rev[::-1], dim=1)  # Actor 的优势目标 [B, R]
            returns = advantages + old_resp_values  # Critic 的回归目标 [B, R]

            # 仅用有效 token 统计均值和方差；标准化优势使策略梯度的尺度更稳定。
            adv_mean = (advantages * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
            adv_var = ((advantages - adv_mean) ** 2 * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
            advantages = (advantages - adv_mean) * torch.rsqrt(adv_var + 1e-8) * resp_policy_mask

        # 同一批在线样本可重复训练数轮；每轮重新打乱后切为小批次。
        mb_size = max(1, min(args.mini_batch_size, B))
        stop_ppo = False
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        kl_sum = 0.0
        kl_ref_sum = 0.0
        clipfrac_sum = 0.0
        aux_loss_sum = 0.0
        log_count = 0
        for ppo_epoch in range(args.ppo_update_iters):
            if stop_ppo:
                break
            b_inds = torch.randperm(B, device=args.device)
            for i in range(0, B, mb_size):
                inds = b_inds[i:i + mb_size]

                # Critic 重新前向以获得带梯度的新 V；旧 V 和 returns 都是固定目标。
                mb_values_seq = critic_model(input_ids=gen_out[inds], attention_mask=full_mask[inds])
                mb_resp_values = mb_values_seq.gather(1, logp_pos[inds])

                with autocast_ctx:
                    # Actor 计算 π_new；MoE 的辅助损失来自模型内部路由负载均衡。
                    res = actor_model(input_ids=gen_out[inds], attention_mask=full_mask[inds])
                    aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
                    # 先沿词表维取“实际生成 token”的 logp，再沿序列维取回答位置。
                    mb_resp_logp = F.log_softmax(res.logits[:, :-1], dim=-1).gather(2, labels[inds].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos[inds])

                # log_ratio = log π_new - log π_old；指数化后是 PPO 的重要性采样比率。
                log_ratio = mb_resp_logp - old_resp_logp[inds]

                # 可开关的诊断：观察首轮首个 minibatch 的 mb 与 old logp 差异。
                if args.debug_log_ratio and ppo_epoch == 0 and i == 0 and is_main_process():
                    _lr = log_ratio.detach()
                    _m = resp_policy_mask[inds].bool()
                    if _m.any():
                        _lrv = _lr[_m]
                        Logger(f"[DBG log_ratio] step={step} max|lr|={_lrv.abs().max().item():.6e} "
                               f"mean|lr|={_lrv.abs().mean().item():.6e} "
                               f"ratio_max={torch.exp(_lrv).max().item():.6f} "
                               f"ratio_min={torch.exp(_lrv).min().item():.6f} "
                               f"dropout={getattr(lm_config, 'dropout', None)} "
                               f"training={actor_model.training}")
                # 这个近似 KL 比较当前策略与本次采样策略，用于判断样本是否已太“旧”。
                approx_kl = (0.5 * (log_ratio ** 2) * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)

                # 同步各卡的 approx_kl，防止某卡 break 而其它卡继续导致 DDP 死锁
                approx_kl_val = approx_kl.detach().clone()
                if dist.is_initialized():
                    dist.all_reduce(approx_kl_val, op=dist.ReduceOp.AVG)
                    
                if approx_kl_val > args.early_stop_kl:
                    stop_ppo = True

                ratio = torch.exp(log_ratio)
                # clipfrac 是超出 [1-ε, 1+ε] 的有效 token 比例，仅用于监控裁剪程度。
                clipfrac = ((((ratio - 1.0).abs() > args.clip_epsilon).float() * resp_policy_mask[inds]).sum()
                            / resp_policy_mask[inds].sum().clamp(min=1))
                # 这是当前策略与冻结 Ref 的逐 token KL 估计，区别于上面的 approx_kl。
                # 写成 exp(ref-new)-(ref-new)-1，可以直接用已生成 token 的 logp 计算。
                kl_ref_penalty = ((torch.exp(ref_resp_logp[inds] - mb_resp_logp) - (ref_resp_logp[inds] - mb_resp_logp) - 1.0)
                                  * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)
                # 最小化损失等价于最大化 min(ratio*A, clip(ratio)*A)，再加 Ref KL 约束。
                # mask 保证 prompt、补齐位置以及首个 EOS 之后的 token 都不贡献损失。
                policy_loss = ((torch.max(-advantages[inds] * ratio,
                                          -advantages[inds] * torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon))
                               * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)
                               + args.kl_coef * kl_ref_penalty)
                # Critic 使用裁剪后的价值预测作保守更新：取原始误差和裁剪误差中较大者。
                # 这里乘 0.5 是常见的平方误差系数，vf_coef 在总损失中进一步调权。
                value_loss = 0.5 * (torch.max((mb_resp_values - returns[inds]) ** 2,
                                              (torch.clamp(mb_resp_values, old_resp_values[inds] - args.cliprange_value,
                                                           old_resp_values[inds] + args.cliprange_value) - returns[inds]) ** 2)
                                    * resp_value_mask[inds]).sum() / resp_value_mask[inds].sum().clamp(min=1)

                kl = approx_kl_val
                kl_ref = kl_ref_penalty.detach()

                # 早停时必须保证 forward-backward 闭环，故只截断 loss 不中断 DDP 通信
                if stop_ppo:
                    loss = (policy_loss + args.vf_coef * value_loss + aux_loss) * 0.0
                else:
                    # 小批次梯度先累积，达到 accumulation_steps 后才更新一次优化器。
                    loss = (policy_loss + args.vf_coef * value_loss + aux_loss) / args.accumulation_steps
                
                loss.backward()

                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                kl_sum += kl.item()
                kl_ref_sum += kl_ref.item()
                clipfrac_sum += clipfrac.item()
                aux_loss_sum += aux_loss.item()
                log_count += 1

                grad_accum_step += 1

                if grad_accum_step % args.accumulation_steps == 0:
                    # 限制梯度范数，随后分别更新 Actor/Critic 与各自的学习率调度器。
                    clip_grad_norm_(actor_model.parameters(), args.grad_clip)
                    clip_grad_norm_(critic_model.parameters(), args.grad_clip)
                    actor_optimizer.step()
                    critic_optimizer.step()
                    actor_scheduler.step()
                    critic_scheduler.step()
                    actor_optimizer.zero_grad()
                    critic_optimizer.zero_grad()

        # 本批最后不足 accumulation_steps 个小批次时，也把残余梯度提交。
        if grad_accum_step % args.accumulation_steps != 0:
            clip_grad_norm_(actor_model.parameters(), args.grad_clip)
            clip_grad_norm_(critic_model.parameters(), args.grad_clip)
            actor_optimizer.step()
            critic_optimizer.step()
            actor_scheduler.step()
            critic_scheduler.step()
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
        
        # 本地引擎只需持有新模型引用；SGLang 引擎按保存间隔把权重推送到推理服务。
        if step % args.save_interval == 0 or step == iters: rollout_engine.update_policy(actor_model)

        if is_main_process():
            # 训练指标按已处理的小批次数平均；reward 与回答长度按本批样本平均。
            critic_loss_val = value_loss_sum / max(log_count, 1)
            reward_val = rewards.mean().item()
            approx_kl_val = kl_sum / max(log_count, 1)
            kl_ref_val = kl_ref_sum / max(log_count, 1)
            clipfrac_val = clipfrac_sum / max(log_count, 1)
            avg_len_val = resp_lengths.float().mean().item()
            actor_lr, critic_lr = actor_optimizer.param_groups[0]['lr'], critic_optimizer.param_groups[0]['lr']

            if wandb is not None:
                wandb.log({
                    "reward": reward_val,
                    "kl_ref": kl_ref_val,
                    "approx_kl": approx_kl_val,
                    "clipfrac": clipfrac_val,
                    "critic_loss": critic_loss_val,
                    "avg_response_len": avg_len_val,
                    "actor_lr": actor_lr,
                    "critic_lr": critic_lr,
                })

            Logger(f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                   f"Reward: {reward_val:.4f}, KL_ref: {kl_ref_val:.4f}, Approx KL: {approx_kl_val:.4f}, "
                   f"ClipFrac: {clipfrac_val:.4f}, Critic Loss: {critic_loss_val:.4f}, "
                   f"Avg Response Len: {avg_len_val:.2f}, Actor LR: {actor_lr:.8f}, Critic LR: {critic_lr:.8f}")

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            # 纯 Actor 权重用于下一阶段/推理；完整断点还保存 Critic、优化器和调度器。
            actor_model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_actor = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            raw_actor = getattr(raw_actor, '_orig_mod', raw_actor)
            actor_state = raw_actor.state_dict()
            torch.save({k: v.half().cpu() for k, v in actor_state.items()}, ckp)
            
            # 断点 step 是当前 epoch 已处理的批次数，供 SkipBatchSampler 恢复时跳过。
            lm_checkpoint(lm_config, weight=args.save_weight, model=actor_model, optimizer=actor_optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints',
                         scheduler=actor_scheduler, critic_model=critic_model, 
                         critic_optimizer=critic_optimizer, critic_scheduler=critic_scheduler)
            actor_model.train()
            del actor_state

        # 显式删除本次 rollout 的大张量引用，避免下批次继续持有中间结果。
        del enc, gen_out, completion_ids, responses_text, rewards, full_mask, values_seq, advantages
        del labels, resp_labels, resp_idx, resp_pad_mask, valid_resp, eos_mask, has_eos, eos_pos, resp_lengths, resp_policy_mask, resp_value_mask, old_resp_logp, ref_resp_logp
        del kl, kl_ref, policy_loss, value_loss, loss, token_rewards, returns, old_resp_values, prompt_lens, logp_pos


if __name__ == "__main__":
    # 参数按用途理解：采样规模/长度、PPO 目标、权重与奖励来源、运行环境。
    # 命令行选项覆盖默认值；相对路径相对于运行命令时的工作目录。
    parser = argparse.ArgumentParser(description="MiniMind PPO (Proximal Policy Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='ppo_actor', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="Actor学习率")
    parser.add_argument("--critic_learning_rate", type=float, default=5e-7, help="Critic学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_seq_len', default=768, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1024, help="生成的最大长度")
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif.jsonl", help="RLAIF数据路径")
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO裁剪参数")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function系数")
    parser.add_argument("--kl_coef", type=float, default=0.02, help="KL散度惩罚系数")
    parser.add_argument("--gamma", type=float, default=1.0, help="GAE折扣因子")
    parser.add_argument("--lam", type=float, default=0.95, help="GAE lambda参数")
    parser.add_argument("--cliprange_value", type=float, default=0.2, help="Value function裁剪范围")
    parser.add_argument("--ppo_update_iters", type=int, default=2, help="同一批rollout重复更新次数")
    parser.add_argument("--early_stop_kl", type=float, default=0.25, help="PPO early stop 的 KL 阈值")
    parser.add_argument("--mini_batch_size", type=int, default=2, help="PPO每次更新的minibatch大小")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-PPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--debug_mode", action="store_true", help="是否打印训练调试采样")
    parser.add_argument("--debug_interval", type=int, default=20, help="debug模式下每隔多少step打印一次采样")
    parser.add_argument("--debug_log_ratio", action="store_true", help="打印首轮首个minibatch的log_ratio差异量级，用于核查ratio≈1是否成立")
    parser.add_argument("--thinking_ratio", type=float, default=0.9, help="按概率开启thinking（0.0~1.0）")
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"], help="rollout引擎类型")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998", help="SGLang服务器URL")
    parser.add_argument("--sglang_model_path", type=str, default="../model", help="SGLang tokenizer路径")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_ppo", help="SGLang共享存储路径")
    args = parser.parse_args()

    # ========== 1. 初始化运行环境和随机种子 ==========
    # torchrun 启动时每个进程负责一张卡；普通 python 启动则是单进程。
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型结构与可选续训断点 ==========
    # 续训断点带 Actor/Critic 和优化状态；from_weight 只指定起始 Actor 权重。
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    # CPU 使用普通前向；CUDA 在 Actor 前向时按 dtype 启用自动混合精度。
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配置实验日志 ==========
    # use_wandb 选项在本仓库通过 SwanLab 的兼容接口记录指标；只由主进程写入。
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-PPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 初始化模型、数据与优化器 ==========
    base_weight = args.from_weight
    # Actor 和 Ref 从同一份 SFT 权重起步；Ref 在整个 PPO 过程中保持冻结。
    actor_model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    moe_suffix = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/{base_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    # Critic 共享“结构和初始主干权重”，但训练时是独立模型，新增 value_head 从头学习。
    # strict=False 容许原 Actor 权重与 Critic 头的参数名不完全一致。
    critic_model = CriticModel(lm_config)
    critic_model.load_state_dict(state_dict, strict=False)
    critic_model = critic_model.to(args.device)
    # 奖励模型只做推理，不放进优化器；它与 Critic 的职责不同。
    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
    # rollout 引擎负责在线生成及旧策略 logp；默认是当前进程内的 PyTorch 引擎。
    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine,
        policy_model=actor_model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )
    # RLAIFDataset 去掉样本中的最后一条参考答案，返回等待 Actor 续写的 prompt。
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=(args.max_seq_len + args.max_gen_len), thinking_ratio=args.thinking_ratio)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    actor_optimizer = optim.AdamW(actor_model.parameters(), lr=args.learning_rate)
    critic_optimizer = optim.AdamW(critic_model.parameters(), lr=args.critic_learning_rate)
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    # 用预计的优化器步数设定余弦退火长度：每批有若干轮 PPO、若干 mini-batch，
    # 再除以梯度累积数。实际步数可能因 KL 早停或末批大小而不同。
    mb_factor = max(1, math.ceil(args.batch_size / args.mini_batch_size))
    total_optimizer_steps = math.ceil(iters * args.epochs * args.ppo_update_iters * mb_factor / args.accumulation_steps)
    actor_scheduler = CosineAnnealingLR(actor_optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    critic_scheduler = CosineAnnealingLR(critic_optimizer, T_max=total_optimizer_steps, eta_min=args.critic_learning_rate / 10)

    # ========== 6. 若存在断点，恢复全部训练状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        actor_model.load_state_dict(ckp_data['model'])
        critic_model.load_state_dict(ckp_data['critic_model'])
        actor_optimizer.load_state_dict(ckp_data['optimizer'])
        critic_optimizer.load_state_dict(ckp_data['critic_optimizer'])
        actor_scheduler.load_state_dict(ckp_data['scheduler'])
        critic_scheduler.load_state_dict(ckp_data['critic_scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    # 在加载断点之后包装，避免保存时的参数名前缀与普通模型不一致。
    if args.use_compile == 1:
        actor_model = torch.compile(actor_model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        # freqs_cos/freqs_sin 各 rank 由 config 确定性算出，默认每步广播一次纯属浪费
        actor_model = DistributedDataParallel(actor_model, device_ids=[local_rank], broadcast_buffers=False)
        critic_model = DistributedDataParallel(critic_model, device_ids=[local_rank], broadcast_buffers=False)
    rollout_engine.update_policy(actor_model)
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # 分布式采样器按 epoch 改变顺序；单卡用固定 epoch 种子产生排列。
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        # 断点记录上次保存的批次编号，仅首个恢复的 epoch 需要跳过这些批次。
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            ppo_train_epoch(epoch, loader, len(loader) + skip, rollout_engine, ref_model, actor_scheduler, critic_scheduler, reward_model, start_step, wandb)
        else:
            ppo_train_epoch(epoch, loader, len(loader), rollout_engine, ref_model, actor_scheduler, critic_scheduler, reward_model, 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
