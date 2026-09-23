"""用 DPO（直接偏好优化）训练 MiniMind 的偏好对齐脚本。

建议先读：
1. ``README.md`` 的“直接偏好优化（DPO）”小节：先认识 chosen、rejected、
   策略模型、参考模型和损失公式。
2. ``model/model_minimind.py`` 的 ``MiniMindForCausalLM.forward``：了解模型如何
   把输入 token 变为每个位置的词表 logits，以及位置 t 预测位置 t+1 的约定。
3. ``dataset/lm_dataset.py`` 的 ``DPODataset``：了解偏好对如何渲染成聊天文本、
   截断/补齐，并产生已错位的 x、y 与仅覆盖 assistant 回复的 mask。
4. ``trainer/trainer_utils.py`` 的 ``init_model``、``get_lr``、``lm_checkpoint``
   和 ``SkipBatchSampler``：了解起始权重、学习率、保存与断点跳过。

读完可继续看 ``trainer/train_full_sft.py``（偏好训练前的监督微调），以及
``trainer/train_ppo.py``（在线生成、显式奖励和 Critic），比较两种对齐训练流程。

阅读约定：B 为一批偏好对的数量，T 为截断/补齐后的序列长度减一；
π 表示可训练策略模型，ref 表示冻结的参考模型。偏好对通常共享问题，
chosen 是偏好回复，rejected 是较差回复。这里用固定偏好数据，不进行在线生成。
"""

import os
import sys

# 允许从 trainer/ 目录直接运行，并让绝对导入找到仓库根目录下的模块。
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # 先导入 datasets，规避 Windows 上 pyarrow/torch 的 DLL 冲突（#771）。
import argparse
import time
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import DPODataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def logits_to_log_probs(logits, labels):
    """从完整词表分布中取出真实目标 token 的逐位置对数概率。

    logits 为 [2B, T, 词表大小]，labels 为 [2B, T]，返回 [2B, T]。
    ``DPODataset`` 已把 y 相对 x 左移一个 token，故同一位置可以直接 gather。
    """
    # 先沿词表维归一化，再沿词表维取出目标 token 的 log P(token | 前文)。
    log_probs = F.log_softmax(logits, dim=2)
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return log_probs_per_token


def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    """按偏好对计算 DPO 损失；输入均按 chosen 在前、rejected 在后排列。

    两组 log_probs 与 mask 的形状都是 [2B, T]。对每条回复求和后，
    损失为 -log σ(β[(log π_chosen - log π_rejected)
                    - (log ref_chosen - log ref_rejected)])，最后对 B 对求平均。
    """
    # 只汇总 assistant 回复及其结束标记；问题、模板 token 和 padding 不参与比较。
    # 这里是 token 对数概率之和，即整条回复的对数概率，没有按长度做平均。
    ref_log_probs = (ref_log_probs * mask).sum(dim=1)
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)

    # 训练循环把 B 条 chosen 与 B 条 rejected 沿 batch 维拼接，因此按半数切回配对。
    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
    reject_ref_log_probs = ref_log_probs[batch_size // 2:]
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
    reject_policy_log_probs = policy_log_probs[batch_size // 2:]

    # 分别计算策略和参考模型认为“好回答相对差回答”有多可能。
    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    # 正值表示策略模型相对参考模型更偏向 chosen；beta 控制偏好信号的尺度。
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()


def train_epoch(epoch, loader, iters, ref_model, lm_config, start_step=0, wandb=None, beta=0.1):
    """遍历一个数据 epoch，按偏好对计算梯度，并定期记录与保存。

    ``step`` 是当前 epoch 的数据批次编号；恢复训练时从 ``start_step+1`` 继续。
    model、optimizer、scaler 和 args 由主程序初始化后供本函数使用。
    """
    start_time = time.time()

    for step, batch in enumerate(loader, start=start_step + 1):
        # DPODataset 为每个偏好对分别提供“好/差”完整对话；x、y 已错开一个 token。
        # 每个张量形状为 [B, T]；mask=1 只在待比较的 assistant 回复位置。
        x_chosen = batch['x_chosen'].to(args.device)
        x_rejected = batch['x_rejected'].to(args.device)
        y_chosen = batch['y_chosen'].to(args.device)
        y_rejected = batch['y_rejected'].to(args.device)
        mask_chosen = batch['mask_chosen'].to(args.device)
        mask_rejected = batch['mask_rejected'].to(args.device)
        # 一次前向处理两种回复：[chosen_0..B-1, rejected_0..B-1]。
        # 这个排列必须与 dpo_loss 中“前半/后半配对”的假设一致。
        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        # get_lr 使用余弦衰减，无预热；每批把学习率写入优化器参数组。
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            # 参考模型冻结且不保留反向图，它给出偏好比较的固定基线。
            with torch.no_grad():
                ref_outputs = ref_model(x)
                ref_logits = ref_outputs.logits
            ref_log_probs = logits_to_log_probs(ref_logits, y)

            # 策略模型保持可训练；同一组 x/y 使两模型分数可以逐样本比较。
            outputs = model(x)
            logits = outputs.logits
            policy_log_probs = logits_to_log_probs(logits, y)

            # aux_loss 是 MiniMind 的 MoE 辅助损失；稠密模型时该值为零。
            dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
            loss = dpo_loss_val + outputs.aux_loss
            # 累积若干批的梯度，再统一执行一次优化器更新。
            loss = loss / args.accumulation_steps

        # GradScaler 仅在 float16 配置下启用；仍用同一接口处理其他精度。
        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0 or step == iters:
            # 裁剪前先取消 loss scaling，否则梯度范数不对应真实尺度。
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            # 把日志里的 loss 还原为除以 accumulation_steps 之前的数值。
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_dpo_loss = dpo_loss_val.item()
            current_aux_loss = outputs.aux_loss.item()
            current_lr = optimizer.param_groups[-1]['lr']
            # 用当前 epoch 已处理批次的平均耗时估算剩余分钟数。
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, dpo_loss: {current_dpo_loss:.4f}, aux_loss: {current_aux_loss:.4f}, learning_rate: {current_lr:.8f}, epoch_time: {eta_min:.3f}min')
            
            if wandb: wandb.log({"loss": current_loss, "dpo_loss": current_dpo_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            # 纯策略权重用于推理或下一训练阶段；完整断点还含优化器和 GradScaler。
            # 仅主进程写文件，避免 DDP 多进程同时覆盖同一路径。
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            # step 保存的是 epoch 内批次编号，续训时可跳过已处理的批次。
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            del state_dict

        # 释放当前批次的大张量引用，为下一批前向腾出内存。
        del x_chosen, x_rejected, y_chosen, y_rejected, mask_chosen, mask_rejected, x, y, mask
        del ref_outputs, ref_logits, ref_log_probs, outputs, logits, policy_log_probs, loss


if __name__ == "__main__":
    # 命令行参数分为模型/数据、优化、日志保存和运行环境几组。
    # 相对路径以执行命令时的工作目录为基准；README 的示例从 trainer/ 目录运行。
    parser = argparse.ArgumentParser(description="MiniMind DPO (Direct Preference Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='dpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=4e-8, help="初始学习率（建议<=5e-8避免遗忘）")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/dpo.jsonl", help="DPO训练数据路径")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument('--beta', default=0.15, type=float, help="DPO中的beta参数")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-DPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    # torchrun 提供 rank 信息时初始化 DDP；普通 python 启动则用单进程。
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型结构、查找可选续训断点 ==========
    # from_weight 是起点模型权重；from_resume 读取本阶段的完整训练状态。
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    # CPU 不使用 CUDA 自动混合精度；GPU 按 dtype 在前向中使用自动混合精度。
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配置实验日志 ==========
    # use_wandb 通过 SwanLab 的兼容接口写指标；只在主进程初始化实验。
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-DPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 初始化策略模型、参考模型、数据与优化器 ==========
    # 两个模型加载同一份 SFT 权重。训练过程中只更新策略模型，参考模型始终固定。
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    Logger(f'策略模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')
    # 参考模型不在优化器中，并关闭梯度计算所需的参数标记。
    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    ref_model.eval()
    ref_model.requires_grad_(False)
    Logger(f'参考模型总参数量：{sum(p.numel() for p in ref_model.parameters()) / 1e6:.3f} M')
    
    # 每条数据包含一对 chosen/rejected；分布式时各 rank 负责不同样本。
    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # 当前代码仅根据 dtype 是否为 float16 决定是否启用梯度缩放。
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 如有断点，恢复策略模型、优化器、缩放器及已完成批次数 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    # 在恢复参数之后包装，避免保存/加载时出现额外参数名前缀。
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # 分布式采样器逐 epoch 变更排序；单进程则用固定种子生成随机下标。
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        # 首个恢复的 epoch 跳过断点前已训练的批次；后续 epoch 从头遍历。
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, ref_model, lm_config, start_step, wandb, args.beta)
        else:
            train_epoch(epoch, loader, len(loader), ref_model, lm_config, 0, wandb, args.beta)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
