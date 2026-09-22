"""MiniMind 预训练入口：用普通文本训练一个从左到右预测下一 token 的语言模型。

建议在阅读本文件前，先看以下代码：
1. ``dataset/lm_dataset.py`` 中的 ``PretrainDataset``：理解一条 JSONL 文本如何被分词、
   添加 BOS/EOS、补齐到固定长度，以及为什么 padding 对应的标签会被设为 ``-100``。
2. ``model/model_minimind.py`` 中的 ``MiniMindConfig``、``MiniMindForCausalLM.forward``：
   理解模型配置、logits、next-token 交叉熵 ``loss`` 与 MoE ``aux_loss`` 从何而来。
3. ``trainer/trainer_utils.py`` 中的 ``get_lr``、``init_model``、``lm_checkpoint``、
   ``init_distributed_mode`` 和 ``SkipBatchSampler``：这些工具负责学习率、模型加载、
   断点保存/恢复、DDP 初始化，以及续训时跳过已经完成的批次。

读完本文件后，推荐继续阅读：
1. ``trainer/train_full_sft.py``：它与本文件的训练骨架几乎相同，但数据集只监督
   assistant 回复，可直观看出“预训练”和“监督微调”的区别。
2. ``eval_llm.py``：了解这里保存的 ``pretrain_*.pth`` 权重如何被加载并用于生成。
3. ``trainer/train_lora.py``、``trainer/train_dpo.py``：继续了解参数高效微调与偏好对齐。

整体流程可以概括为：解析参数 -> 初始化单卡/DDP 环境 -> 创建模型与数据集 ->
按 epoch 训练（混合精度、梯度累积、裁剪、更新）-> 定期保存推理权重和续训检查点。
本文件有意把脚本级对象（如 ``args``、``model``、``optimizer``）作为全局变量，
``train_epoch`` 会直接读取它们；因此它主要作为命令行入口使用，而不是通用训练库。
"""

import os
import sys

# 将当前文件声明为 trainer 包内模块，并把仓库根目录加入模块搜索路径。
# 这样无论通过 ``python train_pretrain.py`` 还是从仓库其他位置启动，都能导入 model、dataset。
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 虽然下文没有直接使用 datasets，但必须先于 torch 导入，以规避 Windows 下
# pyarrow 与 torch 的 DLL 加载冲突（对应项目 issue #771）；F401 表示忽略“未使用导入”。
import datasets  # noqa: F401
import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """训练一个 epoch（若从断点续训，则训练该 epoch 尚未完成的部分）。

    参数：
        epoch: 从 0 开始的 epoch 编号。
        loader: 当前进程使用的 DataLoader；每批返回 ``input_ids`` 和 ``labels``，
            二者形状都是 ``[batch_size, max_seq_len]``。
        iters: 该 epoch 的完整批次数。断点续训时它等于“剩余批数 + 已跳过批数”，
            因而日志、学习率与保存间隔仍沿用恢复前的 step 编号。
        start_step: 此 epoch 中已经完成的批次数，正常训练为 0。
        wandb: 实际传入的是兼容 wandb 接口的 SwanLab 模块；为 ``None`` 时不上传指标。

    注意：这里的 ``step`` 是“微批次（micro-batch）编号”，并不一定等于优化器更新次数；
    每累计 ``accumulation_steps`` 个微批次才执行一次参数更新。
    """
    # 用本 epoch 实际训练部分的起点估算剩余时间；last_step 用于处理末尾不足一组的累积梯度。
    start_time = time.time()
    last_step = start_step
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # DataLoader 默认在 CPU 产生张量；训练前将输入和监督标签搬到当前进程对应设备。
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step

        # get_lr 使用余弦退火，把当前 epoch/step 展平为全局进度。
        # 这里逐个参数组赋值，是为了兼容优化器未来包含多个 param_group 的情况。
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # GPU 上启用 autocast：大部分算子用 BF16/FP16 以降低显存并加速；CPU 上是空上下文。
        # res.loss 是 next-token 交叉熵；稠密模型的 aux_loss 为 0，MoE 模型则用它约束路由均衡。
        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            # 每个微批次只贡献 1/N 的梯度，累计 N 次后总体梯度近似一个更大的 batch。
            loss = loss / args.accumulation_steps

        # GradScaler 仅在 FP16 时真正缩放 loss；BF16 或 CPU 下该调用仍保持统一训练流程。
        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            # 裁剪前必须先还原被 GradScaler 放大的梯度，否则阈值 grad_clip 没有实际意义。
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 若 FP16 梯度出现 inf/NaN，scaler.step 会跳过本次更新；随后动态调整缩放因子。
            scaler.step(optimizer)
            scaler.update()

            # set_to_none=True 比把梯度清零更省内存，下次 backward 会按需重新创建梯度张量。
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            # 训练前 loss 除过 accumulation_steps，此处乘回去，日志才代表原始批次损失。
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            # 用已完成微批次的平均耗时估算本 epoch 剩余分钟数；// 60 得到整分钟近似值。
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            # 只有主进程会创建 wandb/SwanLab 实例，因此 DDP 不会重复上报同一份指标。
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            # 只让 rank 0 写文件，避免多个 DDP 进程同时覆盖同一路径。
            # eval() 会关闭 dropout，使保存期间模型状态明确；保存后再恢复 train()。
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # DDP 和 torch.compile 都会在真实模型外增加包装；保存前逐层解包，保持权重键名干净。
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            # out/ 下只保存半精度模型权重，体积较小，供后续 SFT 或推理通过 init_model 加载。
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            # checkpoints/ 下另存“续训检查点”：除模型外还包含优化器、scaler、epoch、step、
            # world_size 和实验 ID。它用于 --from_resume 1，不等同于上面的推理/阶段权重。
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            del state_dict

        # 尽早释放对本批大张量的 Python 引用，便于后续批次复用显存。
        del input_ids, labels, res, loss

    # 若一个 epoch 的微批次数不是 accumulation_steps 的整数倍，循环内不会更新最后一组梯度。
    # 在这里补做一次更新，避免丢掉尾部样本产生的梯度。其梯度规模会小于完整累积组。
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    # 只有直接运行本文件时才启动训练；被其他模块导入时只会定义 train_epoch。
    parser = argparse.ArgumentParser(description="MiniMind 预训练")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="每个进程每个微批次的样本数")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型：bfloat16 或 float16")
    parser.add_argument("--num_workers", type=int, default=8, help="每个进程的数据加载子进程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="多少个微批次累积后更新一次参数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="每隔多少个微批次打印一次日志")
    parser.add_argument("--save_interval", type=int, default=1000, help="每隔多少个微批次保存一次模型")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--seed', default=42, type=int, help="随机种子（DDP下每个rank为seed+rank，每轮为seed+epoch）")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    # 普通 python 启动时不初始化进程组，返回 local_rank=0；torchrun 启动时从环境变量读取
    # RANK/LOCAL_RANK，以 NCCL 建立进程组，并让每个进程绑定一张 GPU。
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    # 不同 rank 使用不同种子，避免各 GPU 上所有随机行为完全相同。
    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查断点 ==========
    os.makedirs(args.save_dir, exist_ok=True)
    # 未显式传入的词表大小、注意力头数、最大位置等参数使用 MiniMindConfig 的默认值。
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    # --from_resume 读取 checkpoints/*_resume.pth；找不到文件时返回 None，并自然地从头训练。
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    # 当前参数约定：字符串恰为 bfloat16 时用 BF16，其余值按 FP16 处理。
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # CPU 训练不进入 CUDA autocast；GPU 训练则在前向和 loss 计算期间自动选择低精度算子。
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配置实验日志 ==========
    wandb = None
    if args.use_wandb and is_main_process():
        # 项目使用 SwanLab，但别名仍叫 wandb，从而保持 wandb.init/log 风格的调用方式。
        import swanlab as wandb
        # 若续训检查点记录过实验 ID，则接着写入原实验；否则创建一个新实验。
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    # from_weight='none' 表示随机初始化；否则从 ../out/{from_weight}_{hidden_size}[ _moe].pth
    # 加载阶段权重。tokenizer 默认从 ../model 读取。
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    # 每个样本会形成定长 input_ids/labels；模型内部自动错位一位计算“预测下一个 token”。
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # DDP 下由 DistributedSampler 将样本划分给各 rank，防止每张 GPU 重复训练完整数据集。
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # FP16 动态范围较小，需要 GradScaler；BF16 指数范围较大，通常无需 loss scaling。
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从断点恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        # 完整恢复模型、优化器动量和混合精度缩放器，才能尽量延续中断前的训练轨迹。
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        # 保存发生在某个 epoch 的某个 step；稍后仍从该 epoch 开始，但跳过已完成批次。
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        # torch.compile 会先捕获/编译计算图，首轮可能较慢，后续迭代通常更快。
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        # DDP 在反向传播时自动对各 rank 的梯度做同步；device_ids 指定本进程使用的 GPU。
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # DistributedSampler 结合 epoch 改变每轮洗牌顺序；单卡路径则显式生成随机索引。
        train_sampler and train_sampler.set_epoch(epoch)
        # 每个 epoch 固定随机种子，使中断后重启仍能重建同一随机顺序，准确跳过旧批次。
        setup_seed(args.seed + epoch); indices = torch.randperm(len(train_ds)).tolist()
        # 只在恢复后的第一个 epoch 跳过 start_step；进入下一 epoch 后必须从第 1 批开始。
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        # SkipBatchSampler 先按 batch_size 组批，再丢弃前 skip 批；最后不足一批也会保留。
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        # 已经提供 batch_sampler，因此无需再传 batch_size/shuffle；pin_memory 可加快 CPU->GPU 拷贝。
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            # len(loader) 只含剩余批次，加回 skip 才是完整 epoch 的 step 总数。
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        # 先等待所有 rank 完成，避免某个进程提前退出并破坏其他进程尚在进行的通信。
        dist.barrier()
        dist.destroy_process_group()
