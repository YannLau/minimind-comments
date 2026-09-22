"""MiniMind 全参数监督微调（Full SFT）入口。

建议在阅读本文件前，先看以下代码：
1. ``trainer/train_pretrain.py``：本文件通常从预训练权重出发，而训练循环也与预训练
   高度相似；先建立“语言模型如何预测下一个 token”的整体认识会更容易理解 SFT。
2. ``dataset/lm_dataset.py`` 中的 ``SFTDataset``：重点理解聊天模板如何把多轮消息拼成
   token 序列，以及 labels 为什么只保留 assistant 回复、其余位置均设为 ``-100``。
3. ``model/model_minimind.py`` 中的 ``MiniMindConfig`` 和
   ``MiniMindForCausalLM.forward``：理解 logits、next-token 交叉熵 ``loss``，以及
   MoE 模型额外返回的路由均衡 ``aux_loss``。
4. ``trainer/trainer_utils.py`` 中的 ``get_lr``、``init_model``、``lm_checkpoint``、
   ``init_distributed_mode`` 和 ``SkipBatchSampler``：这些工具分别负责学习率调度、
   初始权重加载、断点保存/恢复、DDP 初始化，以及续训时跳过已完成的批次。

读完本文件后，推荐继续阅读：
1. ``eval_llm.py``：观察这里保存的 ``full_sft_*.pth`` 如何被加载并用于多轮对话。
2. ``trainer/train_lora.py``：比较“更新全部参数”和“只训练 LoRA 低秩参数”的差异。
3. ``trainer/train_dpo.py``：了解 SFT 之后如何用 chosen/rejected 偏好对继续对齐模型。
4. ``trainer/train_distillation.py``：了解如何让学生模型学习教师模型的概率分布。

整体流程是：解析参数 -> 初始化单卡/DDP 环境 -> 从预训练权重创建模型 -> 将对话数据
编码为只监督 assistant 的定长样本 -> 按 epoch 训练（混合精度、梯度累积、裁剪、更新）
-> 定期保存阶段权重和可续训检查点。

“全参数”表示优化器接收 ``model.parameters()``，模型中所有可训练参数都会更新；它与
LoRA 这类参数高效微调相对。脚本中的 ``args``、``model``、``optimizer`` 等对象在主
程序中创建，``train_epoch`` 直接读取这些全局对象，因此本文件适合作为命令行入口，
而不是被当作独立训练库调用。
"""

import os
import sys

# 将当前脚本声明为 trainer 包内模块，并把仓库根目录加入模块搜索路径。
# 因此直接执行本文件时，也能正确导入同仓库的 model、dataset 和 trainer 模块。
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 虽然下文没有直接使用 datasets，但必须先于 torch 导入，以规避 Windows 下
# pyarrow 与 torch 的 DLL 加载冲突（项目 issue #771）；F401 表示忽略“未使用导入”。
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
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """训练一个 epoch，或从断点处训练该 epoch 尚未完成的部分。

    参数：
        epoch: 从 0 开始的 epoch 编号；日志展示时会加 1。
        loader: 当前进程的 DataLoader。每批返回 ``(input_ids, labels)``，形状均为
            ``[batch_size, max_seq_len]``；labels 中的 ``-100`` 不参与损失。
        iters: 当前 epoch 的完整微批次数。续训时传入“剩余批数 + 已跳过批数”，
            以便日志、学习率和保存间隔继续使用中断前的 step 编号。
        start_step: 当前 epoch 已完成的微批次数；非续训场景为 0。
        wandb: 实际是采用 wandb 风格接口的 SwanLab 模块；为 ``None`` 时不上传指标。

    这里的 ``step`` 是微批次编号。只有积累 ``accumulation_steps`` 个微批次后才更新
    一次参数，所以它不一定等于优化器更新次数。单进程的近似有效批大小为
    ``batch_size * accumulation_steps``；DDP 下还需再乘进程数（GPU 数）。
    """
    # 计时从本次实际执行的位置开始；last_step 用于判断循环结束后是否还有残余梯度。
    start_time = time.time()
    last_step = start_step
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # DataLoader 在 CPU 端产出张量，训练前把输入和标签搬到当前进程所绑定的设备。
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step
        # 把 epoch 和批次编号展平为全局训练进度，get_lr 据此执行余弦退火。
        # 遍历 param_groups 可兼容今后为不同参数设置不同优化器分组的情况。
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # GPU 上 autocast 会让适合的算子用 BF16/FP16 执行，以节省显存并加速；
        # CPU 上 autocast_ctx 是空上下文，计算精度不作自动转换。
        with autocast_ctx:
            res = model(input_ids, labels=labels)
            # res.loss 是只在 assistant 标签处计算的 next-token 交叉熵。
            # 稠密模型的 aux_loss 为 0；MoE 模型用它鼓励各专家获得更均衡的 token。
            loss = res.loss + res.aux_loss
            # 每个微批次贡献 1/N 的梯度，连续反传 N 次后近似得到大批次平均梯度。
            loss = loss / args.accumulation_steps

        # GradScaler 仅在 FP16 模式真正缩放损失；BF16/CPU 下仍可复用同一套调用流程。
        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            # 梯度裁剪前先撤销 FP16 的缩放，否则 grad_clip 阈值会被错误地应用于放大值。
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # FP16 梯度若含 inf/NaN，scaler.step 会跳过更新；update 会调整后续缩放因子。
            scaler.step(optimizer)
            scaler.update()

            # 设为 None 通常比把梯度张量逐元素清零更省显存，下一次反传时再按需创建。
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            # loss 在反传前除过累积步数，此处乘回去以显示当前微批次的原始总损失。
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            # 总损失 = 语言建模损失 + MoE 辅助损失，二者相减得到便于观察的主损失。
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            # 用本次运行已经完成的批次平均耗时，估算当前 epoch 剩余分钟数。
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            # 只有主进程会创建日志实例，所以 DDP 不会从每个 rank 重复上报相同指标。
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            # 仅 rank 0 写文件，避免多个 DDP 进程同时覆盖同一路径。
            # eval() 会关闭 dropout；保存后恢复 train()，继续保持正常训练行为。
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # DDP 与 torch.compile 都会包裹真实模型；逐层解包可让权重键名保持干净。
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            # save_dir 下保存 FP16 CPU 权重，文件较小，可供后续训练阶段或推理直接加载。
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            # checkpoints 下另存续训文件，其中还包含优化器、scaler、epoch、step、
            # world_size 和实验 ID；它服务于 --from_resume 1，与上面的阶段权重用途不同。
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scaler=scaler)
            model.train()
            del state_dict

        # 及时释放本批大张量的 Python 引用，便于后续迭代复用显存。
        del input_ids, labels, res, loss

    # 若批次数不是 accumulation_steps 的整数倍，循环内最后一组梯度尚未触发更新。
    # 此处补做一次更新，避免丢掉尾部样本；这组梯度规模会小于完整累积组。
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    # 只有直接运行本文件才启动训练；被其他模块导入时只定义 train_epoch。
    parser = argparse.ArgumentParser(description="MiniMind 全参数监督微调")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='full_sft', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="每个进程每个微批次的样本数")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型：bfloat16 或 float16")
    parser.add_argument("--num_workers", type=int, default=8, help="每个进程的数据加载子进程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="多少个微批次累积后更新一次参数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="每隔多少个微批次打印一次日志")
    parser.add_argument("--save_interval", type=int, default=1000, help="每隔多少个微批次保存一次模型")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=768, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--seed', default=42, type=int, help="随机种子（DDP下每个rank为seed+rank，每轮为seed+epoch）")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_t2t_mini.jsonl", help="训练数据路径")
    parser.add_argument('--from_weight', default='pretrain', type=str, help="初始权重名称；none 表示随机初始化，默认从预训练权重开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    # 普通 python 启动时不初始化进程组，local_rank 为 0；torchrun 启动时读取
    # RANK/LOCAL_RANK，用 NCCL 建立进程组，并让每个进程绑定一张 GPU。
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    # 不同 rank 使用不同种子，避免各 GPU 上数据增强等随机行为完全相同。
    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查断点 ==========
    os.makedirs(args.save_dir, exist_ok=True)
    # 未显式传入的词表、注意力头数、中间层宽度等参数采用 MiniMindConfig 默认值。
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    # --from_resume 读取 checkpoints/*_resume.pth；不存在时返回 None，并自然从头训练。
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    # 当前约定：字符串恰为 bfloat16 时使用 BF16，其他值均按 FP16 处理。
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # CPU 训练不进入 CUDA autocast；GPU 前向与损失计算则自动选择指定低精度。
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配置实验日志 ==========
    wandb = None
    if args.use_wandb and is_main_process():
        # 项目使用 SwanLab，但保留变量名 wandb，以沿用熟悉的 init/log 接口风格。
        import swanlab as wandb
        # 断点中若记录了实验 ID，则续写原实验；否则创建新的实验记录。
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Full-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    # 默认从 ../out/pretrain_{hidden_size}[ _moe].pth 加载预训练参数；若 from_weight
    # 为 none，则随机初始化。tokenizer 默认从 ../model 目录读取。
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    # SFTDataset 使用 tokenizer 的 chat_template 渲染多轮对话，并只在 assistant 回复区域
    # 保留监督标签；system、user、工具描述和 padding 位置均为 -100，不计入交叉熵。
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # DDP 下为每个 rank 划分不同数据，避免每张 GPU 重复遍历完整数据集。
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # FP16 动态范围较小，需要动态 loss scaling；BF16 指数范围较大，通常无需缩放。
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    # 把模型全部参数交给 AdamW，正是“全参数微调”；这里没有冻结任何层。
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从断点恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        # 同时恢复模型、AdamW 动量和 GradScaler，尽量延续中断前的优化轨迹。
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        # 仍从保存时所在 epoch 开始，后面由 SkipBatchSampler 跳过已经完成的批次。
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        # torch.compile 首轮需捕获和编译计算图，可能较慢，稳定后通常能提升吞吐。
        model = torch.compile(model)
        Logger('已启用 torch.compile')
    if dist.is_initialized():
        # DDP 会在反向传播时同步各 rank 梯度；每个进程只操作 local_rank 对应的 GPU。
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # DDP 采样器根据 epoch 改变每轮洗牌；单卡则在下方显式生成随机索引顺序。
        train_sampler and train_sampler.set_epoch(epoch)
        # 固定每轮种子，使断点重启后能重建同一数据顺序并准确跳过旧批次。
        setup_seed(args.seed + epoch); indices = torch.randperm(len(train_ds)).tolist()
        # 只在恢复后的第一个 epoch 跳过 start_step，之后的 epoch 都从第 1 批开始。
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        # 按 batch_size 组成批次，并丢弃前 skip 批；末尾不足一整批的样本仍会保留。
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        # 指定 batch_sampler 后不再传 batch_size/shuffle；pin_memory 可加快 CPU 到 GPU 拷贝。
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            # len(loader) 只统计剩余批次，加回 skip 才是完整 epoch 的批次数。
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        # 等待所有 rank 结束后再销毁进程组，避免先退出者打断其他进程的通信。
        dist.barrier()
        dist.destroy_process_group()
