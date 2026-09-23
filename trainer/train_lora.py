"""使用 LoRA（Low-Rank Adaptation，低秩适配）对 MiniMind 做参数高效微调。

建议先阅读（有助于理解本文件）：
    1. ``README.md`` 的“LoRA”章节：先建立训练数据、启动命令、权重命名和推理方式的
       整体认识。
    2. ``trainer/train_full_sft.py``：本文件沿用了它的监督微调主流程；对照阅读可直观看到
       “全参数微调”和“只训练 LoRA 参数”的区别。
    3. ``dataset/lm_dataset.py`` 中的 ``SFTDataset``：理解对话如何由聊天模板编码为
       ``input_ids``，以及为何只有 assistant 回复对应的 ``labels`` 会参与损失。
    4. ``model/model_lora.py``：重点阅读 ``LoRA`` 和 ``apply_lora``，理解原线性层输出
       ``W x`` 如何变成 ``W x + B(A(x))``，以及这里只保存低秩矩阵 A、B 的原因。
    5. ``model/model_minimind.py`` 中的 ``MiniMindForCausalLM.forward``：理解 next-token
       交叉熵、``-100`` 标签掩码，以及 MoE 模型额外返回的 ``aux_loss``。
    6. ``trainer/trainer_utils.py``：理解本文件复用的学习率调度、DDP 初始化、基础权重
       加载、训练断点以及 ``SkipBatchSampler`` 的续训语义。

读完本文件后，推荐继续阅读：
    1. ``eval_llm.py``：查看推理时如何先加载基础模型，再挂载并载入 LoRA 增量权重。
    2. ``scripts/convert_model.py`` 中的 ``convert_merge_base_lora``：查看如何把 ``B @ A``
       合并回基础线性层，导出不再依赖 LoRA 分支的完整模型。
    3. ``trainer/train_dpo.py``：继续了解监督微调之后，如何利用 chosen/rejected 偏好对
       对齐模型行为。

本脚本的核心思路是：先载入已经训练好的基础模型，在符合条件的线性层旁动态挂载一个
低秩增量分支；随后冻结基础模型的所有参数，只把名称含 ``lora`` 的参数交给优化器。
因此前向传播仍使用“基础模型 + LoRA”的完整能力，反向传播却只更新很少的 A、B 矩阵。

常见启动方式（这些相对路径以 ``trainer/`` 为当前工作目录）：
    ``python train_lora.py``
    ``torchrun --nproc_per_node 2 train_lora.py``

注意：脚本大量使用模块级对象（如 ``args``、``model``、``optimizer``），定位是直接运行的
命令行训练入口，而不是一个可独立复用的训练库。
"""

import os
import sys

# 将当前脚本声明为 trainer 包内模块，并把仓库根目录加入模块搜索路径。
# 因而从 trainer/ 目录直接执行时，也能导入同仓库的 model、dataset 和 trainer 模块。
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
from model.model_lora import save_lora, apply_lora
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, lora_params, start_step=0, wandb=None):
    """训练一个 epoch，或从断点处训练该 epoch 尚未完成的部分。

    参数：
        epoch: 从 0 开始的 epoch 编号；日志中显示时加 1。
        loader: 当前进程的 DataLoader。每批返回 ``(input_ids, labels)``，形状通常均为
            ``[batch_size, max_seq_len]``；labels 中的 ``-100`` 不参与交叉熵。
        iters: 当前 epoch 的完整微批次数。续训时传入“剩余批数 + 已跳过批数”，使学习率、
            日志与保存间隔继续沿用中断前的 step 编号。
        lora_params: 唯一允许更新的 LoRA 参数列表，也用于梯度裁剪。
        start_step: 当前 epoch 已经完成的微批次数；从头训练时为 0。
        wandb: 实际是采用 wandb 风格接口的 SwanLab 模块；为 ``None`` 时不上传指标。

    这里的 ``step`` 是微批次编号，不一定等于优化器更新次数：只有累积
    ``accumulation_steps`` 个微批次后才更新一次参数。单进程的近似有效批大小为
    ``batch_size * accumulation_steps``，DDP 下还要再乘进程数（GPU 数）。
    """
    # 计时从本次实际执行的位置开始；last_step 用来判断循环结束时是否残留未更新梯度。
    start_time = time.time()
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # DataLoader 在 CPU 端产生张量，训练前搬到当前进程绑定的 CPU/GPU 设备。
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        # 将 epoch 和批次编号展平为全局训练进度，按余弦曲线逐步衰减学习率。
        # 遍历 param_groups 可兼容今后为不同参数设置不同优化器分组的场景。
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # GPU 上 autocast 让适合的算子以 BF16/FP16 运行，以降低显存和计算开销；
        # CPU 上 autocast_ctx 是空上下文，不会自动更改计算精度。
        with autocast_ctx:
            res = model(input_ids, labels=labels)
            # res.loss 是仅在 assistant 标签处计算的 next-token 交叉熵。
            # 稠密模型的 aux_loss 为 0；MoE 模型用它鼓励 token 在专家间均衡路由。
            loss = res.loss + res.aux_loss
            # 每个微批次贡献 1/N 的梯度，连续反传 N 次后近似得到大批次平均梯度。
            loss = loss / args.accumulation_steps

        # GradScaler 仅在 FP16 模式真正缩放损失；BF16/CPU 下仍可共用这一调用方式。
        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0 or step == iters:
            # 梯度裁剪前先撤销 FP16 缩放；而且只裁剪真正可训练的 LoRA 参数。
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
            # FP16 梯度若出现 inf/NaN，scaler.step 会跳过本次更新；update 会调整缩放因子。
            scaler.step(optimizer)
            scaler.update()
            # 将梯度设为 None 通常比逐元素清零更省显存，下次反传时再按需创建。
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            # loss 在反传前除过累积步数，此处乘回去，展示当前微批次的原始总损失。
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            # 总损失 = 语言建模损失 + MoE 辅助损失，相减即可单独观察主损失。
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            # 根据本次运行已完成批次的平均耗时，估算当前 epoch 剩余分钟数。
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            # 只有主进程会创建日志实例，所以 DDP 不会由每个 rank 重复上报指标。
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            # 仅 rank 0 写文件，避免多个 DDP 进程同时覆盖相同路径。
            # eval() 会关闭 dropout；保存结束后切回 train()，继续正常训练。
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            lora_save_path = f'{args.save_dir}/{args.lora_name}_{lm_config.hidden_size}{moe_suffix}.pth'
            # 面向部署/叠加使用的轻量文件：只保存各线性层的 LoRA A、B 矩阵。
            # 同名文件会在每次保存时覆盖，推理时需与 --from_weight 指定的同一基础权重搭配。
            save_lora(model, lora_save_path)
            # lm_checkpoint 会在 checkpoints/ 写两类文件：一个是“基础参数 + LoRA 参数”的
            # 完整阶段权重，另一个带 _resume 后缀，额外包含 AdamW、GradScaler、epoch/step
            # 和实验 ID 以供续训。它们都不同于上面仅含 A、B 矩阵的轻量 LoRA 文件。
            lm_checkpoint(lm_config, weight=args.lora_name, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()

        # 及时释放本批大张量的 Python 引用，便于后续迭代复用显存。
        del input_ids, labels, res, loss

if __name__ == "__main__":
    # 只有直接运行本文件时才启动训练；被其他模块导入时只定义 train_epoch。
    parser = argparse.ArgumentParser(description="MiniMind LoRA 参数高效微调")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument("--lora_name", type=str, default="lora_medical", help="LoRA权重名称(如lora_identity/lora_medical等)")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="每个进程每个微批次的样本数")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型：bfloat16 或 float16")
    parser.add_argument("--num_workers", type=int, default=8, help="每个进程的数据加载子进程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="多少个微批次累积后更新一次参数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=10, help="每隔多少个微批次打印一次日志")
    parser.add_argument("--save_interval", type=int, default=1000, help="每隔多少个微批次保存一次权重和断点")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/lora_medical.jsonl", help="LoRA训练数据路径")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基础权重名称；默认在 full_sft 权重上训练 LoRA")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-LoRA", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    # 普通 python 启动时不会初始化进程组，local_rank 为 0；torchrun 启动时则读取
    # RANK/LOCAL_RANK，用 NCCL 建立进程组，并让每个进程绑定一张 GPU。
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    # 不同 rank 使用不同种子，避免各 GPU 上的数据增强等随机行为完全相同。
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查断点 ==========
    os.makedirs(args.save_dir, exist_ok=True)
    # 未显式传入的词表大小、注意力头数、中间层宽度等采用 MiniMindConfig 默认值。
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    # --from_resume 会读取 checkpoints/*_resume.pth；不存在时返回 None，并自然从头训练。
    ckp_data = lm_checkpoint(lm_config, weight=args.lora_name, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    # 当前约定：字符串恰为 bfloat16 时采用 BF16，其余值均按 FP16 处理。
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # CPU 训练不进入 CUDA autocast；GPU 前向和损失计算则自动选用指定低精度。
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配置实验日志 ==========
    wandb = None
    if args.use_wandb and is_main_process():
        # 项目使用 SwanLab，但保留变量名 wandb，以沿用熟悉的 init/log 接口风格。
        import swanlab as wandb
        # 断点中若记录了实验 ID，则续写原实验；否则创建新的实验记录。
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-LoRA-{args.lora_name}-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、应用 LoRA、冻结非 LoRA 参数 ==========
    # 默认从 ../out/full_sft_{hidden_size}[ _moe].pth 加载基础权重，tokenizer 则从
    # ../model 读取。LoRA 通常接在已经具备对话能力的 SFT 模型之上做垂域适配。
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    # apply_lora 会遍历模型中的方形 nn.Linear，在原 forward 旁动态挂载 B(A(x)) 分支。
    # 默认 rank=16；B 以 0 初始化，所以刚挂载时增量输出为 0，不会立刻扰动基础模型。
    apply_lora(model)
    
    # numel 统计标量个数而不是张量个数；这里先展示整体规模，再展示真正新增的 LoRA 规模。
    total_params = sum(p.numel() for p in model.parameters())
    lora_params_count = sum(p.numel() for name, p in model.named_parameters() if 'lora' in name)
    Logger(f"LLM 总参数量: {total_params / 1e6:.3f} M")
    Logger(f"LoRA 参数量: {lora_params_count / 1e6:.3f} M")
    Logger(f"LoRA 参数占比: {lora_params_count / total_params * 100:.2f}%")
    
    # 冻结基础参数，只开放名称中含 lora 的 A、B 矩阵，同时收集它们供优化器和梯度裁剪使用。
    # 冻结不会跳过基础模型前向：仍需用原权重算出完整输出，只是不为其保存/更新梯度。
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
            lora_params.append(param)
        else:
            param.requires_grad = False
    
    # ========== 6. 定义数据和优化器 ==========
    # SFTDataset 用 tokenizer 的 chat_template 渲染多轮对话，只在 assistant 回复区域
    # 保留监督标签；system、user、工具描述和 padding 位置均为 -100，不参与交叉熵。
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # DDP 下每个 rank 只采样自己的一份数据，避免每张 GPU 重复遍历完整数据集。
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # FP16 动态范围较小，需要动态 loss scaling；BF16 指数范围较大，通常无需缩放。
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    # 优化器只接收 lora_params，这是参数高效微调与全参数 SFT 的关键区别。
    optimizer = optim.AdamW(lora_params, lr=args.learning_rate)
    
    # ========== 7. 从断点恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        # 先 apply_lora、再载入断点，模型中才存在可接收断点键名的 LoRA 分支。
        # strict=False 允许包装或版本差异带来的少量键名不匹配；同时恢复优化器和缩放器，
        # 尽可能延续中断前的优化轨迹，而不只是恢复参数数值。
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        # 仍从保存时所在 epoch 开始，后续由 SkipBatchSampler 跳过已经完成的批次。
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 8. 编译和分布式包装 ==========
    if args.use_compile == 1:
        # apply_lora 通过替换 Linear.forward 动态注入分支（monkey patch），与当前
        # torch.compile 的图捕获方式不兼容，因此即使命令行请求开启也会安全回退。
        args.use_compile = 0
        Logger('[LoRA] monkey-patch forward 与 torch.compile 不兼容，use_compile 已自动关闭')
    if dist.is_initialized():
        # DDP 会在反向传播时同步各 rank 的 LoRA 梯度；每个进程只操作 local_rank 对应 GPU。
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 9. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # DDP 采样器根据 epoch 改变每轮洗牌；单卡则在下方显式生成随机索引顺序。
        train_sampler and train_sampler.set_epoch(epoch)
        # 固定每轮种子，使断点重启后可以重建相同数据顺序并准确跳过旧批次。
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        # 只在恢复后的第一个 epoch 跳过 start_step，后续 epoch 均从第 1 批开始。
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        # 按 batch_size 组成批次并丢弃前 skip 批；末尾不足整批的样本仍会保留。
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        # 指定 batch_sampler 后无需再传 batch_size/shuffle；pin_memory 可加快 CPU→GPU 拷贝。
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            # len(loader) 只统计剩余批次，加回 skip 才是完整 epoch 的批次数。
            train_epoch(epoch, loader, len(loader) + skip, lora_params, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), lora_params, 0, wandb)
    
    # ========== 10. 清理分布进程 ==========
    if dist.is_initialized():
        # 等待所有 rank 训练结束再销毁进程组，避免先退出的进程打断其他进程通信。
        dist.barrier()
        dist.destroy_process_group()
