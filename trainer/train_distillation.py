"""MiniMind 知识蒸馏训练入口：用教师模型的输出分布指导学生模型学习。

建议在阅读本文件前，先看以下代码：
1. ``dataset/lm_dataset.py`` 中的 ``SFTDataset``：理解对话如何套用 chat template 编码，
   以及为什么只有 assistant 回复的标签有效、其他位置被标为 ``-100``。
2. ``model/model_minimind.py`` 中的 ``MiniMindConfig``、``MiniMindForCausalLM.forward``：
   理解模型输入/输出的形状、``logits`` 如何预测下一个 token，以及 MoE 模型的
   ``aux_loss`` 从哪里来。
3. ``trainer/trainer_utils.py`` 中的 ``init_model``、``get_lr``、``lm_checkpoint``、
   ``init_distributed_mode``、``setup_seed`` 和 ``SkipBatchSampler``：了解模型加载、
   学习率、断点、分布式训练与续训数据跳过逻辑。
4. ``trainer/train_full_sft.py``：先熟悉常规 SFT 的训练循环，再比较本文件增加的教师
   前向计算和 KL 蒸馏损失。

读完本文件后，推荐继续阅读：
1. ``trainer/train_full_sft.py``：对比只使用真实标签交叉熵的监督微调流程。
2. ``model/model_minimind.py`` 中的 ``MOEFeedForward``：理解教师或学生为 MoE 时，
   路由辅助损失为什么会加到学生的交叉熵项中。
3. ``trainer/train_dpo.py``：了解 SFT/蒸馏之后，如何使用偏好数据继续对齐模型。
4. ``eval_llm.py``：了解本脚本保存的学生模型权重如何加载并用于生成。

整体流程是：创建学生与冻结的教师 -> 用同一批 SFT token 分别前向计算 -> 在有效的
assistant 目标 token 上混合真实标签交叉熵与教师分布的 KL 散度 -> 只更新学生 -> 定期
保存学生权重和续训状态。学生与教师可以有不同的层数、隐藏维度或 MoE 结构，但它们
必须使用相同且顺序一致的 tokenizer 词表，因为蒸馏会逐词比较两边的 logits。

这是供命令行使用的训练脚本：``train_epoch`` 会读取主程序创建的 ``args``、``model``、
``optimizer``、``scaler`` 和 ``autocast_ctx`` 等全局对象。
"""

import os
import sys

# 将仓库根目录加入模块搜索路径，使直接运行 trainer/ 下的脚本时也能导入 model、dataset。
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 虽然本脚本没有直接使用 datasets，但必须在 torch 前导入，以规避 Windows 下
# pyarrow 与 torch 的 DLL 加载冲突（项目 issue #771）；F401 表示忽略“未使用导入”提示。
import datasets  # noqa: F401
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
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def distillation_loss(student_logits, teacher_logits, temperature=1.0, reduction='batchmean'):
    """计算教师分布到学生分布的 KL 蒸馏损失。

    ``student_logits`` 和 ``teacher_logits`` 的最后一维都是词表维度；本训练循环会先把
    张量整理成 ``[有效目标 token 数, 词表大小]``，因此 ``batchmean`` 会对有效 token
    数取平均。温度 ``temperature`` 同时缩放两边 logits：较高温度会软化分布，让学生
    也能学习教师对非最大概率 token 的相对判断；乘回温度平方用于补偿缩放 logits 后
    梯度变小的影响。

    PyTorch 的 ``kl_div(input, target)`` 中，``input`` 是对数概率，``target`` 是概率，
    因而这里计算的是 ``KL(教师分布 || 学生分布)``。教师概率在 ``no_grad`` 中产生，
    教师只提供学习目标，不会收到梯度。
    """
    # softmax 把教师的 logits 转成每个词元的概率；禁止记录计算图以节省显存。
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()

    # KL 散度的 input 参数需要对数概率，而不是普通概率。
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    # target 是教师概率，input 是学生对数概率；batchmean 按有效 token 行数求平均。
    kl = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction=reduction
    )
    return (temperature ** 2) * kl


def train_epoch(epoch, loader, iters, teacher_model, lm_config_student, start_step=0, wandb=None, alpha=0.0, temperature=1.0):
    """训练一个 epoch；若从断点恢复，则只处理这个 epoch 中尚未完成的批次。

    参数：
        epoch: 从 0 开始的轮次编号，显示日志时会加 1。
        loader: 当前进程的数据加载器。每批给出 ``input_ids`` 和 ``labels``，形状为
            ``[batch_size, 序列长度]``；``labels`` 中 ``-100`` 的位置不计入监督损失。
        iters: 当前 epoch 的总微批次数。续训时等于“已跳过批次 + 剩余批次”，这样 step
            编号仍与中断前一致。
        teacher_model: 冻结的教师模型；为 ``None`` 时只计算交叉熵项。
        lm_config_student: 学生模型配置，用于判断是否需要 MoE 辅助损失及输出文件名。
        start_step: 当前 epoch 已经完成的微批次数，正常开始时为 0。
        wandb: 采用 wandb 风格接口的 SwanLab 模块；为 ``None`` 时不记录在线指标。
        alpha: 真实标签损失的权重；蒸馏损失权重为 ``1 - alpha``。
        temperature: 计算教师与学生概率分布时使用的蒸馏温度。

    本函数还会读取主程序定义的全局变量。``step`` 是微批次序号；梯度累计时，多个
    微批次的梯度合并后才执行一次优化器更新。
    """
    # 以本次实际训练起点计时，用于估算本轮剩余时间。
    start_time = time.time()
    
    if teacher_model is not None:
        # eval() 关闭 dropout 等训练期行为；requires_grad_(False) 冻结所有教师参数。
        # 前向时还会使用 no_grad，避免为教师保留反向传播所需的中间激活。
        teacher_model.eval()
        teacher_model.requires_grad_(False)

    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # DataLoader 在 CPU 上产出张量；把输入和标签搬到当前进程使用的设备。
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        # 模型位置 t 的 logits 对应标签位置 t+1，因此标签也要先去掉第一个位置。
        # SFTDataset 用 -100 标记 user/system/padding 等不监督的位置；此布尔掩码只保留
        # assistant 回复中的目标 token。float 类型方便后面用乘法计算掩码平均损失。
        loss_mask = (labels[..., 1:] != -100).float()
        # 把 epoch 与 step 合成训练进度，按余弦曲线更新学习率。
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 学生前向传播。logits 原形状为 [批次, 序列长度, 词表大小]；去掉最后一格，
        # 便于与 labels[..., 1:] 对齐，即位置 t 的输出用来预测下一个 token。
        # contiguous() 确保后续 view() 可将张量安全地展平为 token 行。
        with autocast_ctx:
            res = model(input_ids)
            student_logits = res.logits[..., :-1, :].contiguous()

        # 教师前向也使用同一批 input_ids 和相同位置对齐；eval/no_grad 确保教师仅作参考。
        # 这里没有进入 autocast_ctx，因此通常按教师模型参数的精度执行前向。
        if teacher_model is not None:
            with torch.no_grad():
                teacher_logits = teacher_model(input_ids).logits[..., :-1, :].contiguous()
                vocab_size_student = student_logits.size(-1)
                # 若教师词表维度比学生大，截到学生词表大小才能逐词计算 KL。
                # 这不等于通用的词表映射：两边 token ID 的含义和顺序仍必须一致；若教师
                # 词表更小或顺序不同，蒸馏目标就无法正确对齐。
                teacher_logits = teacher_logits[..., :vocab_size_student]

        # ========== 计算训练损失 ==========
        # 1）真实标签交叉熵（Cross Entropy，CE）：学生是否预测对了数据中的目标 token。
        # 这里再次将标签向左错一位，与去尾后的 logits 在序列位置上严格对齐。
        shift_labels = labels[..., 1:].contiguous()
        # 展平为 [批次 × (序列长度 - 1)]，后续对每个 token 单独保留或屏蔽损失。
        loss_mask_flat = loss_mask.view(-1)
        # 先计算每个位置的交叉熵，不立即求平均，以便只对 assistant 有效位置归一化。
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction='none'
        )
        # 无效位置乘 0；分母是有效 token 数，避免把不同数量的回答 token 当作同等总量。
        # 加上极小值，防止极端情况下整批没有有效标签时除以 0。
        ce_loss_raw = torch.sum(ce_loss * loss_mask_flat) / (loss_mask_flat.sum() + 1e-8)
        # MoE 学生还要加上路由均衡辅助损失；日志中的 ce_loss_raw 仍只显示纯交叉熵。
        if lm_config_student.use_moe: ce_loss = ce_loss_raw + res.aux_loss
        else: ce_loss = ce_loss_raw

        # 2）蒸馏损失：比较同一有效目标位置上的学生与教师词表概率分布。
        if teacher_model is not None:
            distill_loss = distillation_loss(
                # 合并批次和序列维度，并仅挑出 assistant 的有效目标 token。
                student_logits.view(-1, student_logits.size(-1))[loss_mask_flat == 1],
                teacher_logits.view(-1, teacher_logits.size(-1))[loss_mask_flat == 1],
                temperature=temperature
            )
        else:
            # 兼容不传教师模型的调用；此时蒸馏项为 0，训练只剩 alpha 加权的 CE 项。
            distill_loss = torch.tensor(0.0, device=args.device)

        # 3）混合损失 = alpha ×（CE 及可能的 MoE 辅助项）+ (1-alpha) × KL。
        # 除以梯度累计步数，使累计多个微批次后梯度近似于大批次平均梯度。
        loss = (alpha * ce_loss + (1 - alpha) * distill_loss) / args.accumulation_steps

        # 反向传播只经过学生模型。FP16 时 scaler 会放大损失以避免小梯度下溢；BF16 下
        # scaler 被禁用，但这套接口仍可直接调用。
        scaler.scale(loss).backward()

        # 每累计 N 个微批次更新一次；step == iters 处理最后不足 N 个批次的梯度，
        # 避免 epoch 末尾已经算出的梯度被丢弃。注意最后一组仍按完整 N 步缩放 loss，
        # 所以不足 N 步时，这一组的梯度幅度会相对更小。
        if step % args.accumulation_steps == 0 or step == iters:
            # 若使用 FP16，先还原缩放后的梯度，再按 grad_clip 限制其整体范数。
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # 梯度有限时真正更新参数；GradScaler 也会据此调整下一轮缩放比例。
            scaler.step(optimizer)
            scaler.update()
            # 清除本轮累计梯度。set_to_none=True 通常比把张量逐元素写零更省内存。
            optimizer.zero_grad(set_to_none=True)

        # 固定间隔或本轮最后一步打印指标；loss 乘回累计步数后才是未缩放的损失值。
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            # CE 单独记录纯真实标签交叉熵；若学生为 MoE，aux_loss 单独记录路由均衡项。
            current_ce_loss = ce_loss_raw.item()
            current_aux_loss = res.aux_loss.item() if lm_config_student.use_moe else 0.0
            current_lr = optimizer.param_groups[-1]['lr']
            # 用从 start_step 到当前 step 的平均耗时估算当前 epoch 剩余分钟数。
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, ce: {current_ce_loss:.4f}, aux_loss: {current_aux_loss:.4f}, distill: {distill_loss.item():.4f}, learning_rate: {current_lr:.8f}, epoch_time: {eta_min:.3f}min')
            
            # 仅主进程初始化了 wandb/SwanLab；其他 rank 的 wandb 为 None。
            if wandb:
                wandb.log({
                    "loss": current_loss,
                    "ce_loss": current_ce_loss,
                    "aux_loss": current_aux_loss,
                    "distill_loss": distill_loss.item() if teacher_model is not None else 0.0,
                    "learning_rate": current_lr,
                    "epoch_time": eta_min
                })

        # 固定间隔及 epoch 末尾保存，并确保 DDP 下只有 rank 0 写文件。
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            # 临时切到评估状态，再在保存后切回训练状态。
            model.eval()
            moe_suffix = '_moe' if lm_config_student.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config_student.hidden_size}{moe_suffix}.pth'
            # DDP 和 torch.compile 都会包装模型；保存前剥除外层包装，保持参数名可直接加载。
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            # 在 save_dir 写入半精度 CPU 权重，供后续训练阶段或推理加载。
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            # checkpoints 目录另存包含优化器、scaler、epoch、step 等状态的续训检查点。
            lm_checkpoint(lm_config_student, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            del state_dict

        # 释放本批次大张量的引用，减少后续迭代的显存占用。
        del input_ids, labels, loss_mask, res, student_logits, ce_loss, distill_loss, loss


if __name__ == "__main__":
    # 常见用法是用 MoE 教师蒸馏稠密学生，也可用更大教师蒸馏更小学生。
    # 两个模型结构可以不同，但必须共享兼容的词表与 token ID 定义。
    parser = argparse.ArgumentParser(description="MiniMind Knowledge Distillation")
    # 输出位置和训练时长。路径按启动命令时的当前工作目录解析，不是相对本脚本位置。
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='full_dist', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=6, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    # accumulation_steps 个微批次反传后才更新一次；近似有效 batch 为 batch_size × 累积步数，
    # 使用 DDP 时还要乘上进程数。
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument("--max_seq_len", type=int, default=340, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_t2t_mini.jsonl", help="训练数据路径")
    # 学生和教师分别配置宽度、层数与 MoE 开关；MiniMindConfig 的其他字段使用默认值。
    parser.add_argument('--student_hidden_size', default=768, type=int, help="学生模型隐藏层维度")
    parser.add_argument('--student_num_layers', default=8, type=int, help="学生模型隐藏层数量")
    parser.add_argument('--teacher_hidden_size', default=768, type=int, help="教师模型隐藏层维度")
    parser.add_argument('--teacher_num_layers', default=8, type=int, help="教师模型隐藏层数量")
    parser.add_argument('--student_use_moe', default=0, type=int, choices=[0, 1], help="学生模型是否使用MoE（0=否，1=是）")
    parser.add_argument('--teacher_use_moe', default=1, type=int, choices=[0, 1], help="教师模型是否使用MoE（0=否，1=是）")
    # init_model 会按权重名、hidden_size 和可选的 _moe 后缀查找权重文件；none 表示随机初始化。
    parser.add_argument('--from_student_weight', default='full_sft', type=str, help="学生模型基于哪个权重")
    parser.add_argument('--from_teacher_weight', default='full_sft', type=str, help="教师模型基于哪个权重")
    # from_resume 读取的是 ../checkpoints 下的完整训练状态，而非仅含模型权重的 .pth 文件。
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    # alpha=1 时只使用真实标签 CE（学生为 MoE 时还会带辅助项）；alpha=0 时只使用教师 KL。
    # 两者之间的值按比例混合这两项目标。
    # temperature 越高，教师/学生的概率分布越平滑；常见做法是从略大于 1 的值起试。
    parser.add_argument('--alpha', default=0.5, type=float, help="CE损失权重，总损失=alpha*CE+(1-alpha)*KL")
    parser.add_argument('--temperature', default=1.5, type=float, help="蒸馏温度（推荐范围1.0-2.0）")
    # 实验跟踪、图编译是可选项；本项目用 SwanLab 提供兼容 wandb 的 init/log 接口。
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Distillation", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    # python 单进程运行时不初始化进程组；torchrun 会设置环境变量并启动多个 DDP 进程。
    local_rank = init_distributed_mode()
    # DDP 下每个进程绑定自己的 GPU；单卡设备保留命令行参数指定的值。
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    # 让各 rank 的随机行为不同；后面每轮还会重设种子以复现该轮的数据打乱顺序。
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    # 创建权重输出目录；save_dir 和下方检查点目录都按当前工作目录解析。
    os.makedirs(args.save_dir, exist_ok=True)
    # 两个配置各自创建模型结构，未传入的词表大小、注意力头数等采用 MiniMindConfig 默认值。
    lm_config_student = MiniMindConfig(hidden_size=args.student_hidden_size, num_hidden_layers=args.student_num_layers, use_moe=bool(args.student_use_moe))
    lm_config_teacher = MiniMindConfig(hidden_size=args.teacher_hidden_size, num_hidden_layers=args.teacher_num_layers, use_moe=bool(args.teacher_use_moe))
    # 续训状态按学生的配置与 save_weight 定位；未启用续训时 ckp_data 为 None。
    ckp_data = lm_checkpoint(lm_config_student, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    # 建立混合精度上下文：CPU 上是空上下文；GPU 上按 --dtype 选择 BF16 或 FP16。
    # 训练循环目前只在学生前向时进入该上下文，教师前向没有使用 autocast。
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    # 仅 rank 0 创建实验并记录指标，避免多卡重复建立相同实验或上报重复日志。
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        # 如果续训检查点保存了实验 ID，就连接回原实验；否则开始一个新实验。
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Distill-S{args.student_hidden_size}T{args.teacher_hidden_size}-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义学生和教师模型 ==========
    # 学生模型会接收梯度并更新；这里返回的 tokenizer 用来把 SFT 对话编码为 input_ids。
    model, tokenizer = init_model(lm_config_student, args.from_student_weight, device=args.device)
    Logger(f'学生模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')
    # 教师用同一输入 token 生成目标分布。init_model 默认从同一 ../model 目录加载 tokenizer，
    # 因而两者的 token ID 语义一致；教师参数冻结，不交给优化器。
    teacher_model, _ = init_model(lm_config_teacher, args.from_teacher_weight, device=args.device)
    teacher_model.eval()
    teacher_model.requires_grad_(False)
    Logger(f'教师模型总参数量：{sum(p.numel() for p in teacher_model.parameters()) / 1e6:.3f} M')
    # SFTDataset 负责聊天模板和 assistant 标签掩码；DataLoader 后续将多条样本组成批次。
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # 分布式时按 rank 切分数据；普通单卡时在每个 epoch 中手动打乱索引。
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # GradScaler 只在 FP16 开启；BF16 动态范围较大，通常不需要 loss scaling。
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    # 优化器只接收学生参数，所以无论教师是否参与前向，它都不会被更新。
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        # 除学生权重外，也恢复 AdamW 动量与 GradScaler 状态，以延续此前的训练过程。
        # 尚未达到 accumulation_steps 的待更新梯度不会保存在检查点中，因此在梯度累计
        # 组的中途恢复时，结果不会与不中断训练完全相同。
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        # 从保存时的 epoch/step 继续；后续 sampler 会跳过该 epoch 已经训练过的批次。
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        # torch.compile 可能加速学生前向/反向；首次调用会有编译开销。
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        # DDP 在反向传播时同步不同 rank 上学生模型的梯度；教师不需要梯度同步。
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # DistributedSampler 需要每轮设置 epoch，才能在各 rank 间一致地重新洗牌。
        train_sampler and train_sampler.set_epoch(epoch)
        # 重设本轮随机种子。非 DDP 时使用随机索引列表；DDP 时实际由 train_sampler 采样。
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        # 仅恢复时所在的首个 epoch 跳过已训练 step；之后的 epoch 从开头训练。
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        # 将样本下标组成批次；SkipBatchSampler 会跳过恢复前已经完成的批次。
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            # len(loader) 现在只包含剩余批次；加回 skip 才是完整 epoch 的步数，供学习率、
            # 日志和保存条件继续使用原来的 step 编号。
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, teacher_model, lm_config_student, start_step, wandb, args.alpha, args.temperature)
        else:
            train_epoch(epoch, loader, len(loader), teacher_model, lm_config_student, 0, wandb, args.alpha, args.temperature)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        # 等所有 rank 完成本轮训练再销毁通信组，避免进程提前退出。
        dist.barrier()
        dist.destroy_process_group()
