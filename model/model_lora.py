"""MiniMind 的 LoRA（Low-Rank Adaptation，低秩适配）核心实现。

建议先阅读（有助于理解本文件）：
    1. ``model/model_minimind.py`` 中的 ``Attention`` 和 ``FeedForward``：了解模型里
       各个 ``nn.Linear`` 分别负责 Q/K/V、注意力输出和 MLP 投影，以及它们的输入、
       输出维度。
    2. ``trainer/train_lora.py``：了解本文件在训练流程中的位置——先加载基础模型，
       再调用 ``apply_lora``，冻结非 LoRA 参数，最后用 ``save_lora`` 只保存增量权重。
    3. ``trainer/trainer_utils.py`` 中的 ``init_model``：了解基础模型配置、权重和设备是
       怎样准备好的，进而理解这里为什么可以使用 ``model.device``。

读完本文件后，推荐继续阅读：
    1. ``eval_llm.py`` 中的 ``init_model``：查看推理时为何必须先 ``apply_lora`` 创建
       A、B 层，再用 ``load_lora`` 填入训练好的参数。
    2. ``scripts/convert_model.py`` 中的 ``convert_merge_base_lora``：查看如何调用
       ``merge_lora``，把基础权重与 LoRA 增量合成为普通模型权重。
    3. ``README.md`` 的“LoRA”章节：了解训练数据格式、命令行用法、权重文件命名和
       领域微调示例。

核心公式：
    普通线性层输出为 ``y = x W^T``。LoRA 冻结原权重 ``W``，只学习低秩增量
    ``ΔW = B A``，于是输出变为 ``y = x W^T + x (B A)^T``。若输入维度为
    ``d_in``、输出维度为 ``d_out``、秩为 ``r``，则：

    * ``A.weight`` 的形状为 ``[r, d_in]``，先把特征压缩到 r 维；
    * ``B.weight`` 的形状为 ``[d_out, r]``，再把 r 维特征还原到输出维度；
    * 新增参数量从完整增量矩阵的 ``d_out * d_in`` 降为
      ``r * (d_in + d_out)``。

本仓库采用便于教学的极简版本：没有常见实现里的 ``alpha / rank`` 缩放系数和 LoRA
dropout，并且只给输入、输出维度相等的方形线性层加 LoRA。按 MiniMind 默认配置，这
主要对应每个注意力层的 ``q_proj`` 和 ``o_proj``；GQA 的 ``k_proj/v_proj``、MLP
投影和 ``lm_head`` 因为不是方阵，不会被注入。
"""

import torch
from torch import optim, nn


class LoRA(nn.Module):
    """由两个无偏置线性层组成的低秩增量分支 ``B(A(x))``。

    参数：
        in_features: 原线性层的输入特征数 ``d_in``。
        out_features: 原线性层的输出特征数 ``d_out``。
        rank: 低秩瓶颈宽度 ``r``。越小，新增参数和计算量越少，但表达能力也越受限。

    输入的最后一维必须是 ``in_features``；前面的维度可以是任意批次维。例如语言模型
    中常见输入 ``[batch_size, seq_len, d_in]``，输出仍为
    ``[batch_size, seq_len, d_out]``。

    注意：本模块只计算增量项，不包含原线性层。原分支与 LoRA 分支的相加发生在
    :func:`apply_lora` 动态替换的前向函数里。
    """

    def __init__(self, in_features, out_features, rank):
        super().__init__()
        # 保存 rank 便于调试和检查结构；真正决定参数形状的是下面两个 Linear。
        self.rank = rank

        # PyTorch 的 Linear 权重布局是 [输出维度, 输入维度]：
        # A.weight:[rank, in_features]，B.weight:[out_features, rank]。
        # 两层均不使用 bias，确保整个分支严格等价于一个低秩权重增量 ΔW=B@A。
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)

        # A 使用小方差高斯分布初始化，使低秩瓶颈一开始具有非零、但幅度较小的特征。
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # B 初始化为全 0，因此刚注入时 B(A(x)) 恒为 0，模型输出与基础模型完全一致。
        # 第一次反向传播时 B 可以获得梯度；随着 B 离开 0，A 也会逐步获得有效梯度。
        self.B.weight.data.zero_()

    def forward(self, x):
        """先降维到 ``rank``，再升维到原线性层的输出宽度。"""
        return self.B(self.A(x))


def apply_lora(model, rank=16):
    """给模型中的方形 ``nn.Linear`` 动态挂载 LoRA 分支。

    ``named_modules`` 会递归遍历模型。每找到一个满足
    ``module.in_features == module.out_features`` 的线性层，就执行三件事：

    1. 创建与原层输入、输出相匹配的 :class:`LoRA`；
    2. 以子模块名 ``lora`` 注册到原层，使 A、B 自动进入 ``parameters`` 和
       ``state_dict``；
    3. 把该层的前向计算从 ``original_forward(x)`` 改成
       ``original_forward(x) + lora(x)``。

    参数 ``rank`` 必须和之后加载的 LoRA 权重一致，否则 A、B 的张量形状无法匹配。
    本函数不会替调用者冻结基础参数；训练脚本会在注入之后单独设置
    ``requires_grad``。此外，同一个模型实例应只调用一次本函数：重复调用会再次包装
    ``forward``，导致增量分支叠加或产生未注册的旧分支。

    ``model`` 需提供本仓库模型使用的 ``.device`` 属性。新建分支会立即移动到该设备，
    避免前向时出现 CPU/GPU 张量混用。
    """
    for name, module in model.named_modules():
        # 本实现用“方阵”作为简洁的目标层筛选规则。name 主要供遍历和调试使用；
        # 实际是否注入仅由模块类型和输入、输出维度决定。
        if isinstance(module, nn.Linear) and module.in_features == module.out_features:
            lora = LoRA(module.in_features, module.out_features, rank=rank).to(model.device)

            # 通过 setattr 注册子模块后，参数名会类似：
            # model.layers.0.self_attn.q_proj.lora.A.weight。
            setattr(module, "lora", lora)
            # 保存替换前的绑定方法；它仍负责使用冻结的基础权重 W 计算原输出。
            original_forward = module.forward

            # 默认参数在函数定义时固定当前循环的原层和 LoRA 层，避免 Python 闭包的
            # “延迟绑定”让所有层最终错误地引用循环中最后创建的那一层。
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                # 两条分支形状相同，逐元素相加得到 W x + B(A(x))。
                return layer1(x) + layer2(x)

            # 这是实例级的动态替换（monkey patch），不会修改 nn.Linear 类本身，也不会
            # 影响其他模型实例。仓库训练脚本因此会关闭与这种写法不兼容的 torch.compile。
            module.forward = forward_with_lora


def load_lora(model, path):
    """从轻量权重文件中加载各层的 LoRA A、B 参数。

    调用前必须已经用 :func:`apply_lora` 创建完全相同的 LoRA 结构，包括目标层和 rank。
    ``path`` 通常指向 ``save_lora`` 生成的 ``lora_xxx_*.pth``，而不是包含基础模型、
    优化器等内容的训练续跑断点。

    权重会直接映射到 ``model.device``，因此加载后即可前向推理。若保存文件来自 DDP，
    参数名可能带有最外层 ``module.`` 前缀；这里会统一移除它。
    """
    # torch.load 会反序列化文件内容，因此 path 应来自可信来源。
    state_dict = torch.load(path, map_location=model.device)
    # DDP 会把真实模型放在名为 module 的包装层下；去掉前缀后即可兼容普通单卡模型。
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            # 从全局键名（如 xxx.q_proj.lora.A.weight）筛出当前层，再裁成 LoRA 模块
            # 自己认识的局部键名（A.weight、B.weight）。load_state_dict 默认严格校验，
            # 因而 rank 不一致或某层权重缺失时会明确报错，而不是静默使用错误参数。
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)


def save_lora(model, path):
    """只保存 LoRA 参数，不重复保存体积更大的基础模型参数。

    保存结果的键名保留 LoRA 所属线性层的完整路径，以便 :func:`load_lora` 将每组
    ``A.weight``、``B.weight`` 分配回正确位置。张量在保存前转到 CPU 并压缩成 FP16，
    可减小文件体积；载入模块时 PyTorch 会按目标参数的 dtype 复制数值。

    ``torch.compile`` 可能用 ``_orig_mod`` 包住原模型，DDP 则会在参数名前增加
    ``module.``；本函数分别解开前者、清理后者，使输出键名与普通模型保持一致。
    调用者需要事先创建 ``path`` 的父目录。
    """
    # 未经 torch.compile 包装时回退到 model 本身。
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            # DDP 下 name 形如 module.model.layers...；轻量文件不保留最外层 module.。
            clean_name = name[7:] if name.startswith("module.") else name
            # module.lora.state_dict() 只含当前分支的 A.weight 与 B.weight。
            # 加回所属层路径，避免不同 Transformer 层的同名参数互相覆盖。
            lora_state = {f'{clean_name}.lora.{k}': v.cpu().half() for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)


def merge_lora(model, lora_path, save_path):
    """将 LoRA 增量 ``B @ A`` 合入基础线性层并保存普通模型权重。

    调用顺序应为“创建并加载基础模型 → :func:`apply_lora` → 本函数”。本函数内部先从
    ``lora_path`` 加载增量参数，再为每个基础线性层计算
    ``merged_weight = original_weight + B.weight @ A.weight``。矩阵形状依次为
    ``[d_out, d_in] = [d_out, r] @ [r, d_in]``，正好与原 ``weight`` 一致。

    写入 ``save_path`` 的 state_dict 会排除所有包含 ``.lora.`` 的键，因此结果可由未
    调用 ``apply_lora`` 的普通 MiniMind 直接加载。所有张量以 CPU FP16 保存；合并只
    构造待保存的张量，不会把当前模型的基础 ``module.weight`` 原地改写。
    """
    # 此时 model 中必须已有 lora 子模块，否则 load_lora 找不到接收 A、B 的位置。
    load_lora(model, lora_path)
    raw_model = getattr(model, '_orig_mod', model)

    # 先复制所有非 LoRA 参数和持久化 buffer，保证输出仍是完整的基础模型 state_dict。
    state_dict = {k: v.cpu().half() for k, v in raw_model.state_dict().items() if '.lora.' not in k}
    for name, module in raw_model.named_modules():
        # 排除 LoRA 内部的 A/B Linear，避免把增量分支本身当作基础层再次处理。
        if isinstance(module, nn.Linear) and '.lora.' not in name:
            # clone 确保后面对 state_dict 张量的加法不会修改正在使用的模型权重。
            state_dict[f'{name}.weight'] = module.weight.data.clone().cpu().half()
            if hasattr(module, 'lora'):
                # nn.Linear 的权重按 [输出, 输入] 存储，所以合并顺序必须是 B @ A。
                state_dict[f'{name}.weight'] += (module.lora.B.weight.data @ module.lora.A.weight.data).cpu().half()
    torch.save(state_dict, save_path)
