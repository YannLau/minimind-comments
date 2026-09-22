"""MiniMind 的核心模型定义。

本文件把一个 decoder-only（仅解码器）大语言模型需要的主要部件集中在一起：配置、
RMSNorm、RoPE/YaRN 位置编码、GQA 因果自注意力、SwiGLU 前馈网络、可选的 MoE，
以及训练和生成所需的 Hugging Face 接口。

建议先阅读（有助于理解本文件）：
1. ``README.md`` 的“模型结构”“训练”和“RoPE 长度外推”相关章节：先建立整个项目的
   结构与训练流程概念。
2. ``dataset/lm_dataset.py`` 中的 ``PretrainDataset``、``SFTDataset``：理解
   ``input_ids``、``labels``、padding 和 ``-100`` 忽略标签是怎样构造的。
3. ``trainer/train_pretrain.py`` 或 ``trainer/train_full_sft.py``：了解本文件返回的
   ``loss`` 与 ``aux_loss`` 如何参与反向传播。

进一步推荐阅读：
1. ``eval_llm.py``：观察配置、权重加载、聊天模板和本文件 ``generate`` 的实际调用。
2. ``trainer/rollout_engine.py``：理解 ``logits_to_keep``、逐 token 对数概率及强化学习
   rollout 如何复用模型前向传播。
3. ``scripts/convert_model.py``：了解 MiniMind 参数名如何映射到 Qwen3/Qwen3-MoE，
   以及如何导出为 Transformers 格式。
4. ``model/model_lora.py``：继续理解线性投影层怎样被 LoRA 低秩分支改造。

阅读时可始终记住主要张量形状：``B`` 为批大小，``S`` 为序列长度，``D`` 为隐藏维度，
``H`` 为查询头数，``H_kv`` 为键值头数，``D_h`` 为每个注意力头的维度。
"""

import math, torch, torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind 配置
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class MiniMindConfig(PretrainedConfig):
    """保存模型结构和生成所需的超参数，并接入 Transformers 配置体系。

    只有最常调整的三个参数显式列在函数签名中，其余参数通过 ``kwargs`` 读取。这样既能
    保持项目内调用简洁，也能让 ``save_pretrained/from_pretrained`` 从 ``config.json``
    传入完整配置。
    """

    # Transformers 用该字符串识别自定义模型类型，并写入导出的 config.json。
    model_type = "minimind"

    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        # 主干规模：token 向量宽度、Transformer Block 数量，以及是否以 MoE 替换普通 MLP。
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe

        # 词表与特殊 token。默认值需与 model/ 下的 tokenizer 配置保持一致。
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)

        # 注意力采用 GQA：H 个查询头共享 H_kv 组 K/V；当 H_kv < H 时可减少 KV Cache。
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)

        # 默认的 SiLU + 门控乘法构成 SwiGLU。中间层宽度约为 pi*D，并向上对齐到 64。
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)

        # RoPE 表会预计算到最大位置；theta 越大，低频分量变化越慢。
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)

        # 权重绑定让输入 embedding 与输出词表投影共用参数，减少参数量并保持语义空间一致。
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)

        # 推理时可选 YaRN 外推。原始训练上下文按仓库配置视为 2048，factor=16 表示目标伸缩倍数。
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        # MoE 专属配置；use_moe=False 时仍会保存在配置里，但不会创建专家层。
        # 默认有 4 个专家，每个 token 只路由到得分最高的 1 个专家（top-1）。
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind 模型
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class RMSNorm(torch.nn.Module):
    """均方根归一化（RMSNorm）。

    与 LayerNorm 不同，它不减去均值，只按最后一维的均方根缩放，再乘可学习权重。
    计算临时转成 FP32，可降低半精度下平方、求均值和开方的数值误差。
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        # 对每个 token 的 D 个隐藏分量独立归一化，keepdim=True 便于广播回 [B, S, D]。
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 归一化完成后转回输入 dtype，避免后续层因 FP32 结果而增加显存和计算量。
        return (self.weight * self.norm(x.float())).type_as(x)

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    """预计算 RoPE 所需的余弦表与正弦表。

    参数 ``dim`` 是单个注意力头维度 ``D_h``，``end`` 是最大位置数。返回两个
    ``[end, D_h]`` 张量；模型前向时只切出当前 token 所在位置的一段，因此无需每层
    重复计算三角函数。
    """
    # 每两个通道共享一个旋转频率；指数随通道增加，使不同维度覆盖不同位置尺度。
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    # YaRN：f'(i)=f(i)((1-γ)+γ/s)，其中 γ∈[0,1] 是从高频到低频的线性过渡。
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            # 根据“一个上下文窗口内旋转多少圈”求无需插值/完全插值的维度边界。
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            # ramp=0 的频率保持原样，ramp=1 的频率除以 factor，中间频率平滑过渡。
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)
    # 外积得到“每个位置 × 每个频率”的旋转角，再复制一份以匹配前后两个半维度。
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """把 RoPE 旋转应用到 Q、K；位置只影响注意力关系，不直接修改 V。

    输入 Q/K 的布局为 ``[B, S, H, D_h]``。cos/sin 原为 ``[S, D_h]``，在头维插入
    一个轴后即可广播到所有 batch 和注意力头。
    """
    # 将向量 [x1, x2] 变为 [-x2, x1]，与 cos/sin 组合即完成二维平面旋转。
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """把每组 K/V 头复制 ``n_rep`` 次，使其数量与查询头一致。

    GQA 参数投影只生成 ``H_kv`` 个头；注意力矩阵计算前将其逻辑扩展到 ``H`` 个头。
    ``expand`` 本身不复制存储，最后的 ``reshape`` 得到期望布局。
    """
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim))

class Attention(nn.Module):
    """分组查询注意力（GQA），同时支持 Flash Attention、因果遮罩和 KV Cache。"""

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        # 若未指定 KV 头数就退化为标准多头注意力；否则每 n_rep 个 Q 头共享一组 K/V。
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.is_causal = True

        # 输入 x:[B,S,D]。Q 输出 H*D_h，K/V 只输出 H_kv*D_h，这是 GQA 节省参数和缓存的来源。
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        # attn_dropout 作用于注意力概率，resid_dropout 作用于输出投影。
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        # PyTorch 提供 SDPA 且配置允许时，优先走通常更省显存的融合实现。
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        """计算一层自注意力。

        ``past_key_value`` 是当前层历史 K/V，形状均为 ``[B, S_past, H_kv, D_h]``；
        ``attention_mask`` 通常是 ``[B, S_total]``，1 表示有效 token、0 表示 padding。
        返回值为注意力输出 ``[B,S,D]`` 和可选的新 KV Cache。
        """
        bsz, seq_len, _ = x.shape
        # 线性投影后拆出头维：[B,S,D] -> Q:[B,S,H,D_h]，K/V:[B,S,H_kv,D_h]。
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        # QK-Norm 可稳定注意力分数范围；随后只给 Q/K 注入当前 token 的位置信息。
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        if past_key_value is not None:
            # 增量生成时，新 K/V 接到历史缓存末尾；Q 只包含本次新输入，无需缓存。
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        # 注意力算子要求 [B,H,S,D_h]；同时把分组 K/V 扩展到与 Q 相同的头数。
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))
        # 无历史缓存、无 padding 且一次处理多个 token 时，可直接让 SDPA 构造标准因果遮罩。
        # 带 KV Cache 时 Q 长度和 K 长度不同，故回退到下方手写路径以精确对齐遮罩。
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
        else:
            # scores:[B,H,S_query,S_key]。除以 sqrt(D_h) 防止点积随维度增大而使 softmax 饱和。
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # 只对右下角“本次新增 token 之间”的区域加上上三角 -inf；历史 token 均可被看到。
            if self.is_causal: scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
            # padding 位置加一个极小值，使其 softmax 概率趋近于 0。
            if attention_mask is not None: scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
        # 合并多头后做输出投影，恢复 [B,S,D]，供残差连接使用。
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv

class FeedForward(nn.Module):
    """SwiGLU 风格的门控前馈网络：down(SiLU(gate(x)) * up(x))。"""

    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # 两条上投影分支逐元素相乘，让网络能以输入相关的“门”筛选中间特征。
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class MOEFeedForward(nn.Module):
    """稀疏混合专家前馈层（MoE）。

    路由器为每个 token 选择 top-k 个 ``FeedForward`` 专家。这里只有被选中的专家参与
    该 token 的前向计算；辅助损失则鼓励 token 不要长期集中到少数专家。
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        """将 ``[B,S,D]`` token 路由给专家，再聚合回相同形状。"""
        batch_size, seq_len, hidden_dim = x.shape
        # 路由互不依赖 batch/序列结构，先展平为 N=B*S 个 token。
        x_flat = x.view(-1, hidden_dim)
        # scores:[N,E] 是每个 token 选择各专家的概率；topk_idx:[N,K] 保存被选专家编号。
        scores = F.softmax(self.gate(x_flat), dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
        if self.config.norm_topk_prob:
            if self.config.num_experts_per_tok > 1:
                # top-k 路由时，把入选专家权重重新归一化为和等于 1。
                topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
            else:
                # top-1 前向权重刻意固定为 1，但用直通估计器保留路由概率的梯度。
                top1 = torch.topk(F.softmax(self.gate(x_flat.detach()), dim=-1), k=1, dim=-1, sorted=False)[0]
                topk_weight = top1 - top1.detach() + 1.0  # k=1：前向值为 1.0，梯度经直通估计器传播
        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            # mask:[N,K] 指示 token 的第几个候选是否为当前专家 i。
            mask = (topk_idx == i)
            if mask.any():
                # 收集发往该专家的 token，计算后按路由权重缩放，再累加回原 token 位置。
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1, 1)
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                # 即使本批没有 token 命中该专家，也用“0×参数”把它接入计算图，兼容 DDP 梯度同步。
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
        if self.training and self.config.router_aux_loss_coef > 0:
            # load 表示实际分配比例，scores.mean 表示路由器期望比例；二者内积越大通常越不均衡。
            # 乘专家数使均匀路由附近的尺度保持稳定，再由系数控制它对总训练损失的影响。
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            # 稠密推理或关闭辅助项时仍返回同设备标量，便于上层统一相加。
            self.aux_loss = scores.new_zeros(1).squeeze()
        return y.view(batch_size, seq_len, hidden_dim)

class MiniMindBlock(nn.Module):
    """一个 Pre-Norm Transformer Block：注意力子层 + 前馈/MoE 子层。"""

    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        # layer_id 暂未参与计算，但保留该参数便于未来做逐层配置或调试定位。
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        # 第一条残差支路：RMSNorm -> 自注意力 -> 与原输入相加。
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        # 第二条残差支路：RMSNorm -> SwiGLU（或 MoE）-> 与注意力结果相加。
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value

class MiniMindModel(nn.Module):
    """不含词表输出头的 MiniMind Transformer 主干。"""

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        # token id -> D 维向量；随后依次通过 N 个结构相同但参数独立的 Block。
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # cos/sin 是可由配置重新计算的派生数据，所以注册为 non-persistent buffer：会跟随
        # model.to(device) 移动，但不会写进 state_dict，从而减小权重文件体积。
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        """运行 Transformer 主干。

        参数：
            input_ids: ``[B,S]`` 的 token id。
            attention_mask: ``[B,S_total]``，屏蔽 padding；增量生成时包含历史和当前位置。
            past_key_values: 长度为层数的列表，每项是该层历史 ``(K,V)``。
            use_cache: 是否返回更新后的逐层 KV Cache。

        返回 ``(hidden_states, presents, aux_loss)``：最终隐藏状态、各层缓存，以及所有
        MoE 层的负载均衡辅助损失之和。
        """
        batch_size, seq_length = input_ids.shape
        # Transformers 5.x 可能传入新版 Cache 对象；当前轻量实现只处理旧式 (K,V) 列表。
        if hasattr(past_key_values, 'layers'): past_key_values = None
        # 首次前向为每层补 None；增量前向则从第一层缓存长度推断当前 token 的绝对起点。
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        # Transformers>=5.x 在 meta device 初始化时可能丢失非持久化 RoPE buffer；检测后重算。
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
        # 只切取本次输入对应的位置。例如缓存已有 10 个 token，则新 token 从位置 10 开始。
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])
        presents = []
        # 每一层消费自己的历史缓存，并产生自己的新缓存；隐藏状态则按层顺序传递。
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
        # 最后一层之后再归一化，这是常见的 Pre-Norm decoder-only 架构收尾方式。
        hidden_states = self.norm(hidden_states)
        # 稠密模型没有 MOEFeedForward，此时 sum 从同设备的 0 标量开始，统一返回接口。
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss

class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    """在 MiniMind 主干上增加语言模型头，并提供训练损失与自回归生成接口。"""

    # 告诉 Transformers 应使用哪个配置类；权重映射则声明 lm_head 与 embedding 可绑定。
    config_class = MiniMindConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        # 对每个位置把 D 维隐藏状态投影为 vocab_size 个“未归一化对数概率”（logits）。
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # 直接让两个 Parameter 引用同一份权重，而非复制数值。
        if self.config.tie_word_embeddings: self.model.embed_tokens.weight = self.lm_head.weight
        # 执行 PreTrainedModel 约定的权重初始化与绑定收尾工作。
        self.post_init()

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        """完成主干前向、词表投影，并在提供标签时计算 next-token loss。

        ``logits_to_keep=0`` 利用 ``-0 == 0`` 保留全部位置；传入正整数时只投影最后若干
        位置，可供 RL rollout 等场景节省显存。训练时标签通常与 ``input_ids`` 等长，
        其中 ``-100`` 的位置会被交叉熵忽略。
        """
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        # 整数表示“保留末尾几个位置”，也允许高级调用者直接传入切片/索引。
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            # 因果语言模型用位置 t 的输出预测位置 t+1，故 logits 去尾、labels 去头后对齐。
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            # dataset/lm_dataset.py 将 padding 或无需监督的位置标成 -100，在这里自动忽略。
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        # 复用 Transformers 标准输出类型，使 .loss/.logits/.past_key_values 等属性可直接访问。
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
    
    # 生成实现的相关讨论：https://github.com/jingyaogong/minimind/discussions/611
    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        """逐 token 自回归生成文本。

        这是一个刻意保持直观的轻量生成循环，用来替代 ``GenerationMixin`` 中功能更全但
        调用链更长的实现。每轮只取最后一个位置的 logits，经温度、重复惩罚、top-k 和
        top-p 过滤后采样下一个 token，再把它追加到序列末尾。

        关键参数：
            inputs/input_ids: 初始提示词 ``[B,S_prompt]``；两种名称均兼容。
            max_new_tokens: 最多新生成多少个 token，不包含提示词长度。
            temperature: logits 的缩放温度，越小分布越尖锐。
            top_k/top_p: 分别执行候选数量截断和累计概率（nucleus）截断。
            use_cache: 为 True 时每层保存历史 K/V，后续每轮通常只需计算一个新 token。
            num_return_sequences: 将每条输入复制多少份，随机采样时可得到多个候选答案。
            streamer: Transformers 的流式输出器；存在时会立即推送提示词和每个新 token。

        返回完整 token 序列（提示词 + 新生成内容）；若额外传入 ``return_kv=True``，则
        返回同时包含序列与最终 KV Cache 的字典。
        """
        # 优先使用关键字 input_ids，以兼容 Hugging Face 常见调用；随后复制出候选序列。
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)
        # finished 按 batch 分别记录是否已生成 EOS；已结束序列后续只会继续填 EOS 以对齐长度。
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer: streamer.put(input_ids.cpu())
        for _ in range(max_new_tokens):
            # 有缓存时跳过已计算的前缀，只把缓存之后的新 token 送入模型；首次则输入完整提示词。
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
            # 新生成的 token 都是有效位置，因此在 mask 右侧追加一列 1，供下一轮使用。
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
            # 只用序列最后位置预测下一个 token；除以温度改变分布尖锐程度而不改变排序。
            logits = outputs.logits[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                # 对已经出现过的 token 降权：正 logits 做除法，负 logits 做乘法，二者都会变小。
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i]); score = logits[i, seen]; logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
            if top_k > 0:
                # 低于第 k 大分数的候选置为 -inf，使其 softmax 概率严格为 0。
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            if top_p < 1.0:
                # 按概率从高到低累加，只保留累计质量不超过 top_p 的最小候选集合。
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                # 右移一位可确保第一个使累计概率越过阈值的 token 也被保留，且至少留一个候选。
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                # 将排序空间中的 mask 散射回原词表索引，再屏蔽对应 logits。
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            # 采样模式从概率分布抽取；贪心模式直接选最大 logits。
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            # batch 内已完成的样本固定补 EOS，避免它们继续产生有意义的 token。
            if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer: streamer.put(next_token.cpu())
            if eos_token_id is not None:
                # 所有样本都遇到 EOS 即提前结束，否则继续到 max_new_tokens 上限。
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break
        if streamer: streamer.end()
        # return_kv 是本项目扩展参数，适合调用方跨阶段复用已经计算好的上下文缓存。
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids
