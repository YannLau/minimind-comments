"""强化学习训练使用的在线生成（rollout）引擎。

本文件把“给定 prompt 采样回答”和“把更新后的策略同步给采样端”统一成一个接口。
训练脚本因此不必关心回答是在当前 PyTorch 进程里生成，还是由独立的 SGLang 服务生成。
一次典型的同步训练循环是：rollout 引擎用当前策略采样并记录旧策略的逐 token log 概率，
训练脚本据此计算奖励和损失、更新策略，随后调用 ``update_policy`` 让下一批 rollout 使用
更新后的权重。生成阶段不参与反向传播；策略梯度在训练脚本中通过重新计算 log 概率得到。

建议先阅读（有助于理解本文件）：
1. ``trainer/train_grpo.py`` 或 ``trainer/train_ppo.py``：先看训练脚本怎样调用
   ``rollout``，以及 ``RolloutResult`` 的哪些字段会进入奖励、策略损失和回答掩码。
2. ``model/model_minimind.py`` 中的 ``MiniMindForCausalLM.forward`` 和 ``generate``：
   理解自回归模型如何生成 token、位置 t 的 logits 如何预测位置 t+1，以及
   ``logits_to_keep`` 怎样只计算回答附近的 logits。
3. ``trainer/trainer_utils.py`` 中的 ``init_model``：了解策略模型、tokenizer 和设备的来源。
4. ``dataset/lm_dataset.py`` 中的 ``RLAIFDataset``：了解在线训练批次里的 prompt 是怎样
   构造的，以及为什么回答由策略在 rollout 阶段生成。

进一步推荐阅读：
1. ``trainer/train_agent.py`` 的 ``rollout_single``：查看单条回答如何扩展为多轮工具交互，
   以及工具观察为什么不算作策略生成的动作。
2. ``trainer/train_grpo.py``、``trainer/train_ppo.py`` 和 ``trainer/train_agent.py``：比较
   三种训练流程如何消费同一份 rollout 结果。
3. ``README.md`` 的 Agentic RL / SGLang 章节：查看训推分离的整体流程和 SGLang 启动方式。

张量形状约定：B 是 prompt 数，G 是每个 prompt 的生成数，N=B*G，P 是输入张量中的
prompt 宽度，R 是批次中回答的补齐宽度。生成结果按 prompt 展开：同一个 prompt 的 G 条
回答连续排列。Torch 后端保留左补齐后的统一 prompt 宽度；SGLang 后端先去掉左侧 padding，
所以不同样本的真实 prompt 长度可能不同。``prompt_lens`` 表示回答在 ``output_ids`` 中的
起始位置，而 ``completion_mask`` 标记后端认为属于回答的列（SGLang 会把批内补齐列标为 0；
Torch 当前将整个返回区标为 1）。EOS 及 EOS 后的填充值是否进入损失由训练调用方决定。当前
接口按批次同步采样和更新，不是异步 rollout buffer。

如果使用 SGLang 加速，请先在仓库根目录启动 Transformers 格式的模型：
``python -m sglang.launch_server --model-path ./minimind-3 --attention-backend triton --host 0.0.0.0 --port 8998```
"""
import os
import sys

__package__ = "trainer"
# 允许从仓库根目录或直接运行 trainer/ 下的训练脚本；模型和数据集按仓库绝对包路径导入。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import requests
import torch
import torch.distributed as dist
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional, Tuple
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer


def compute_per_token_logps(model, input_ids: Tensor, n_keep: int, attention_mask: Optional[Tensor] = None) -> Tensor:
    """计算序列末尾 ``n_keep`` 个目标 token 在模型下的 log 概率。

    因果语言模型在位置 ``t`` 输出的 logits 预测位置 ``t+1`` 的 token。若输入是
    ``[prompt, completion]``，这里通常只取 completion 的概率，不必为整段序列都保留词表
    维 logits。模型的 ``logits_to_keep`` 参数因此请求末尾 ``n_keep + 1`` 个位置：多出的
    一个位置用于预测回答的第一个 token，随后去掉最后一个没有对应目标 token 的 logits。

    参数：
        model: 当前策略模型；若外面包着 DDP，则解开包装后直接调用内部模型。
        input_ids: 完整输入 token，形状 ``[N, S]``，最后 ``n_keep`` 个 token 是待打分目标。
        n_keep: 每行要打分的末尾 token 数；零或负数时返回 ``[N, 0]`` 空张量。
        attention_mask: 可选的 ``[N, S]`` 注意力掩码，1 表示有效上下文，0 表示 padding。

    返回：
        ``[N, n_keep]`` 浮点张量，每个元素是对应目标 token 的 log-softmax 概率。
        此函数本身不包裹 ``no_grad``；调用方应根据用途决定是否需要梯度。
    """
    # 没有待评分 token 时显式返回空的二维张量，避免后面的切片和 stack 无法处理空列表。
    if n_keep <= 0:
        return input_ids.new_empty((input_ids.size(0), 0), dtype=torch.float32)
    # 解开 DDP 后直接调用模型本体；rollout 调用方通常已用 no_grad 包住这次前向。
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    # inference_mode 创建的张量不能在某些后续算子中作为普通输入保存；复制后再交给模型。
    input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids
    # 末尾 n_keep+1 个 logits 覆盖“prompt 最后位置预测首个回答 token”到“回答倒数第二位预测末位”。
    logits = unwrapped(input_ids, attention_mask=attention_mask, logits_to_keep=n_keep + 1).logits[:, :-1, :]
    per_token_logps = []
    # 逐行 gather 每个实际 token 的词表概率；避免一次性构造完整的 [N, n_keep, vocab] 中间张量。
    for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
        ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row
        per_token_logps.append(
            torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1)
        )
    return torch.stack(per_token_logps)


@dataclass
class RolloutResult:
    """一次批量采样的统一返回结构；两个后端都必须填充这些字段。

    各字段第 0 维都对应展平后的 N=B*G 条回答，并保持“同一 prompt 的生成相邻”的顺序。
    后端间的序列长度可能不同，因此部分张量会在右侧补 PAD/0。``completion_mask`` 能标出
    SGLang 的批内补齐位置；Torch 后端当前把整个返回回答区标为 1，EOS 后的填充需由调用方处理。
    """
    # prompt 与回答拼接后的 token 序列 [N, S]；SGLang 的变长行在末尾补齐。
    output_ids: Tensor
    # 仅含回答 token 的矩阵 [N, R]；SGLang 会在较短回答右侧补 tokenizer.pad_token_id。
    completion_ids: Tensor
    # rollout 时策略对生成 token 给出的旧 log 概率 [N, R]；补齐位置为 0 占位。
    per_token_logps: Tensor
    # 每条回答解码后的字符串，顺序与上述张量的行一致，供奖励函数或日志使用。
    completions: List[str]
    # 每行回答在 output_ids 中开始的位置；Torch 是统一补齐宽度，SGLang 是真实 prompt 长度。
    prompt_lens: Tensor
    # [N, R] 后端回答区掩码；Torch 当前全为 1，SGLang 的批内右补齐位置为 0；EOS 由调用方处理。
    completion_mask: Tensor


class RolloutEngine(ABC):
    """采样后端的最小公共接口。

    具体引擎只需实现批量 ``rollout`` 和策略同步 ``update_policy``；训练代码依赖这两个
    方法及 ``RolloutResult``，不直接依赖后端的 HTTP API 或模型生成实现。
    """
    tokenizer = None
    
    @abstractmethod
    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        """为每个 prompt 采样指定数量的回答，并返回 token、文本和旧策略 log 概率。

        输入 ``prompt_ids`` 与 ``attention_mask`` 形状均为 ``[B, P]``。输出首维为
        ``B * num_generations``；同一个 prompt 的多条回答必须连续排列，以便训练脚本分组。
        """
        pass
    
    @abstractmethod
    def update_policy(self, model: torch.nn.Module):
        """使下一次采样使用传入的最新策略参数。"""
        pass


class TorchRolloutEngine(RolloutEngine):
    """在训练进程内直接调用 PyTorch 模型 ``generate`` 的后端。"""
    def __init__(self, policy_model: torch.nn.Module, tokenizer, device: str = "cuda", autocast_ctx=None):
        # policy_model 由训练脚本传入；DDP 包装的模型也可保存，生成时再解开包装。
        self.policy_model = policy_model
        self.tokenizer = tokenizer
        # device 为统一工厂接口保留；实际张量设备由调用方的 prompt_ids/model 决定。
        self.device = device
        # 可选混合精度上下文与训练模型共享，降低生成前向的显存占用。
        self.autocast_ctx = autocast_ctx
    
    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        """对输入 prompt 采样，并用同一个策略重算生成 token 的 log 概率。"""
        # DDP 的 wrapper 不一定暴露模型自定义的 generate 方法，因此使用内部 module 生成。
        model = self.policy_model.module if isinstance(self.policy_model, DistributedDataParallel) else self.policy_model
        ctx = self.autocast_ctx if self.autocast_ctx else nullcontext()
        # 采样只收集训练数据，不需要保存反向传播图；autocast 仅影响本地模型推理精度。
        with torch.no_grad(), ctx:
            output_ids = model.generate(
                # repeat_interleave 的顺序是 prompt0*G、prompt1*G……，与 GRPO/Agent 分组约定一致。
                input_ids=prompt_ids.repeat_interleave(num_generations, dim=0),
                attention_mask=attention_mask.repeat_interleave(num_generations, dim=0),
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            ).clone()  # [N, P+R]；clone 将生成阶段的推理张量变为普通张量。
            # Torch 的 generate 对整批使用统一 prompt 宽度 P；左侧 padding 仍占据这段宽度。
            prompt_len = prompt_ids.size(1)
            completion_ids = output_ids[:, prompt_len:]  # [N, R]，截去统一宽度的 prompt 部分。
            # 前缀掩码保留 prompt 左 padding 的无效位置；生成区在 attention 上全部设为有效。
            full_mask = torch.cat([attention_mask.repeat_interleave(num_generations, dim=0), attention_mask.new_ones(output_ids.size(0), completion_ids.size(1))], dim=1)
            # 这里对原始模型 logits 取概率；temperature 影响 generate 的抽样分布，但未应用于这次重算。
            # 训练调用方将返回值作为 rollout 时策略的旧 log 概率，用于重要性比率或轨迹训练。
            per_token_logps = compute_per_token_logps(self.policy_model, output_ids, completion_ids.size(1), attention_mask=full_mask)
        # skip_special_tokens 影响可读文本，不会改动训练实际使用的 token 张量。
        completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        # Torch 输出长度统一，因此当前引擎将返回的整个回答区都视为存在；调用方仍需处理 EOS。
        return RolloutResult(output_ids, completion_ids, per_token_logps, completions,
                             prompt_ids.new_full((output_ids.size(0),), prompt_len),
                             attention_mask.new_ones(output_ids.size(0), completion_ids.size(1)))
    
    def update_policy(self, model: torch.nn.Module):
        # 本地采样直接持有最新模型对象，不需要把参数重新写盘或通过网络传输。
        self.policy_model = model


class SGLangRolloutEngine(RolloutEngine):
    """通过 HTTP 请求外部 SGLang 服务采样，并通过共享磁盘同步策略权重。"""
    def __init__(self, base_url: str, model_path: str, shared_ckpt_path: str = "./sglang_ckpt", timeout: int = 120):
        # 去掉 URL 末尾斜杠，避免拼接 endpoint 时出现双斜杠。
        self.base_url = base_url.rstrip('/')
        # 训练进程将模型保存到此目录；该目录必须能被 SGLang 服务进程读取。
        self.shared_ckpt_path = shared_ckpt_path
        self.timeout = timeout
        # HTTP 服务端负责模型推理；训练端仍需要同词表 tokenizer 来编码/解码 token。
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # 保存 requests 模块引用，便于 rollout 与权重同步共用同一 HTTP 客户端。
        self.http = requests
    
    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        """将有效 prompt token 发给 SGLang，并把变长 HTTP 结果补成批量张量。"""
        # 训练批次通常采用左补齐；按 attention_mask 过滤 padding 后，发送服务端的是紧凑序列。
        input_ids_list = []
        for ids, mask in zip(prompt_ids, attention_mask):
            valid_ids = ids[mask.bool()].tolist()
            input_ids_list.append(valid_ids)
        # 每个 prompt 连续复制 G 次，维持与 Torch 后端相同的行排序约定。
        all_input_ids = [ids for ids in input_ids_list for _ in range(num_generations)]
        
        # SGLang /generate 接口接收 token ID 而不是文本；return_logprob 请求采样时的旧策略概率。
        payload = {
            "input_ids": all_input_ids,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "stop_token_ids": [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else [],
            },
            "return_logprob": True,
        }
        
        # 网络或服务端错误会由 raise_for_status 转成异常，交给训练入口显示并中止当前批次。
        resp = self.http.post(f"{self.base_url}/generate", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        
        results = resp.json()
        # 单请求时服务可能直接返回对象；统一包装成列表，简化后续逐条解析逻辑。
        if not isinstance(results, list):
            results = [results]
        
        # 分别收集完整序列、回答 token、回答 log 概率与可读文本，稍后再做批内补齐。
        all_output_ids, all_completion_ids, all_logprobs = [], [], []
        completions = []
        
        for i, result in enumerate(results):
            # SGLang 通常将生成元信息放在 meta_info；保留顶层 output_ids 作为兼容回退。
            meta = result.get("meta_info", {})
            completion_ids = meta.get("output_ids", result.get("output_ids", []))
            raw_logprobs = meta.get("output_token_logprobs", [])
            
            logprobs = []
            # SGLang 的 token logprob 常以 [logprob, token_id, token_text] 一类 tuple 返回；
            # 也兼容服务端直接返回数值的情况。
            for item in raw_logprobs:
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    logprobs.append(item[0])
                elif isinstance(item, (int, float)):
                    logprobs.append(item)
            
            # 对齐服务端两组数组长度：缺失概率用 0 占位，多出的概率只保留末尾对应项。
            # 0 只是形状补齐值，不代表模型真实的 log 概率；完整旧概率仍依赖服务端正常返回。
            if len(logprobs) < len(completion_ids):
                logprobs = [0.0] * (len(completion_ids) - len(logprobs)) + logprobs
            elif len(logprobs) > len(completion_ids):
                logprobs = logprobs[-len(completion_ids):] if completion_ids else []
            # 请求和响应按输入顺序一一对应；将原 prompt 与生成 token 拼回完整序列。
            prompt = all_input_ids[i]
            full_output = prompt + completion_ids
            all_output_ids.append(full_output)
            all_completion_ids.append(completion_ids)
            all_logprobs.append(logprobs)
            completions.append(self.tokenizer.decode(completion_ids, skip_special_tokens=True))
        
        device = prompt_ids.device
        # 至少保留一列回答宽度，使全空回答也能形成合法二维张量；不同回答以右侧 PAD 对齐。
        max_comp_len = max(1, max(len(ids) for ids in all_completion_ids))
        # prompt 本身也可能变长，因此完整序列宽度取最长 prompt + 最长回答。
        max_out_len = max(len(ids) for ids in all_input_ids) + max_comp_len
        
        def pad_to_tensor(seqs, max_len, pad_val=0):
            """将 Python token/概率列表在右侧补指定值，再放到 prompt 所在设备。"""
            return torch.tensor([s + [pad_val] * (max_len - len(s)) for s in seqs], device=device)
        
        pad_id = self.tokenizer.pad_token_id
        # prompt_lens 是每条未补齐 prompt 的真实长度，即其回答在 output_ids 中的起点。
        # completion_mask 让训练脚本能忽略 SGLang 为短回答补出的 PAD 位置。
        return RolloutResult(
            output_ids=pad_to_tensor(all_output_ids, max_out_len, pad_val=pad_id),
            completion_ids=pad_to_tensor(all_completion_ids, max_comp_len, pad_val=pad_id),
            per_token_logps=pad_to_tensor(all_logprobs, max_comp_len, pad_val=0.0),
            completions=completions,
            prompt_lens=torch.tensor([len(ids) for ids in all_input_ids], device=device),
            completion_mask=torch.tensor([[1] * len(ids) + [0] * (max_comp_len - len(ids)) for ids in all_completion_ids], device=device),
        )
    
    def update_policy(self, model: torch.nn.Module):
        """将训练端当前策略导出到共享目录，并通知 SGLang 加载新权重。

        DDP 下仅 global rank 0 写文件并请求服务端，其余 rank 等待广播同步结果和 barrier；
        这样可避免多个进程同时覆盖同一 checkpoint 或重复更新服务。
        """
        ok = True
        if not dist.is_initialized() or dist.get_rank() == 0:
            try:
                # 去掉 DDP 包装；torch.compile 模型再取回原始模块，调用 Hugging Face 保存接口。
                unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
                unwrapped = getattr(unwrapped, '_orig_mod', unwrapped)
                abs_path = os.path.abspath(self.shared_ckpt_path)
                # 暂存半精度 CPU 权重，降低共享 checkpoint 的体积和 GPU 峰值显存占用。
                state_dict = {k: v.detach().half().cpu() for k, v in unwrapped.state_dict().items()}
                # 同目录保存模型配置/权重和 tokenizer，确保 SGLang 看到一套相匹配的模型文件。
                unwrapped.save_pretrained(abs_path, state_dict=state_dict, safe_serialization=False)
                self.tokenizer.save_pretrained(abs_path)
                # 服务端需能访问 abs_path 对应的共享文件系统，然后重新加载该目录中的权重。
                resp = self.http.post(f"{self.base_url}/update_weights_from_disk", json={"model_path": abs_path}, timeout=self.timeout)
                if resp.status_code != 200: print(f"[SGLANG WARNING] update_weights 失败: {resp.status_code}, {resp.text}")
                ok = resp.status_code == 200
            except Exception as e:
                print(f"[SGLANG WARNING] update_weights 异常: {e}"); ok = False
        if dist.is_initialized():
            # 将 rank 0 的成功/失败状态广播给所有训练进程，再一起越过同步点。
            ok_t = torch.tensor(int(ok), device=next(model.parameters()).device)
            dist.broadcast(ok_t, src=0); dist.barrier(); ok = bool(ok_t.item())
        # 不能静默继续使用旧服务端权重，否则训练数据对应的策略与更新后的 Policy 不一致。
        if not ok: raise RuntimeError("SGLang update_policy failed")
        return ok
    
    def flush_cache(self) -> bool:
        """请求 SGLang 清空推理缓存；返回服务端是否以 HTTP 200 确认。"""
        resp = self.http.post(f"{self.base_url}/flush_cache", timeout=30)
        return resp.status_code == 200
    
    def health(self) -> bool:
        """探测 SGLang 健康检查接口；连接失败或非 200 状态均视为不健康。"""
        try:
            resp = self.http.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except:
            return False


def create_rollout_engine(
    engine_type: str = "torch",
    policy_model: torch.nn.Module = None,
    tokenizer = None,
    device: str = "cuda",
    autocast_ctx = None,
    sglang_base_url: str = None,
    sglang_model_path: str = None,
    sglang_shared_path: str = None,
) -> RolloutEngine:
    """按名称创建 rollout 后端，供 PPO、GRPO 和 Agent 训练脚本共用。

    ``torch`` 需要当前进程中的策略模型与 tokenizer；``sglang`` 需要服务 URL、可读取的
    tokenizer 模型目录，以及训练端和推理端共同可见的权重目录。其他名称会立即报错，避免
    因配置拼写错误而意外选择后端。
    """
    if engine_type == "torch":
        # 本地后端保留模型对象，更新策略时只替换它的引用。
        return TorchRolloutEngine(policy_model, tokenizer, device, autocast_ctx)
    elif engine_type == "sglang":
        # 远端后端根据 URL 发请求，并在构造时从模型目录加载用于编码/解码的 tokenizer。
        return SGLangRolloutEngine(sglang_base_url, sglang_model_path, sglang_shared_path)
    else:
        raise ValueError(f"不支持的引擎类型: {engine_type}")
