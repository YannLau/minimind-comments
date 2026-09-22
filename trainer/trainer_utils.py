"""MiniMind 各训练脚本共用的基础设施。

本文件不负责某一种具体训练算法，而是集中提供训练入口都会反复用到的“小积木”：
模型参数统计、仅主进程打印、余弦学习率、DDP 初始化、随机种子、断点保存/恢复、
模型与分词器初始化、恢复训练时跳过旧批次，以及强化学习训练所需的奖励模型和安全
数学表达式求值器。

建议先阅读（有助于理解本文件）：
1. ``model/model_minimind.py`` 中的 ``MiniMindConfig``、``MOEFeedForward`` 和
   ``MiniMindForCausalLM``：理解这里创建的模型、MoE 配置和参数命名。
2. ``trainer/train_pretrain.py``：观察 ``get_lr``、``init_model``、``lm_checkpoint``、
   ``init_distributed_mode``、``setup_seed`` 和 ``SkipBatchSampler`` 如何串起完整训练流程。
3. ``dataset/lm_dataset.py`` 中的 ``PretrainDataset``、``SFTDataset``：理解采样器产生的
   “样本下标”最终怎样变成模型输入。

进一步推荐阅读：
1. ``trainer/train_full_sft.py`` 与 ``trainer/train_dpo.py``：比较监督微调和偏好对齐如何
   复用同一套训练设施。
2. ``trainer/train_grpo.py``、``trainer/train_ppo.py`` 和 ``trainer/train_agent.py``：理解
   ``LMForRewardModel`` 在强化学习奖励计算中的用途。
3. ``trainer/rollout_engine.py``：继续了解策略模型如何批量生成回复并计算逐 token 概率。
4. ``scripts/eval_toolcall.py`` 与 ``scripts/web_demo.py``：查看 ``safe_math_eval`` 怎样执行
   模型生成的计算器参数，同时避免直接使用危险的 ``eval``。

路径参数大多采用仓库训练命令所处位置对应的相对路径；如果从其他工作目录直接调用
这些函数，应显式传入 ``tokenizer_path``、``save_dir`` 等路径。
"""
import os
import sys
# 把仓库根目录加入模块搜索路径，使直接运行 trainer/ 下脚本时也能导入 model、dataset。
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ast
import operator
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from model.model_minimind import MiniMindForCausalLM

def get_model_params(model, config):
    """打印模型总参数量；对于 MoE 模型，同时估算单个 token 的激活参数量。

    参数量统一除以 ``1e6``，输出单位为 M（百万）。普通稠密模型的每个 token 都会经过
    全部参数，因此只打印总量。MoE 模型虽然保存了所有路由专家，但每个 token 只会进入
    ``num_experts_per_tok`` 个专家，因此还会打印 ``AxxM``（active parameters）。

    这里通过参数名中的 ``mlp.experts.0.`` 统计“一个路由专家”的大小，再乘专家个数；
    这依赖 ``model/model_minimind.py`` 的命名约定。共享专家始终参与计算，所以总量与激活
    量都包含全部共享专家。``n_routed_experts`` 是对其他兼容配置的适配，MiniMind 自身使用
    ``num_experts``。
    """
    # numel() 返回张量元素个数；生成器求和不会额外复制参数。
    total = sum(p.numel() for p in model.parameters()) / 1e6
    # getattr 的嵌套默认值让本函数也兼容采用 n_routed_experts 命名的 MoE 配置。
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)
    # 只数编号 0 的专家，用它代表每个同构专家的参数量，避免逐个专家重复求和。
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6
    # base 是剔除所有专家后的公共骨干；active 再加回一次前向真正会用到的专家。
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: Logger(f'Model Params: {total:.2f}M')


def is_main_process():
    """判断当前进程是否负责日志和保存等只应执行一次的操作。

    单进程训练没有初始化 ``torch.distributed``，自然视为主进程；DDP 训练中只有全局
    rank 0 是主进程。注意 local rank 只是单台机器内的 GPU 编号，不能替代全局 rank。
    """
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    """仅由主进程打印内容，防止 DDP 的每张 GPU 重复输出同一行日志。"""
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    """按余弦曲线把学习率从 ``lr`` 平滑衰减到 ``0.1 * lr``。

    当 ``current_step=0`` 时，括号内为 ``0.1 + 0.45*2 = 1``；到达
    ``total_steps`` 时余弦为 -1，只剩初始学习率的 10%。本函数没有 warmup，调用方会在
    每一步把返回值写进优化器的各 ``param_group['lr']``。
    """
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))


def init_distributed_mode():
    """根据 ``torchrun`` 注入的环境变量初始化单机/多机 DDP，并返回本地 GPU 编号。

    没有 ``RANK`` 表示普通单进程启动，此时返回 0 但不会初始化进程组。由 ``torchrun``
    启动时使用 NCCL 后端，并把当前进程绑定到 ``LOCAL_RANK`` 对应的 CUDA 设备。调用方
    应再通过 ``dist.is_initialized()`` 区分两种模式，不能只根据返回值判断。
    """
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非 DDP 模式；保持返回整数，方便调用方统一初始化 local_rank。

    # torchrun 还会提供 RANK、WORLD_SIZE、MASTER_ADDR、MASTER_PORT 等初始化信息。
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    # 此后未显式指定设备的新 CUDA 张量会落到当前进程负责的 GPU 上。
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    """为 Python、NumPy 和 PyTorch 设置随机种子，并优先保证 CUDA 结果可复现。

    随机种子相同只是在相同软硬件、算子和执行顺序下尽量复现；分布式规约、某些 CUDA
    算子或版本差异仍可能带来细小偏差。关闭 cuDNN benchmark 会牺牲部分性能，以避免它
    根据运行时测量选择不同算法。训练脚本通常给不同 rank 加偏移，并在每个 epoch 重设
    种子，从而兼顾各卡随机性与断点恢复时的数据顺序。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # manual_seed 覆盖 CPU；下面两行覆盖当前 GPU 以及所有可见 GPU。
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    """保存或加载训练断点；是否传入 ``model`` 决定工作模式。

    保存模式（``model is not None``）会生成两个文件：

    * ``{weight}_{hidden_size}[_moe].pth``：仅含 FP16 CPU 模型权重，体积较小，适合后续
      阶段通过 :func:`init_model` 加载或用于推理。
    * 同名前缀的 ``_resume.pth``：除模型外还包含优化器、epoch、step、训练时 GPU 数和
      SwanLab/W&B 实验 ID，并接纳 ``scaler``、``scheduler`` 等额外状态，供精确续训。

    加载模式（``model is None``）只读取 ``_resume.pth``；文件不存在则返回 ``None``。
    写入都先落到 ``.tmp`` 再用 ``os.replace`` 原子替换，可降低进程中断留下半个文件的
    风险。调用方负责只让主进程保存；本函数自身不做 rank 判断。
    """
    os.makedirs(save_dir, exist_ok=True)
    # 文件名把稠密模型和 MoE 模型分开，避免相同 hidden_size 时相互覆盖。
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:
        # DDP 和 torch.compile 都会包一层模块；保存时要还原到原始模型，保持参数名前缀稳定。
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = raw_model.state_dict()
        # 转 FP16 并移到 CPU，既缩小磁盘文件，也避免 torch.save 长时间占用 GPU 引用。
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        wandb_id = None
        if wandb:
            # SwanLab 的 wandb 兼容层提供 get_run；原生风格对象可能直接暴露 id。
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        # kwargs 让不同训练任务按需保存 scaler、scheduler 等，而无需不断扩展函数签名。
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    # 额外对象若同样经过 DDP/compile 包装，也剥去包装后再保存状态。
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, resume_data
        # 删除 CPU 字典后清理 PyTorch 的空闲 CUDA 缓存；不会释放仍被模型引用的显存。
        torch.cuda.empty_cache()
    else:  # 加载模式：恢复文件包含优化器等训练状态，普通权重文件不够完整。
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                # step 表示每个 rank 已处理的批次数。总 GPU 数改变时按比例换算，近似保持
                # 已消费的全局样本量；整数除法可能舍去不足一个新批次的部分。
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


def init_model(lm_config, from_weight='pretrain', tokenizer_path='../model', save_dir='../out', device='cuda'):
    """创建 MiniMind 模型和分词器，并可加载某一训练阶段的纯模型权重。

    ``from_weight='none'`` 表示随机初始化，常用于预训练；其他值会拼出
    ``{save_dir}/{from_weight}_{hidden_size}[_moe].pth``。这里的权重文件不同于
    :func:`lm_checkpoint` 的 ``_resume.pth``：它只恢复模型参数，不恢复优化器和步数。
    ``strict=False`` 允许阶段之间存在少量缺失/新增参数，例如 LoRA 或任务头变化；但也
    意味着参数名不匹配不会直接报错，修改模型结构后应额外检查加载结果。
    """
    # AutoTokenizer 会读取 model/ 中的 tokenizer 配置、词表与聊天模板。
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    # 先按配置创建完整结构，再把磁盘权重覆盖到同名参数上。
    model = MiniMindForCausalLM(lm_config)

    if from_weight != 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)

    get_model_params(model, lm_config)
    # requires_grad=False 的冻结参数仍属于总参数，但不会被优化器更新，因此另行报告。
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model.to(device), tokenizer


class SkipBatchSampler(Sampler):
    """把样本采样器组成批次，并在断点续训时丢弃开头若干个已完成批次。

    ``sampler`` 可以是 ``DistributedSampler``，也可以是训练脚本预先打乱的下标列表。
    与逐样本跳过相比，按批跳过能让保存的 ``step`` 直接对应训练循环的步数。最后不足
    ``batch_size`` 的样本仍会作为一个小批次产出，行为相当于 ``drop_last=False``。

    本类只跳过数据，不恢复随机数生成器的瞬时状态；训练脚本通过每个 epoch 重设固定
    种子来重建一致的采样顺序。
    """
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        """依次组装下标批次，先丢弃 ``skip_batches`` 批，再向 DataLoader 产出。"""
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        """返回跳过后剩余的批次数，包含可能存在的最后一个不完整批次。"""
        # 加 batch_size-1 后整除，即整数形式的 ceil(N / batch_size)。
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)


class LMForRewardModel:
    """对话奖励模型的轻量包装器，把候选回答映射为 ``[-3, 3]`` 分数。

    该包装针对支持自定义 ``get_score(tokenizer, messages)`` 接口的远程模型代码（仓库
    默认使用 InternLM 奖励模型），不是任意 ``AutoModel`` 都能直接使用。模型只负责
    推理：初始化后切到 ``eval``，``get_score`` 也由 ``torch.no_grad`` 禁用梯度。
    """
    def __init__(self, model_path, device="cuda", dtype=torch.float16):
        # trust_remote_code=True 允许模型目录提供自定义类和 get_score 实现；因此 model_path
        # 应来自可信来源。torch_dtype 在加载时直接使用目标精度，可减少峰值内存。
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
        self.model = self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def get_score(self, messages, response):
        """将对话历史、最新问题和候选回复整理成奖励模型格式并返回裁剪后的分数。

        ``messages`` 预计是 ``{'role': ..., 'content': ...}`` 字典列表。最后一条被视为当前
        问题，之前各条以 ``角色: 内容`` 拼成历史；``response`` 则作为 assistant 回答。
        裁剪极端分数可避免强化学习时单个异常样本主导优势估计。
        """
        # messages[:-1] 不含最新问题；空列表时 history_text 与 last_query 都安全退化为空。
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages[:-1]])
        last_query = messages[-1]['content'] if messages else ""
        message_context = f"{history_text}\n以上是对话历史。我的新问题是：\n{last_query}" if history_text else last_query
        eval_messages = [
            {"role": "user", "content": message_context},
            {"role": "assistant", "content": response}
        ]
        score = self.model.get_score(self.tokenizer, eval_messages)
        # 先 min 再 max，把任意正常标量限制到闭区间 [-3, 3]。
        return max(min(score, 3.0), -3.0)


# ===== 数学表达式安全求值：替代 eval，只放行算术运算与 math 白名单（长度上限 512） =====
def safe_math_eval(expression):
    """安全计算模型生成的简单数学表达式，而不执行任意 Python 代码。

    支持 ``+ - * / // % **``、一元正负号、``pi/e/tau``，以及 ``sqrt``、``log``、
    三角函数等白名单函数；函数既可写作 ``sqrt(4)``，也可写作 ``math.sqrt(4)``。
    同时把常见的 ``^ × ÷ − ² ³`` 和中文括号规范化为 Python 算术符号。

    实现先用 ``ast.parse`` 把字符串解析成语法树，再由 ``walk`` 逐节点解释，只接受明确
    放行的数字和运算节点。诸如变量访问、下标、列表、lambda、关键字参数等语法都会被
    拒绝，因此比直接 ``eval`` 模型输出安全得多。表达式长度上限为 512，并额外限制幂
    运算规模，减少恶意或错误输入消耗过多 CPU/内存的风险。
    """
    def pow_guard(base, exp):
        """执行带规模检查的幂运算，拦截 ``9**9**9``、``10**99999`` 等输入。"""
        # 用结果十进制位数的近似值 exp*log10(base) 预判巨大结果，避免先算后检查。
        if abs(exp) > 1e4 or (abs(base) > 1 and abs(exp) * math.log10(abs(base)) > 100): raise ValueError('幂运算结果过大')
        return base ** exp

    def resolve(node):
        """从函数/常量节点取白名单名称，兼容 ``sqrt`` 与 ``math.sqrt`` 两种写法。"""
        if isinstance(node, ast.Name): return node.id
        if isinstance(node, ast.Attribute) and getattr(node.value, 'id', '') == 'math': return node.attr

    def walk(node):
        """递归解释一个 AST 节点；遇到未明确支持的节点立即拒绝。"""
        # bool 是 int 的子类，因此必须用 type(...) 精确限制为 int/float，避免接收布尔值。
        if isinstance(node, ast.Constant) and type(node.value) in (int, float): return node.value
        name = resolve(node)
        if name in consts: return consts[name]
        # 用节点类型查表而不是调用节点中的任意名称，限制可执行操作集合。
        if isinstance(node, ast.UnaryOp): return unary_ops[type(node.op)](walk(node.operand))
        if isinstance(node, ast.BinOp): return bin_ops[type(node.op)](walk(node.left), walk(node.right))
        # 不允许关键字参数；函数名不在 funcs 中时会触发 KeyError，并在外层转成统一错误。
        if isinstance(node, ast.Call) and not node.keywords: return funcs[resolve(node.func)](*map(walk, node.args))
        raise ValueError(f'不支持的表达式语法: {type(node).__name__}')

    # 常量、函数、二元运算和一元运算四张白名单共同定义了这门“小型算术语言”。
    consts = {'pi': math.pi, 'e': math.e, 'tau': math.tau}
    funcs = {'pow': pow_guard, **{n: getattr(math, n) for n in 'sqrt exp log log2 log10 sin cos tan asin acos atan atan2 sinh cosh tanh floor ceil trunc fabs fmod hypot gcd degrees radians'.split()}}
    bin_ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: pow_guard}
    unary_ops = {ast.UAdd: operator.pos, ast.USub: operator.neg}
    chars = str.maketrans({'^': '**', '×': '*', '÷': '/', '−': '-', '²': '**2', '³': '**3', '（': '(', '）': ')'})
    # 先转字符串，允许工具调用传入数字；strip 去掉模型输出两侧常见空白。
    expr = str(expression).translate(chars).strip()
    if not expr or len(expr) > 512: raise ValueError('表达式为空或过长')
    try:
        # mode='eval' 只解析单个表达式，赋值、import 等语句在解析阶段就无法通过。
        return walk(ast.parse(expr, mode='eval').body)
    except (KeyError, SyntaxError, TypeError):
        # 对外隐藏 AST/字典查找细节，统一给工具调用方可理解的错误。
        raise ValueError('不支持的表达式语法') from None
