"""MiniMind 各训练阶段共用的数据集与样本编码逻辑。

建议先阅读（有助于理解本文件）：
    1. ``trainer/train_pretrain.py``：查看 ``PretrainDataset`` 返回的
       ``input_ids``、``labels`` 如何送入模型。
    2. ``trainer/train_full_sft.py``：查看 ``SFTDataset`` 与 ``DataLoader``、
       因果语言模型损失之间的连接。
    3. ``model/model_minimind.py`` 中 ``MiniMindForCausalLM.forward``：重点理解
       “当前位置预测下一个 token”的标签错位，以及 ``-100`` 为什么不参与损失。
    4. 当前所用 tokenizer 的 ``tokenizer_config.json``：其中的 ``chat_template``
       决定 system/user/assistant/tool 等消息最终被拼成什么文本和特殊 token。

读完本文件后，推荐继续阅读：
    1. ``trainer/train_dpo.py``：理解 chosen/rejected 偏好对和 ``mask_*`` 如何计算 DPO 损失。
    2. ``trainer/train_ppo.py``、``trainer/train_grpo.py``：理解 ``RLAIFDataset``
       只提供生成起点、答案由策略模型在线采样的强化学习流程。
    3. ``trainer/train_agent.py``：理解 ``AgentRLDataset`` 提供的工具定义、消息历史和
       标准答案如何用于多轮工具调用与奖励计算。
    4. ``trainer/train_lora.py``、``trainer/train_distillation.py``：观察同一
       ``SFTDataset`` 如何复用于 LoRA 微调和知识蒸馏。

本文件的核心职责不是训练模型，而是把 JSON/JSONL 中的一条原始样本转换成训练器
可以直接使用的 token 张量或结构化字段。所有类都遵循 PyTorch ``Dataset`` 协议：
``__len__`` 返回样本数，``__getitem__`` 返回一条经过处理的样本；``DataLoader`` 再将
多条样本自动堆叠为批次。
"""

from torch.utils.data import Dataset
import torch
import json
import os
import random
from datasets import load_dataset, Features, Sequence, Value

# Hugging Face fast tokenizer 底层可能自行创建线程；而训练脚本的 DataLoader 也会创建
# 多个 worker。关闭 tokenizer 内部并行可避免 fork 后的线程告警及不必要的线程竞争。
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def pre_processing_chat(conversations, add_system_ratio=0.2):
    """在套用聊天模板前，对一组对话做轻量的数据增强。

    普通对话若没有 system 消息，会以 ``add_system_ratio`` 的概率随机补一条 system
    提示词，使模型能够适应“有 system”和“无 system”两种输入形式。包含工具定义的
    样本必须原样保留：工具通常挂在 system 消息上，随意插入新 system 消息可能改变
    工具模板的语义或结构。

    注意：此函数可能返回新的列表，但不会就地修改传入的 ``conversations``。
    """
    # 工具调用数据完整保留，不做随机增强。
    if any(conv.get('tools') for conv in conversations):
        return conversations

    # system 提示词中保留中英文两类表达，让模型接触不同语言的系统指令。
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]

    # 仅在首条消息不是 system 时才可能补充，避免连续出现两个 system 消息。
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations


def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    """随机清理聊天模板生成的空思考块。

    参数名表示“保留空思考块的概率”。默认值为 0.2，因此约 80% 的样本会移除
    ``<think>...</think>`` 空块，约 20% 保留。这样可以降低模型机械输出空思考标签的
    倾向，同时仍让模型见过这种合法模板形式。
    """
    # random.random() 落在 [0, 1)，大于 0.2 的概率约为 80%。
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content


class PretrainDataset(Dataset):
    """将纯文本预训练样本编码成固定长度的因果语言模型输入。

    期望每行 JSON 至少包含 ``{"text": "..."}``。返回二元组
    ``(input_ids, labels)``，二者形状均为 ``[max_length]``。labels 与 input_ids
    表面上相同；真正的向右错一位由 ``MiniMindForCausalLM.forward`` 完成。
    """

    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # datasets 会按 JSON/JSONL 格式读取文件；split='train' 表示直接取得唯一的数据切分。
        # 它通常采用 Arrow 按需访问，而不是在这里手工把整个文件解析成 Python 列表。
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        """返回文件中的样本总数，供 DataLoader 计算批次数和采样索引。"""
        return len(self.samples)

    def __getitem__(self, index):
        """读取并编码第 ``index`` 条文本。"""
        sample = self.samples[index]
        # 为首尾的 BOS、EOS 各预留一个位置，所以正文最多只能占 max_length - 2。
        # 此处关闭自动特殊 token，避免 tokenizer 再次插入 BOS/EOS。
        tokens = self.tokenizer(str(sample['text']), add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        # DataLoader 默认要求同一批次内张量形状一致，因此短样本在右侧补 PAD。
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        # PyTorch 交叉熵约定 target=-100 时忽略该位置；PAD 只是对齐占位，不应贡献损失。
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels


class SFTDataset(Dataset):
    """监督微调（SFT）对话数据集，只监督 assistant 回复部分。

    每条样本的 ``conversations`` 是按时间排列的消息列表。聊天模板会把结构化消息转成
    单段文本；``generate_labels`` 随后把 system、user、工具描述等非 assistant 区域
    标为 ``-100``，使损失只训练模型应该生成的 assistant 内容。
    """

    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 显式声明 schema，让 datasets 在某些记录缺少可选字段时仍生成一致的列结构。
        # tools/tool_calls 在 JSONL 中可能本来是对象，但这里统一以字符串保存，使用前再解析。
        features = Features({'conversations': [{'role': Value('string'), 'content': Value('string'), 'reasoning_content': Value('string'), 'tools': Value('string'), 'tool_calls': Value('string')}]})
        self.samples = load_dataset('json', data_files=jsonl_path, split='train', features=features)
        # chat_template 用“BOS + assistant + 换行”标记每段助手回复的开始，用“EOS + 换行”
        # 标记结束。先把这两个边界编码成 token 序列，后面才能在完整 input_ids 中定位。
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        """还原 JSON 字符串字段，并用 tokenizer 的聊天模板渲染整段对话。"""
        messages = []
        tools = None
        for message in conversations:
            # datasets 返回的记录可能不是普通 dict；复制后便于安全地替换字段。
            message = dict(message)
            # 本仓库约定工具定义放在 system 消息的 tools 字段中，并单独传给模板。
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            # 工具调用也要从序列化字符串还原为模板期望的 Python 对象。
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)
        # tokenize=False 先得到文本，是为了能在下一步统一做空思考块处理，然后再编码。
        # add_generation_prompt=False：训练样本已经包含答案，无需追加“等待 assistant 回答”的头部。
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def generate_labels(self, input_ids):
        """生成与 ``input_ids`` 等长的监督标签，仅保留 assistant 区间。

        初始值全为 -100。扫描到 assistant 起始标记后，将其正文一直到结束标记（包含
        EOS）复制到 labels。包含 EOS 能教会模型何时停止回答；起始标记本身不参与损失，
        因为它属于模板提供的上下文。
        """
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            # 切片与边界 token 列表完全相等，说明当前来到一段 assistant 回复之前。
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                # 寻找该回复的 EOS；多轮对话会在外层循环中继续寻找下一段 assistant。
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # 最多写到 max_length，防止被截断的样本越过标签数组边界。
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                # 已处理的回复无需逐 token 重扫；若没找到 EOS，则直接结束扫描。
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]
        # 先做消息级增强，再渲染模板，最后做模板文本级清理。
        conversations = pre_processing_chat(sample['conversations'])
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)
        # tokenizer 默认行为由其配置决定；先截断，再在右侧手工补齐固定长度。
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)
        # 需要检查标签对齐时，可临时启用下面的调试输出：它展示位置 i 的输入 token、
        # 模型在该位置应预测的下一个 token，以及该目标是否被 -100 屏蔽。
        # # === 调试输出 ===
        # print(f"\n--- 样本 {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # # ================
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class DPODataset(Dataset):
    """把偏好对编码成 DPO 训练所需的 chosen/rejected 张量。

    一条数据同时包含 ``chosen``（更优回复的完整对话）和 ``rejected``（较差回复的完整
    对话）。二者分别编码，并只在 assistant 回复位置计算序列对数概率。返回值已经手工
    做好 next-token 错位：``x_* = ids[:-1]``，``y_* = ids[1:]``。
    """

    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 记录 padding id；当前实现实际由 tokenizer 的 padding='max_length' 完成填充，
        # 因而此字段暂未被后续代码读取，保留它是为了兼容早期/后续实现。
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        self.samples = load_dataset('json', data_files=file_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        # 两者都是消息列表，每个元素至少包含 role 与 content；通常共享相同问题、答案不同。
        chosen = sample['chosen']
        rejected = sample['rejected']
        # DPO 比较的是完整回复的条件概率，因此保留现有答案，不追加生成提示头。
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        chosen_prompt = post_processing_chat(chosen_prompt)

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = post_processing_chat(rejected_prompt)
        # DPO 的一个 batch 需要张量等长；这里直接让 tokenizer 截断并补齐到 max_length。
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)
        # 模型位置 t 的 logits 预测 token[t+1]，故输入去掉最后一个 token、目标去掉第一个。
        # mask 同样去掉第一个元素，才能与 y 和每个目标 token 的 log-prob 对齐。
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def generate_loss_mask(self, input_ids):
        """生成 0/1 掩码：assistant 正文与其 EOS 为 1，其余上下文为 0。

        逻辑与 ``SFTDataset.generate_labels`` 相同，区别是 DPO 训练器需要显式掩码来对
        每个 token 的 log-prob 求和，而不是把无关标签写成 -100 后交给交叉熵。
        """
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


class RLAIFDataset(Dataset):
    """为 PPO/GRPO 在线采样提供“尚未回答的问题提示词”。

    强化学习阶段不会直接学习数据中的最后一条标准回复，而是取 ``conversations[:-1]``
    作为上下文，让当前策略模型自行生成答案，再由奖励函数/奖励模型评分。因此返回的
    ``answer`` 目前只是训练器接口所需的空占位符。
    """

    def __init__(self, jsonl_path, tokenizer, max_length=1024, thinking_ratio=0.5):
        super().__init__()
        self.tokenizer = tokenizer
        # 当前类只返回字符串 prompt，截断发生在后续 rollout 流程；保留该参数以统一接口。
        self.max_length = max_length
        # 每次取样时，以该概率让聊天模板打开思考模式，用于混合训练思考/非思考回答。
        self.thinking_ratio = thinking_ratio
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        # 这两个边界字段目前未被本类后续方法使用，保留是为了与其他对话数据集接口一致。
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        """移除样本中的最后一条参考回复，并渲染等待模型续写的提示词。"""
        conversations = pre_processing_chat(conversations)
        use_thinking = random.random() < self.thinking_ratio
        return self.tokenizer.apply_chat_template(
            # 数据约定最后一条是 assistant 参考答案；RL rollout 不应提前看到它。
            conversations[:-1],
            tokenize=False,
            open_thinking=use_thinking,
            # 追加 assistant 起始标记，明确告诉生成模型接下来轮到它作答。
            add_generation_prompt=True
        )

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = self.create_chat_prompt(sample['conversations'])

        return {
            'prompt': prompt,
            # PPO/GRPO 的答案来自在线 rollout，并非离线数据集提供。
            'answer': ""
        }


class AgentRLDataset(Dataset):
    """为 Agent 强化学习加载消息、工具定义和奖励判定所需的标准答案。

    与前几个类不同，这里直接逐行读取 JSONL。每条样本最终返回：

    - ``messages``：去掉最后一条参考回复后的对话历史；
    - ``tools``：从 system 消息中取出的工具 schema；
    - ``gt``：ground truth（标准答案/关键结果），供奖励函数判断生成结果是否正确。
    """

    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        # tokenizer/max_length 当前未在数据集内编码使用；实际编码发生在 train_agent.py 的
        # 多轮 rollout 中。仍保存它们以与其他 Dataset 构造方式保持一致并便于未来扩展。
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                # JSONL 的每一行都是一条独立、完整的 JSON 记录。
                self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def parse_conversations(self, conversations):
        """拆出工具定义，并返回不含末尾参考回复的消息历史。"""
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            # 工具 schema 可能以 JSON 字符串存储，也可能已经是 Python 列表/字典。
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            messages.append(message)
        # 最后一条通常是示范 assistant 答案；在线 rollout 必须让模型自己产生它。
        return messages[:-1], tools

    def __getitem__(self, index):
        sample = self.samples[index]
        messages, tools = self.parse_conversations(sample['conversations'])
        # gt 不作为模型输入，只在 trainer/train_agent.py 中参与奖励计算。
        return {'messages': messages, 'tools': tools, 'gt': sample['gt']}


if __name__ == "__main__":
    # 本文件通常作为模块被训练脚本导入；直接运行时不执行任何操作。
    pass
