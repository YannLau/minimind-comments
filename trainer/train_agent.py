"""MiniMind 的多轮工具调用强化学习训练入口。

建议先阅读以下代码，能帮助理解本文件中的对象从何而来、彼此如何衔接：

1. ``dataset/lm_dataset.py`` 中的 ``AgentRLDataset``：了解一条样本如何拆成
   ``messages``（模型可见的历史）、``tools``（可用工具 schema）和 ``gt``（只用于评分的答案）。
2. ``minimind-3/chat_template.jinja`` 或 ``model/tokenizer_config.json`` 中的
   ``chat_template``：了解结构化消息如何变成模型看到的文本，以及工具调用、工具结果、
   ``<think>`` 和生成提示头的格式。
3. ``trainer/rollout_engine.py``：了解本文件如何通过 Torch 或 SGLang 生成回答，以及
   ``RolloutResult`` 中 token、逐 token 对数概率和掩码的含义。
4. ``model/model_minimind.py`` 中的 ``MiniMindForCausalLM.forward``：理解因果语言模型
   的 logits、MoE 辅助损失，以及当前位置预测下一个 token 的错位关系。
5. ``trainer/trainer_utils.py`` 中的 ``init_model``、``lm_checkpoint``、
   ``SkipBatchSampler``、``LMForRewardModel`` 和 ``safe_math_eval``：了解模型加载、断点、
   可选奖励模型和模拟计算器的实现。

读完后推荐继续阅读：

1. ``README.md`` 的“Agentic RL”小节：了解数据格式、整条轨迹奖励和训推分离的总体设计。
2. ``trainer/train_grpo.py``：对照理解组内相对优势；本文件同样按同一个 prompt 的多条
   rollout 计算优势，但加入了多轮工具交互和工具结果观察。
3. ``trainer/train_ppo.py``：对照 PPO、参考策略 KL 约束、混合精度、DDP 与断点续训流程。
4. ``trainer/rollout_engine.py`` 的 ``SGLangRolloutEngine``：如果要用远端 SGLang 采样，
   进一步了解训练侧如何同步新策略权重。

整体数据流是：数据集给出对话起点和工具 → 策略模型采样多轮回答/工具调用 → 本脚本模拟
执行工具并把结果作为观察拼回上下文 → 对完整轨迹计算奖励 → 用 GRPO 或 CISPO 风格的
策略损失更新模型。这里的工具实现是本地模拟数据，不会请求真实天气、时间或汇率服务。
"""

import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 只需导入即可规避 Windows 上 pyarrow/torch 的 DLL 冲突（仓库问题 #771）；本文件不直接用它。
import datasets  # noqa: F401
import re
import gc
import json
import math
import random
import argparse
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import AgentRLDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model, LMForRewardModel, safe_math_eval
from trainer.rollout_engine import create_rollout_engine, compute_per_token_logps

warnings.filterwarnings('ignore')

# ================================ 工具与奖励模块开始 ================================

def rep_penalty(text, n=3, cap=0.5):
    """计算重复 n-gram 惩罚，结果在 ``[0, cap]`` 范围内。

    先将小写文本切成单词和标点，再统计重复出现的连续 n 个 token 片段。短文本若不足
    n 个 token，则没有可比较片段，返回 0。它是奖励函数中的轻量启发式项，不是语言模型分数。
    """
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0

# ======== 工具定义：schema 会交给 chat_template，指导模型按约定生成调用 ========
# 结构采用 OpenAI 风格的 function schema。这里声明的是模型可见的工具“说明书”；
# 真正执行逻辑由下方 MOCK_RESULTS 提供。required 为空表示工具参数可省略。
TOOLS = [
    {"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "unit_converter", "description": "单位换算", "parameters": {"type": "object", "properties": {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}}, "required": ["value", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {"name": "get_current_weather", "description": "获取天气", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}},
    {"type": "function", "function": {"name": "get_current_time", "description": "获取时间", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "default": "Asia/Shanghai"}}, "required": []}}},
    {"type": "function", "function": {"name": "get_exchange_rate", "description": "查询汇率", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string"}, "to_currency": {"type": "string"}}, "required": ["from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "translate_text", "description": "翻译文本", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "target_language": {"type": "string"}}, "required": ["text", "target_language"]}}},
]

# ======== 模拟数据：训练演示用固定字典，不会发起真实外部 API 请求 ========
# 查询不到的城市、时区、货币对或翻译组合会在下面的模拟函数中使用默认值。
WEATHER_DATA = {"北京": ("28°C", "晴"), "上海": ("15°C", "多云"), "广州": ("32°C", "闷热"), "深圳": ("30°C", "晴"), "杭州": ("22°C", "阴"), "成都": ("18°C", "小雨"), "武汉": ("25°C", "多云"), "南京": ("20°C", "晴"), "西安": ("16°C", "大风"), "重庆": ("26°C", "阴"), "Tokyo": ("12°C", "晴"), "New York": ("8°C", "多云"), "London": ("5°C", "小雨"), "Paris": ("10°C", "阴"), "Sydney": ("25°C", "晴朗")}
TIME_DATA = {"Asia/Shanghai": "2025-03-07 14:30:00", "America/New_York": "2025-03-07 01:30:00", "Europe/London": "2025-03-07 06:30:00", "Asia/Tokyo": "2025-03-07 15:30:00", "Europe/Paris": "2025-03-07 07:30:00", "Australia/Sydney": "2025-03-07 17:30:00"}
EXCHANGE_DATA = {("USD", "CNY"): 7.21, ("EUR", "CNY"): 7.85, ("GBP", "CNY"): 9.12, ("JPY", "CNY"): 0.048, ("USD", "EUR"): 0.92, ("USD", "GBP"): 0.79, ("CNY", "JPY"): 20.83, ("AUD", "CNY"): 4.72}
TRANSLATE_DATA = {("你好世界", "english"): "Hello World", ("Good morning", "chinese"): "早上好", ("今天天气真好", "english"): "The weather is nice today", ("I love programming", "chinese"): "我喜欢编程", ("机器学习很有趣", "english"): "Machine learning is interesting", ("Happy birthday", "chinese"): "生日快乐"}
UNIT_DATA = {"km_miles": 0.621371, "miles_km": 1.60934, "kg_pounds": 2.20462, "pounds_kg": 0.453592, "meters_feet": 3.28084, "feet_meters": 0.3048, "celsius_fahrenheit": 1.8, "fahrenheit_celsius": 0.5556}

# ======== 模拟执行：工具名映射到纯 Python 函数，统一返回可 JSON 序列化的结果 ========
# calculate_math 使用 trainer_utils.safe_math_eval，避免对模型生成内容直接调用 eval。
# 单位换算只支持 UNIT_DATA 中列出的单位组合；未列出的组合按倍率 1.0 处理。
MOCK_RESULTS = {
    "calculate_math": lambda args: {"result": str(safe_math_eval(args.get("expression", "0")))},
    "unit_converter": lambda args: {"result": round(float(args.get("value", 0)) * UNIT_DATA.get(f"{args.get('from_unit', '').lower()}_{args.get('to_unit', '').lower()}", 1), 4)},
    "get_current_weather": lambda args: (lambda w: {"city": args.get("location"), "temperature": w[0], "humidity": "65%", "condition": w[1]})(WEATHER_DATA.get(args.get("location"), ("22°C", "晴"))),
    "get_current_time": lambda args: {"datetime": TIME_DATA.get(args.get("timezone", "Asia/Shanghai"), "2025-03-07 14:30:00"), "timezone": args.get("timezone", "Asia/Shanghai")},
    "get_exchange_rate": lambda args: {"from": args.get("from_currency"), "to": args.get("to_currency"), "rate": EXCHANGE_DATA.get((args.get("from_currency"), args.get("to_currency")), 1.0)},
    "translate_text": lambda args: {"translated_text": TRANSLATE_DATA.get((args.get("text"), args.get("target_language")), args.get("text", ""))},
}

# ======== 参数校验：奖励计算用它判断调用名与必填参数是否有效 ========
# 这只是本训练脚本的基本格式检查，并不等同于完整 JSON Schema 校验。
CHECK_ARGS = {
    "calculate_math": lambda a: bool(a.get("expression")),
    "unit_converter": lambda a: a.get("value") is not None and a.get("from_unit") and a.get("to_unit"),
    "get_current_weather": lambda a: bool(a.get("location")),
    "get_current_time": lambda a: True,
    "get_exchange_rate": lambda a: bool(a.get("from_currency")) and bool(a.get("to_currency")),
    "translate_text": lambda a: bool(a.get("text")) and bool(a.get("target_language")),
}

# ======== 工具调用解析与执行 ========
def parse_tool_calls(text):
    """从一段模型输出中提取所有 ``<tool_call>...</tool_call>`` JSON 对象。

    使用非贪婪匹配，因此一段回复中可以有多个调用。格式错误的片段会被跳过；调用方随后
    还会根据实际提供的工具列表和参数检查结果判断有效性。
    """
    calls = []
    for m in re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL):
        try: calls.append(json.loads(m.strip()))
        except: pass
    return calls

def execute_tool(name, args):
    """按名称执行本地模拟工具；工具不存在或执行异常时返回 ``None``。"""
    fn = MOCK_RESULTS.get(name)
    if not fn: return None
    try:
        return fn(args)
    except Exception:
        return None

# ======== 多轮 Rollout ========
def rollout_single(rollout_engine, tokenizer, messages, tools, max_turns=3, max_new_tokens=256, thinking_ratio=0.5, device="cuda"):
    """为一个 prompt 采样一条可能包含工具交互的轨迹。

    ``messages`` 会在本函数中追加 assistant 与 tool 消息；调用前应传入副本。第一轮的
    prompt token 固定保存在 ``prompt_ids``，之后每轮的模型输入由它加上累计的
    ``response_ids`` 组成。这里把两类生成 token 区分开：策略自己生成的 token 掩码为 1，
    模板补入的工具观察/下一轮提示 token 掩码为 0，因此环境提供的文字不会被当成策略动作
    来训练。返回值还包含各轮原始回答，供延迟奖励解析。
    """
    all_outputs = []
    prompt_ids = None
    response_ids = []
    response_mask = []
    response_old_logps = []
    final_context = ""
    unfinished = False
    # 有一定概率让提示以未闭合的 <think> 开始，训练模型在“需要推理”的输入格式下继续生成。
    open_thinking = random.random() < thinking_ratio
    for turn in range(max_turns):
        # add_generation_prompt 在每轮末尾放置 assistant 起始头；工具定义也由模板注入。
        context = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools, open_thinking=open_thinking)
        if prompt_ids is None:
            # 只在第一轮编码输入前缀。后续 assistant 输出和工具观察由 response_ids 追加，
            # 避免重复编码整段历史，也保持逐 token logprob 与最终训练序列对齐。
            prompt_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
        input_ids = torch.tensor([prompt_ids + response_ids], device=device)
        # 每次 rollout 只生成一条；分批多样本/多生成数由 rollout_batch 的循环完成。
        rollout_result = rollout_engine.rollout(
            prompt_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            num_generations=1,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
        )
        valid_len = int(rollout_result.completion_mask[0].sum().item())
        # completion_mask 排除引擎为批次对齐补的 PAD；仅取真实生成部分及其 logprob。
        new_ids = rollout_result.completion_ids[0, :valid_len].tolist()
        new_logps = rollout_result.per_token_logps[0, :valid_len].tolist()
        if len(new_ids) != len(new_logps):
            raise RuntimeError(f"rollout token/logprob length mismatch: {len(new_ids)} vs {len(new_logps)}")
        new_text = rollout_result.completions[0]
        all_outputs.append(new_text)
        response_ids.extend(new_ids)
        # EOS 是停止符，不作为要优化的回答内容；环境观察部分稍后统一用 0 掩码。
        response_mask.extend([int(t != tokenizer.eos_token_id) for t in new_ids])
        response_old_logps.extend(new_logps)
        final_context = context + new_text
        calls = parse_tool_calls(new_text)
        if not calls:
            # 没有工具调用意味着当前 assistant 输出就是终止回答，无需再进入下一轮。
            break
        # 如果最后一个允许的回合仍在请求工具，轨迹没有机会再根据结果生成最终回答。
        unfinished = turn == max_turns - 1
        assistant_message = {"role": "assistant", "content": new_text}
        messages.append(assistant_message)
        for call in calls:
            # 模型可能把 arguments 生成为 JSON 字符串；统一还原为字典再校验/执行。
            name, raw = call.get("name", ""), call.get("arguments", {})
            if isinstance(raw, str):
                try: raw = json.loads(raw)
                except: raw = {}
            result = execute_tool(name, raw)
            # 给工具结果设上限，避免异常长的模拟结果意外拉长序列。
            result_str = (json.dumps(result, ensure_ascii=False) if result else '{"error": "tool not found"}')[:2048]  # 防止天文数字撑爆tokenizer
            messages.append({"role": "tool", "content": result_str})

        # 需要把新产生的 tool 消息按 chat_template 编码，但不能再次把本轮 assistant 输出
        # 加入 response_ids。临时在其末尾放唯一标记，从完整渲染文本中切出标记之后的增量。
        marker = f"<|agent_observation_{id(messages)}_{len(response_ids)}|>"
        assistant_message["content"] += marker
        marked_context = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=not unfinished, tools=tools, open_thinking=open_thinking)
        assistant_message["content"] = new_text
        _, found, observation = marked_context.partition(marker)
        if not found:
            raise RuntimeError("chat template did not preserve the assistant content boundary")
        # 模板增量可能以 EOS 开始，而 EOS 已经在模型生成 token 中；丢掉重复的那个边界 token。
        obs_delta = tokenizer(observation, add_special_tokens=False)["input_ids"]
        if new_ids and new_ids[-1] == tokenizer.eos_token_id and obs_delta[:1] == [tokenizer.eos_token_id]:
            obs_delta = obs_delta[1:]
        observe_context = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=not unfinished, tools=tools, open_thinking=open_thinking)
        response_ids.extend(obs_delta)
        # 工具观察来自环境，不是策略采样动作：保留在上下文里，但从策略目标和 KL 项中屏蔽。
        response_mask.extend([0] * len(obs_delta))
        response_old_logps.extend([0.0] * len(obs_delta))
        final_context = observe_context

    final_output = all_outputs[-1] if all_outputs else ""
    prompt_ids = prompt_ids or []
    # final_output 保持为最后一轮模型原始输出；完整轨迹还需结合 turn_outputs 与上下文理解。
    return final_output, final_context, prompt_ids, response_ids, response_mask, response_old_logps, list(all_outputs), unfinished

def rollout_batch(rollout_engine, tokenizer, messages_batch, tools_batch, num_gen, max_turns=3, max_new_tokens=256, thinking_ratio=0.5, device="cuda"):
    """对批次中的每条输入各采样 ``num_gen`` 条轨迹，并按输入顺序展平结果。

    展平顺序是先 prompt、后该 prompt 的第 0...``num_gen-1`` 个生成，因此后续奖励函数
    可通过 ``sample_idx = idx // num_gen`` 找回每条轨迹所属的原始样本。
    """
    all_completions = []
    all_contexts = []
    all_prompt_ids = []
    all_response_ids = []
    all_response_masks = []
    all_response_old_logps = []
    all_turn_outputs = []
    all_unfinished = []
    for messages, tools in zip(messages_batch, tools_batch):
        for _ in range(num_gen):
            # 每条采样轨迹必须拥有独立历史，避免某次工具调用污染同 prompt 的其他生成。
            msgs_copy = [dict(m) for m in messages]
            completion, context, prompt_ids, response_ids, response_mask, response_old_logps, turn_outputs, unfinished = rollout_single(rollout_engine, tokenizer, msgs_copy, tools, max_turns, max_new_tokens, thinking_ratio, device)
            all_completions.append(completion)
            all_contexts.append(context)
            all_prompt_ids.append(prompt_ids)
            all_response_ids.append(response_ids)
            all_response_masks.append(response_mask)
            all_response_old_logps.append(response_old_logps)
            all_turn_outputs.append(turn_outputs)
            all_unfinished.append(unfinished)
    return all_completions, all_contexts, all_prompt_ids, all_response_ids, all_response_masks, all_response_old_logps, all_turn_outputs, all_unfinished

# ======== 奖励计算 ========
def validate_gt_in_text(text, gt_list):
    """找出文本中命中的 ground-truth 项。

    普通答案用不区分大小写的子串匹配；纯数字答案会额外去掉千位逗号，并与正文中解析到的
    数字按 ``1e-6`` 容差比较，以兼容 ``1,000`` 与 ``1000.0`` 这类格式差异。
    """
    text, text_num = str(text), str(text).replace(',', '')
    nums = [float(x) for x in re.findall(r'(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w.])', text_num)]
    return {g for g in gt_list if ((s := str(g).strip()) and s.lower() in text.lower()) or (re.fullmatch(r'[-+]?\d+(?:\.\d+)?', str(g).strip().replace(',', '')) and any(abs(float(str(g).strip().replace(',', '')) - n) < 1e-6 for n in nums))}

def calculate_rewards(prompts, completions, gt_batch, tools_batch, num_gen, reward_model=None, device="cuda", turn_outputs_batch=None, unfinished_batch=None):
    """为展开后的轨迹列表计算奖励，每条轨迹得到一个裁剪到 ``[-3, 3]`` 的标量。

    不调用工具的回答主要按长度、思考格式、可选 Reward Model 和重复度评分；包含工具调用
    的轨迹则按调用合法性、GT 命中、是否在回合上限处未完成和重复度评分。奖励在轨迹结束
    后才产生，后面会广播到该轨迹的有效回答 token 上。``prompts`` 是模板渲染后的字符串，
    仅在需要调用奖励模型时还原成 role/content 消息。
    """
    rewards = torch.zeros(len(completions), device=device)
    for idx, response in enumerate(completions):
        reward, answer = 0.0, response
        sample_idx = idx // num_gen
        tools = tools_batch[sample_idx]
        turn_outputs = turn_outputs_batch[idx] if turn_outputs_batch is not None else [response]
        unfinished = unfinished_batch[idx] if unfinished_batch is not None else False
        turn_answers = [turn.split('</think>', 1)[-1].strip() if '</think>' in turn else turn.strip() for turn in turn_outputs]
        # 最后一轮输出用于最终答案；工具调用计数则覆盖整条轨迹的所有轮次。
        answer = turn_answers[-1] if turn_answers else response.strip()
        valid_names = {t['function']['name'] for t in tools} if tools else set()
        tool_calls = []
        for turn_answer in turn_answers: tool_calls.extend(parse_tool_calls(turn_answer))  # 解析工具调用
        reward -= 0.5 * sum(abs(turn.count('<tool_call>') - turn.count('</tool_call>')) for turn in turn_answers)  # 工具调用标签未闭合时扣分
        # -------- 无工具调用：格式与奖励模型评分 --------
        if not tool_calls:
            # 普通回答奖励：先限制极短/极长输出，再单独检查思考块格式与最终答案质量。
            reward += 0.5 if 5 <= len(response.strip()) <= 800 else -0.5  # 长度分
            if '</think>' in response:
                think, answer = response.split('</think>', 1)
                reward += 1.0 if 20 <= len(think.strip()) <= 300 else -0.5  # 思考内容长度分
                reward += 0.25 if response.count('</think>') == 1 else -0.25  # 思考结束标签格式分
                answer = answer.strip()
            if reward_model is not None:
                # 奖励模型只评估最终答案，不把隐藏的 think 段当作面向用户的答复。
                prompt = prompts[sample_idx]
                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]
                score = reward_model.get_score(messages, answer)
                reward += score  # 奖励模型分数
            reward -= rep_penalty(answer)
            rewards[idx] = max(min(reward, 3.0), -3.0)  # 将总奖励裁剪到 [-3, 3]
        # -------- 有工具调用：按工具调用与结果校验奖励 --------
        else:
            # 工具轨迹奖励：只认可本样本声明过的工具，并检查对应参数是否完整。
            gt = gt_batch[sample_idx]
            valid_call_count = 0
            for tool_call in tool_calls:
                name, raw = tool_call.get("name", ""), tool_call.get("arguments", {})
                if isinstance(raw, str):
                    try: raw = json.loads(raw)
                    except: raw = {}
                check = CHECK_ARGS.get(name)
                valid_call_count += int(bool(name in valid_names and check and check(raw)))
            tool_gap = abs(valid_call_count - len(gt)) + max(0, len(tool_calls) - valid_call_count)  # 工具调用数量与有效性差值
            reward += 0.5 if tool_gap == 0 else -0.5 * tool_gap  # 工具调用对齐分
            
            # 未完成轨迹不能把最后一条工具调用内容误当最终答案；已完成时去掉尾随的工具调用片段。
            final_text = "" if unfinished else (answer.split('</tool_call>')[-1] if '</tool_call>' in answer else answer)
            verified = validate_gt_in_text(final_text, gt) if gt else set()
            if gt: reward += 2.5 * len(verified) / len(gt)  # 标准答案命中分
            if unfinished: reward -= 0.5  # 轨迹未完成扣分
            reward -= rep_penalty(final_text if final_text else answer)
            rewards[idx] = max(min(reward, 3.0), -3.0)  # 将总奖励裁剪到 [-3, 3]
    return rewards

# ================================ 工具与奖励模块结束 ================================

def rl_train_epoch(epoch, loader, iters, rollout_engine, ref_model, reward_model=None, start_step=0, wandb=None, use_sglang=False):
    """完成一个 epoch 的在线采样、延迟奖励计算和策略参数更新。

    ``loader`` 每次给出一批原始 prompt；每条 prompt 会先采样 ``num_generations`` 条完整
    轨迹，再将轨迹补齐为张量。奖励按 prompt 分组标准化成 advantage，随后在有效回答 token
    上计算策略损失和参考模型 KL 惩罚。``iters`` 是循环中使用的 epoch 步数上界，恢复训练
    时会包含已跳过的步数，以便日志、保存间隔和最后一步判断仍使用原始 step 编号。

    ``use_sglang`` 当前仅由调用端传入、函数体没有读取；实际 rollout 后端由
    ``rollout_engine`` 对象决定。
    """
    for step, batch in enumerate(loader, start=start_step + 1):
        # AgentRLDataset 的 collate_fn 保留变长的结构化消息，不在 DataLoader 阶段做 token 化。
        messages_batch = batch['messages']
        tools_batch = batch['tools']
        gt_batch = batch['gt']

        with torch.no_grad():
            # rollout 不参与反向传播；引擎同时返回采样时策略的逐 token logprob，供重要性比率使用。
            completions, contexts, prompt_ids_batch, response_ids_batch, response_masks_batch, response_old_logps_batch, turn_outputs_batch, unfinished_batch = rollout_batch(rollout_engine, tokenizer, messages_batch, tools_batch, args.num_generations, max_turns=3, max_new_tokens=args.max_gen_len, thinking_ratio=args.thinking_ratio, device=args.device)

        # 奖励模型接收模板化 prompt。它只在 calculate_rewards 的无工具回答分支中使用。
        prompts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True, tools=t) for m, t in zip(messages_batch, tools_batch)]
        packed_samples = []
        for p, r, m, old_lp in zip(prompt_ids_batch, response_ids_batch, response_masks_batch, response_old_logps_batch):
            # 序列 = prompt + 生成/观察部分；prompt 与环境观察均不作为策略动作计算损失。
            ids = p + r
            mask = [0] * len(p) + m
            # 因果 LM 的第一个预测目标是 ids[1]，所以对齐到长度 len(ids)-1 的 logprob 序列时，
            # prompt 的前 len(p)-1 个预测位置填 0，再接 rollout 对回答/观察 token 保存的旧 logprob。
            old_logps = [0.0] * max(len(p) - 1, 0) + old_lp
            if len(ids) > args.max_total_len:
                # 超长时从左边截掉最早上下文，保留最新交互；掩码和旧 logprob 做同样截断。
                ids = ids[-args.max_total_len:]
                mask = mask[-args.max_total_len:]
                old_logps = old_logps[-(len(ids) - 1):]
            # 第一处 mask=1 是策略开始生成的位置。若没有任何策略 token，则视作整条序列均为 prompt/观察。
            prompt_len = next((i for i, v in enumerate(mask) if v == 1), len(mask))
            packed_samples.append((ids, mask, prompt_len, old_logps))
        # 将可变长轨迹 pad 成批张量。seq_lens 保留真实长度，避免 PAD 进入注意力和调试输出。
        seq_lens = torch.tensor([len(ids) for ids, _, _, _ in packed_samples], device=args.device)
        max_len = seq_lens.max().item()
        input_ids = torch.tensor([ids + [tokenizer.pad_token_id] * (max_len - len(ids)) for ids, _, _, _ in packed_samples], device=args.device)
        prompt_lens = torch.tensor([prompt_len for _, _, prompt_len, _ in packed_samples], device=args.device)
        # full_response_masks 与 input_ids 等长；之后切掉位置 0，和“预测下一个 token”的 logits 对齐。
        full_response_masks = torch.tensor([mask + [0] * (max_len - len(mask)) for _, mask, _, _ in packed_samples], device=args.device, dtype=torch.float32)
        # 逐 token 旧 logprob 长度为真实序列长度-1；批内补零到 max_len-1。
        old_per_token_logps = torch.tensor([old_logps + [0.0] * ((max_len - 1) - len(old_logps)) for _, _, _, old_logps in packed_samples], device=args.device, dtype=torch.float32)
        # 注意力掩码中 prompt、模型回答和工具观察都有效；只有补齐 PAD 的位置为 0。
        full_mask = (torch.arange(max_len, device=args.device).unsqueeze(0) < seq_lens.unsqueeze(1)).long()

        # 奖励是每条轨迹一个标量，后续由该轨迹内的组标准化生成 advantage。
        rewards = calculate_rewards(prompts, completions, gt_batch, tools_batch, args.num_generations, reward_model, device=args.device, turn_outputs_batch=turn_outputs_batch, unfinished_batch=unfinished_batch)

        with autocast_ctx:
            # 当前策略对完整序列前向；CausalLM 的位置 t logits 对应 input_ids[t+1]。
            res = model(input_ids, attention_mask=full_mask)
            # MoE 路由均衡辅助损失直接相加；稠密模型没有该项。
            aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
            logits = res.logits[:, :-1, :]
            # 取出每个实际下一个 token 在当前策略分布下的 log p(a_t | prefix_t)，形状 [样本数, 长度-1]。
            per_token_logps = F.log_softmax(logits, dim=-1).gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            # ref_model 是初始化时复制的冻结策略，提供 KL 正则的参照分布。
            ref_per_token_logps = compute_per_token_logps(ref_model, input_ids, input_ids.size(1) - 1, attention_mask=full_mask)

        # 目标 token 的掩码要从序列位置 1 开始，才能和 logprob 的 next-token 维对齐。
        completion_mask = full_response_masks[:, 1:]
        # 找到每条回复中第一个被标记为有效动作的 EOS，用于去掉其后的尾部 token。
        is_eos = (input_ids[:, 1:] == tokenizer.eos_token_id) & completion_mask.bool()
        eos_idx = torch.full((completion_mask.size(0),), completion_mask.size(1) - 1, device=args.device, dtype=torch.long)
        has_eos = is_eos.any(dim=1)
        eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
        pos = torch.arange(completion_mask.size(1), device=args.device).unsqueeze(0)
        completion_mask = completion_mask * (pos <= eos_idx.unsqueeze(1)).float()
        token_counts = completion_mask.sum(dim=1)
        # 空动作序列无法按 token 平均损失，后面会从批次均值中排除。
        valid_rows = token_counts > 0

        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            # 展示模板上下文、完整采样轨迹和奖励，便于检查工具模板、token 边界及 GT 奖励。
            for i in range(len(messages_batch)):
                Logger(f"[DEBUG] step={step}, gt[{i}]: {repr(gt_batch[i])}")
                Logger('-'*100)
                for j in range(args.num_generations):
                    idx = i * args.num_generations + j
                    plen, slen = prompt_lens[idx].item(), seq_lens[idx].item()
                    Logger(f"{'=' * 30} [DEBUG] gen[{i}][{j}] CONTEXT_BEGIN {'=' * 30}")
                    Logger(contexts[idx])
                    Logger(f"{'=' * 31} [DEBUG] gen[{i}][{j}] CONTEXT_END {'=' * 31}")
                    Logger(f"[DEBUG] gen[{i}][{j}] prompt_len={plen}, seq_len={slen}")
                    tokens = input_ids[idx, plen:slen].tolist()
                    text = tokenizer.decode(tokens, skip_special_tokens=False)
                    Logger(f"{'=' * 28} [DEBUG] gen[{i}][{j}] COMPLETION_BEGIN [{plen}:{slen}] {'=' * 28}")
                    Logger(text)
                    Logger(f"{'=' * 29} [DEBUG] gen[{i}][{j}] COMPLETION_END {'=' * 29}")
                    Logger(f"[DEBUG] gen[{i}][{j}] reward={rewards[idx].item():.4f}")
                    Logger('='*100)

        # 同一原始 prompt 的生成结果成组比较。组内均值作为 baseline，标准差归一化奖励尺度；
        # 标准差为 0 时加上小常数以避免除零，此时该组优势为 0。
        grouped_rewards = rewards.view(-1, args.num_generations)
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)
        std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)
        advantages = (rewards - mean_r) / (std_r + 1e-4)

        # 逐 token KL 估计：令 d=log p_ref-log p_policy，则 exp(d)-d-1 在分布期望下非负。
        # 只在 completion_mask 覆盖的策略生成 token 上计入最终目标。
        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1
        # 新旧策略对采样动作的概率比率；新策略前向可求梯度，old_logps 来自 rollout 时的旧策略。
        ratio = torch.exp(per_token_logps - old_per_token_logps)
        if args.loss_type == "cispo":
            # CISPO 截断重要性权重并 detach 它，策略梯度仍通过 log p_theta 保留；
            # 这与 GRPO 对 ratio 做 PPO 式双边裁剪的目标不同。
            clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
            per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl)
        else:
            # GRPO/PPO clipped surrogate：取未裁剪和裁剪目标中的较小值，限制策略单步变化。
            clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
            per_token_loss1 = ratio * advantages.unsqueeze(1)
            per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
            per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)
        # 先对每条轨迹的有效 token 求平均，再对有有效 token 的轨迹求均值，避免长回答天然权重大。
        policy_loss = (((per_token_loss * completion_mask).sum(dim=1)[valid_rows] / token_counts[valid_rows].clamp(min=1)).mean()
                       if valid_rows.any() else per_token_loss.sum() * 0.0)
        # 除以累积步数后逐批反传；达到累积边界或 epoch 最后一步时再更新参数。
        loss = (policy_loss + aux_loss) / args.accumulation_steps
        loss.backward()

        if step % args.accumulation_steps == 0 or step == iters:
            # 梯度裁剪限制全局梯度范数；随后更新策略、余弦学习率并清空累计梯度。
            if args.grad_clip > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()

        if step % args.log_interval == 0 or step == iters:
            # 监控奖励、参考 KL、组内奖励差异（学习信号）、优势、损失、长度与学习率。
            pl = loss.item() * args.accumulation_steps
            ar = rewards.mean().item()
            al = token_counts.float().mean().item()
            kl = ((ref_per_token_logps - per_token_logps) * completion_mask).sum().item() / max(token_counts.sum().item(), 1)
            gs = grouped_rewards.std(dim=1, unbiased=False).mean().item()
            am, ast = advantages.mean().item(), advantages.std().item()
            lr = optimizer.param_groups[0]['lr']
            Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}), Reward:{ar:.4f}, KL:{kl:.4f}, GrpStd:{gs:.4f}, AdvStd:{ast:.4f}, Loss:{pl:.4f}, AvgLen:{al:.2f}, AdvMean:{am:.4f}, LR:{lr:.8f}')
            if wandb and is_main_process():
                wandb.log({"reward":ar,"kl_ref":kl,"group_reward_std":gs,"advantages_std":ast,"policy_loss":pl,"avg_response_len":al,"advantages_mean":am,"learning_rate":lr})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            # 保存可直接加载的半精度权重，以及包含优化器/调度器/步数的续训 checkpoint。
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scheduler=scheduler)
            model.train()
            del state_dict

        # 同步采样引擎到刚更新的策略；Torch 引擎替换模型引用，SGLang 会保存并通知服务端加载权重。
        if step % args.save_interval == 0 or step == iters: rollout_engine.update_policy(model)

        # 及时释放大张量引用，降低长序列在线 RL 的显存峰值。
        del per_token_logps, ref_per_token_logps
        del completions, rewards, grouped_rewards, mean_r, std_r, advantages, completion_mask


if __name__ == "__main__":
    # 该脚本依赖相对项目路径读取数据、tokenizer 与权重；通常从 trainer 目录启动，
    # 也可以通过项目约定的启动命令运行。参数可覆盖模型规模、采样方式和训练行为。
    parser = argparse.ArgumentParser(description="MiniMind Agent RL")
    # 输出、轮数和每步数据量。
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='agent', type=str, help="保存权重名称")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="数据类型 bfloat16/float16")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    # 策略模型结构与序列长度。max_seq_len 用于配置模型容量，max_total_len 是训练侧实际截断上限。
    parser.add_argument('--hidden_size', default=768, type=int, help="模型隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="模型层数")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="最大序列长度")
    parser.add_argument("--max_gen_len", type=int, default=768, help="单次最大生成长度")
    parser.add_argument("--max_total_len", type=int, default=2500, help="训练侧最终总长度上界")
    # 每条数据会展开成 num_generations 条轨迹；下列参数控制采样和策略目标。
    parser.add_argument("--data_path", type=str, default="../dataset/agent_rl.jsonl", help="训练数据路径")
    parser.add_argument("--num_generations", type=int, default=4, help="每个prompt生成数量")
    parser.add_argument("--beta", type=float, default=0.1, help="KL散度惩罚系数")
    parser.add_argument("--loss_type", type=str, default="cispo", choices=["grpo", "cispo"], help="loss类型")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GRPO的PPO clip epsilon")
    parser.add_argument("--epsilon_high", type=float, default=5.0, help="epsilon上界")
    # 初始化权重与断点恢复。from_weight 只提供模型参数；from_resume 还会恢复优化器、调度器与步数。
    parser.add_argument('--from_weight', default='full_sft', type=str, help="加载预训练权重名称")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否从checkpoint恢复")
    # 可选训练观测与调试输出。
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb记录")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Agent-RL", help="wandb项目名称")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile")
    parser.add_argument("--debug_mode", action="store_true", help="调试模式")
    parser.add_argument("--debug_interval", type=int, default=20, help="调试日志间隔")
    # 对话格式与奖励模型；thinking_ratio 是 rollout 起始提示采用开放思考头的概率。
    parser.add_argument("--thinking_ratio", type=float, default=0.1, help="按概率开启thinking（0.0~1.0）")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    # Torch 在当前进程内采样；SGLang 通过 HTTP 服务采样，需要预先启动对应服务。
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"], help="rollout引擎类型")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998", help="SGLang服务器URL")
    parser.add_argument("--sglang_model_path", type=str, default="../model", help="SGLang tokenizer路径")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_agent", help="SGLang共享存储路径")
    args = parser.parse_args()

    # 单机运行时返回 rank 0；torchrun 下初始化进程组并为每个进程选择独立 GPU。
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    # DDP 中各进程使用不同随机种子；数据采样顺序会在每个 epoch 另行统一设种。
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    os.makedirs(args.save_dir, exist_ok=True)
    # 模型上下文容量预留 prompt 最大长度加上生成上限，并在此配置 MoE 开关。
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               max_seq_len=args.max_seq_len + args.max_gen_len, use_moe=bool(args.use_moe))
    # 显式启用时读取续训状态；未传 model 给 lm_checkpoint 表示执行读取模式。
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume == 1 else None

    # CPU 使用普通精度上下文；CUDA 使用 autocast 降低前向显存和计算开销。
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    wandb = None
    if args.use_wandb and is_main_process():
        # 仓库使用 SwanLab 的 W&B 兼容接口；续训时尽量接回 checkpoint 记录的同一 run。
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb.init(project=args.wandb_project, name=f"Agent-RL-E{args.epochs}-B{args.batch_size}-LR{args.learning_rate}", id=wandb_id, resume=resume)

    # 可训练策略模型从指定权重初始化；tokenizer 同时提供词表与 chat_template。
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)

    # Ref 与策略从相同初始权重开始，但之后冻结不更新，为 KL 惩罚提供稳定参照。
    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)

    # 奖励模型用于无工具回答的文本质量打分；工具轨迹主要走 GT 和调用格式规则分支。
    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
    Logger(f'Loaded reward model from {args.reward_model_path}')
    # 统一构造采样接口：本地 Torch 引擎直接持有策略模型，SGLang 引擎连接外部服务。
    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine,
        policy_model=model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )
    # 数据集输出结构化对话、工具 schema 和只用于奖励校验的 gt；模型不会在输入中看到 gt。
    train_ds = AgentRLDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    # 分布式时由 DistributedSampler 给各 rank 划分样本；单进程时后面使用随机排列的索引。
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    # 保留不同样本的 list 字段，而不尝试默认堆叠变长的消息历史。
    def collate_fn(batch): return {'messages': [b['messages'] for b in batch], 'tools': [b['tools'] for b in batch], 'gt': [b['gt'] for b in batch]}
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, collate_fn=collate_fn)
    iters = len(loader_for_count)
    total_optimizer_steps = math.ceil(iters / args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)

    # 恢复策略、优化器与学习率调度器状态；start_epoch/start_step 用于重建数据位置。
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        # RoPE buffer 在各 rank 一致；关闭 buffer 广播，避免每次前向同步重复数据。
        model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
    # 训练开始前也同步一次，确保外部采样引擎加载的是当前初始化/恢复后的策略。
    rollout_engine.update_policy(model)

    for epoch in range(start_epoch, args.epochs):
        # DistributedSampler 需要每轮设定 epoch，才能在各 rank 上生成一致的新排列。
        train_sampler and train_sampler.set_epoch(epoch)
        # 单进程的随机排列由种子保证可重建；DDP 场景实际优先使用 train_sampler。
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        # 只在恢复所在 epoch 跳过已完成批次；后续 epoch 从头开始。
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        # SkipBatchSampler 按批跳过，并兼容 DistributedSampler 与普通索引列表。
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
        if skip > 0:
            Logger(f'Epoch [{epoch+1}/{args.epochs}]: skip {start_step} steps')
            # len(loader) 只统计剩余批次，加回 skip 后得到原始 epoch 的 step 编号上界。
            rl_train_epoch(epoch, loader, len(loader) + skip, rollout_engine, ref_model, reward_model, start_step, wandb, use_sglang = (args.rollout_engine == "sglang"))
        else:
            rl_train_epoch(epoch, loader, len(loader), rollout_engine, ref_model, reward_model, 0, wandb, use_sglang = (args.rollout_engine == "sglang"))

    if dist.is_initialized():
        # 所有 rank 完成后再一起销毁通信进程组。
        dist.barrier()
        dist.destroy_process_group()
