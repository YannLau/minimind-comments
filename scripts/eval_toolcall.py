"""交互式验证 MiniMind 的工具调用（Tool Calling）能力。

本脚本并不是只检查一次模型输出，而是演示完整的“智能体式”多轮闭环：

1. 把用户问题和当前允许使用的工具描述交给模型；
2. 模型可以直接作答，也可以生成一个或多个工具调用；
3. Python 解析调用、执行本文件中的模拟工具，再把结果写回对话历史；
4. 模型读取工具结果并继续生成，直到不再请求工具为止。

建议先阅读（有助于理解本文件）：
1. ``README.md`` 的“5.1 Tool Calling”：了解训练样本、消息角色和标签格式。
2. ``trainer/train_tokenizer.py`` 中的 ``chat_template``：理解 ``tools``、
   ``tool_calls`` 和 ``role='tool'`` 最终如何展开为模型能看到的文本。
3. ``model/model_minimind.py`` 中的 ``MiniMindConfig``、``MiniMindForCausalLM``：
   了解本地模型的结构，以及为何能直接使用 Transformers 的 ``generate``。

进一步推荐阅读：
1. ``trainer/train_agent.py`` 的 ``parse_tool_calls``、``execute_tool`` 和
   ``rollout_single``：观察训练阶段如何执行相同的“生成—调用—回填”循环。
2. ``scripts/chat_api.py``：了解普通多轮聊天如何通过 OpenAI 兼容接口流式生成。
3. ``scripts/serve_openai_api.py``：继续追踪本地模型如何被包装成 OpenAI 兼容服务。
4. ``trainer/trainer_utils.py`` 的 ``safe_math_eval``、``setup_seed`` 和
   ``get_model_params``：理解本脚本复用的数学求值、随机种子与参数统计工具。

两种后端的关键区别：``local`` 直接加载权重并解析文本中的 ``<tool_call>``；
``api`` 则请求一个 OpenAI 兼容服务，优先读取结构化 ``tool_calls`` 字段。二者最后都会
被归一化为 ``name``、``arguments`` 等字段，再进入同一个工具执行循环。
"""

import os
import sys

# 直接执行 ``python scripts/eval_toolcall.py`` 时，Python 默认只把 scripts/ 放进
# 模块搜索路径。这里显式加入仓库根目录，才能导入同级的 model/ 和 trainer/ 包。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re
import json
import time
import random
import argparse
import warnings
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from openai import OpenAI
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import setup_seed, get_model_params, safe_math_eval

# 推理时部分第三方库会输出不影响演示的警告。该设置作用于整个进程，因此排查环境或
# 精度问题时可临时注释掉，以免遗漏有价值的警告信息。
warnings.filterwarnings('ignore')

# OpenAI 风格的工具定义列表。每项都由两部分组成：
# - ``name`` / ``description`` 告诉模型“工具叫什么、能做什么”；
# - ``parameters`` 使用 JSON Schema 约束参数名、类型和必填项。
# 这里只是把约束提供给模型与 API 服务；``execute_tool`` 本身没有完整执行 JSON Schema
# 校验，因此模型传错参数时，最终会由下方模拟函数的默认值或异常处理兜底。
TOOLS = [
    {"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式的结果，支持加减乘除、幂运算、开方等", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "数学表达式，如123+456、2**10、sqrt(144)"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "get_current_time", "description": "获取当前日期和时间，支持指定时区", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "description": "时区名称，如Asia/Shanghai、America/New_York", "default": "Asia/Shanghai"}}, "required": []}}},
    {"type": "function", "function": {"name": "random_number", "description": "生成指定范围内的随机数", "parameters": {"type": "object", "properties": {"min": {"type": "integer", "description": "最小值", "default": 0}, "max": {"type": "integer", "description": "最大值", "default": 100}}, "required": []}}},
    {"type": "function", "function": {"name": "text_length", "description": "计算文本的字符数和单词数", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "要统计的文本"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "unit_converter", "description": "进行单位换算，支持长度、重量、温度等", "parameters": {"type": "object", "properties": {"value": {"type": "number", "description": "要转换的数值"}, "from_unit": {"type": "string", "description": "源单位，如km、miles、kg、pounds、celsius、fahrenheit"}, "to_unit": {"type": "string", "description": "目标单位"}}, "required": ["value", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {"name": "get_current_weather", "description": "获取指定城市的当前天气信息，包括温度、湿度和天气状况", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "城市名称，如北京、上海、New York"}, "unit": {"type": "string", "description": "温度单位，celsius或fahrenheit", "enum": ["celsius", "fahrenheit"], "default": "celsius"}}, "required": ["location"]}}},
    {"type": "function", "function": {"name": "get_exchange_rate", "description": "查询两种货币之间的实时汇率", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string", "description": "源货币代码，如USD、CNY、EUR"}, "to_currency": {"type": "string", "description": "目标货币代码，如USD、CNY、EUR"}}, "required": ["from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "translate_text", "description": "将文本翻译成目标语言", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "要翻译的文本"}, "target_language": {"type": "string", "description": "目标语言，如english、chinese、japanese、french"}}, "required": ["text", "target_language"]}}},
]

# 工具名称到 Python 实现的映射。这里全部是便于离线测试的模拟实现，而非生产服务：
# - 天气、汇率和翻译返回固定结果；
# - 当前时间只回显请求的时区名称，并未真的按该时区换算；
# - 单位换算固定乘以“公里到英里”的系数，不会根据单位组合选择公式；
# - ``safe_math_eval`` 来自 trainer_utils.py，用于受限地计算数学表达式。
# 因此，本脚本评测的重点是模型能否正确选择工具、组织参数并利用返回值，而不是外部数据
# 是否实时、各工具的业务实现是否完备。
MOCK_RESULTS = {
    "calculate_math": lambda args: {"result": str(safe_math_eval(args.get("expression", "0")))},
    "get_current_time": lambda args: {"datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "timezone": args.get("timezone", "Asia/Shanghai")},
    "random_number": lambda args: {"result": random.randint(int(args.get("min", 0)), int(args.get("max", 100)))},
    "text_length": lambda args: {"characters": len(args.get("text", "")), "words": len(args.get("text", "").split())},
    "unit_converter": lambda args: {"result": round(float(args.get("value", 0)) * 0.621371, 2), "from": f"{args.get('value', 0)} {args.get('from_unit', '')}", "to": args.get("to_unit", "")},
    "get_current_weather": lambda args: {"city": args.get("location"), "temperature": "22°C", "humidity": "65%", "condition": "晴"},
    "get_exchange_rate": lambda args: {"from": args.get("from_currency", ""), "to": args.get("to_currency", ""), "rate": 7.15},
    "translate_text": lambda args: {"translated": "hello world"},
}

# 预先建立名称索引，自动测试时即可由简短的名称列表快速取回完整 JSON Schema。
TOOL_MAP = {t["function"]["name"]: t for t in TOOLS}


def get_tools(names):
    """按名称挑选本轮允许模型使用的完整工具定义。

    ``names`` 中若出现未注册名称会直接触发 ``KeyError``，这有助于尽早发现测试用例
    拼写错误，而不是悄悄把错误工具忽略掉。
    """
    return [TOOL_MAP[n] for n in names]


# 内置测试既覆盖单工具，也覆盖需要连续两次调用的任务。例如“先生成随机数，再算平方”
# 会检验第二次模型生成能否利用第一次工具响应。每个用例只暴露少量候选工具，还可顺带
# 观察模型是否能避开语义不相关的干扰工具。最后一个英文用例用于做简单的跨语言检查。
TEST_CASES = [
    {"prompt": "帮我算一下 256 乘以 37 等于多少", "tools": ["calculate_math", "get_current_time"]},
    {"prompt": "现在几点了？", "tools": ["get_current_time", "random_number"]},
    {"prompt": "帮我把100公里换算成英里", "tools": ["unit_converter", "calculate_math"]},
    {"prompt": "帮我生成一个1到1000的随机数，然后计算它的平方", "tools": ["random_number", "calculate_math", "text_length"]},
    {"prompt": "北京今天天气怎么样？", "tools": ["get_current_weather", "get_current_time"]},
    {"prompt": "查一下美元兑人民币汇率", "tools": ["get_exchange_rate", "get_current_time"]},
    {"prompt": "把'你好世界'翻译成英文", "tools": ["translate_text", "text_length"]},
    {"prompt": "What is the weather in Tokyo? Also convert 30 celsius to fahrenheit.", "tools": ["get_current_weather", "unit_converter", "get_current_time"]},
]


def init_model(args):
    """初始化本地推理所需的模型和分词器。

    仓库支持两类模型目录：

    * ``load_from`` 路径包含 ``model`` 时，按项目原生方式创建 MiniMind 结构，再从单独
      的 ``.pth`` 文件载入参数；
    * 否则视为已经导出的 Hugging Face Transformers 目录，交给
      ``AutoModelForCausalLM.from_pretrained`` 直接恢复结构和权重。

    返回前会切换到半精度和评估模式，并移动到指定设备。``eval`` 会关闭 dropout；
    ``half`` 能降低显存占用，但 CPU 对 FP16 算子的支持和性能可能不如 GPU。
    """
    # 分词器还携带仓库自定义的 chat_template；工具描述和历史消息依赖它拼成最终提示词。
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        # 原生权重文件只保存 state_dict，因此必须用命令行参数重建完全一致的模型结构。
        model = MiniMindForCausalLM(MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe)))
        moe_suffix = '_moe' if args.use_moe else ''
        # 例如：./../out/full_sft_768.pth 或 ./../out/full_sft_768_moe.pth。
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
    else:
        # ``trust_remote_code`` 允许导出目录通过自定义代码注册 MiniMind 模型类。
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    # 仅打印模型参数规模，方便确认实际加载的模型配置是否符合预期。
    get_model_params(model, model.config)
    return model.half().eval().to(args.device), tokenizer


def parse_tool_calls(text):
    """从本地模型生成文本中提取所有 ``<tool_call>...</tool_call>``。

    标签内部应是形如 ``{"name": "工具名", "arguments": {...}}`` 的 JSON 对象。
    ``re.DOTALL`` 让点号也能匹配换行，所以模型输出多行 JSON 时仍可解析。单个片段若不是
    合法 JSON 会被跳过，其他合法调用仍然保留；返回空列表表示没有可执行调用。
    """
    matches = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    calls = []
    for m in matches:
        try:
            calls.append(json.loads(m.strip()))
        except Exception:
            # 评测脚本选择容错而非中断：格式错误会被视为模型没有产出这个调用。
            pass
    return calls


def parse_tool_call_from_text(content):
    """把 API 返回正文里的标签式调用转换为近似 OpenAI 的结构化格式。

    部分 OpenAI 兼容服务不会填写响应的 ``tool_calls`` 字段，而会把 MiniMind 原生的
    ``<tool_call>`` 文本原样放进 ``content``。这个兼容分支为每个调用补一个本地 ID，
    并把 ``arguments`` 重新序列化为字符串，使后续逻辑可以与标准 API 响应共用。

    没有合法调用时返回 ``None``，而不是空列表，与 OpenAI SDK 的缺省语义保持接近。
    """
    pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    matches = re.findall(pattern, content, re.DOTALL)
    if not matches:
        return None
    tool_calls = []
    for i, match in enumerate(matches):
        try:
            data = json.loads(match)
            tool_calls.append({
                # 标准协议要求工具响应通过 tool_call_id 对应到原调用。
                "id": f"call_{i}",
                "function": {"name": data.get("name", ""), "arguments": json.dumps(data.get("arguments", {}), ensure_ascii=False)}
            })
        except Exception:
            # 一个坏片段不应阻止同一响应中的其他工具调用继续执行。
            pass
    return tool_calls if tool_calls else None


def execute_tool(call, arguments=None):
    """统一执行本地或 API 后端产生的工具调用，并始终返回可 JSON 化的字典。

    本地模式传入完整 ``call`` 字典；API 模式传入工具名字符串和单独的 ``arguments``。
    API 的 arguments 通常是 JSON 字符串，本地模式通常已经是字典，因此这里先统一反序列化。
    未知工具、非法参数或模拟函数内部异常都会转换成错误字典，随后仍会作为工具响应交还
    模型，让模型有机会解释错误或改用其他参数。
    """
    # 兼容两种调用约定，先取得工具名。
    name = call.get("name", "") if isinstance(call, dict) else call
    try:
        raw_args = call.get("arguments", {}) if isinstance(call, dict) else arguments
        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
    except Exception:
        # arguments 不是合法 JSON 时用空字典兜底，具体工具再应用默认值或返回错误。
        args = {}
    fn = MOCK_RESULTS.get(name)
    if not fn:
        return {"error": f"未知工具: {name}"}
    try:
        return fn(args)
    except Exception as e:
        return {"error": f"工具执行失败: {str(e)[:80]}"}


def generate(model, tokenizer, messages, tools, args):
    """使用本地模型完成一轮生成，并返回本轮新增的纯文本。

    ``messages`` 包含截至当前轮的全部用户、助手和工具消息；``tools`` 是本轮可见的工具
    Schema。聊天模板负责把这些结构化数据展开为训练时见过的标签格式。TextStreamer
    只负责边生成边打印，真正供程序解析的文本仍由完整 token 序列解码得到。
    """
    # 跳过输入提示和特殊 token，只把本轮模型新生成的可读内容实时打印到终端。
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    # open_thinking=False 会由仓库模板注入空 think 块，让模型直接进行工具调用或回答。
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools, open_thinking=False)
    # truncation 防止上下文超过分词器上限；BatchEncoding.to 会移动其中的所有张量。
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(args.device)
    st = time.time()
    print('🧠: ', end='')
    generated_ids = model.generate(
        inputs["input_ids"], attention_mask=inputs["attention_mask"],
        # do_sample=True 表示采用采样生成，temperature 与 top_p 共同控制随机性。
        max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        top_p=args.top_p, temperature=args.temperature
    )
    # generate 的结果含“原提示 + 新输出”，因此按输入长度切片，只保留当前助手回复。
    response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
    # 速度使用本轮生成 token 数除以总耗时，是包含首 token 延迟的整体平均速度。
    print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s') if args.show_speed else print()
    return response


def chat_api(client, messages, tools, args, stream=True):
    """通过 OpenAI 兼容接口完成一轮生成，返回正文与工具调用。

    非流式响应可以一次读取完整 ``message``；流式响应则把服务器逐块发送的正文、调用 ID、
    函数名和参数片段重新拼接。兼容服务若只在正文输出 MiniMind 标签，本函数还会退回到
    ``parse_tool_call_from_text`` 解析。
    """
    response = client.chat.completions.create(
        model=args.api_model, messages=messages, tools=tools,
        stream=stream, temperature=args.temperature,
        max_tokens=8192, top_p=args.top_p
    )
    if not stream:
        choice = response.choices[0]
        # 工具调用轮的自然语言正文经常为 None，统一成空串便于打印和写入历史。
        content = choice.message.content or ""
        tool_calls = choice.message.tool_calls
        if not tool_calls:
            # 兼容“正文中输出 <tool_call> 标签、但未填结构化字段”的服务端。
            tool_calls = parse_tool_call_from_text(content)
        print(f'🧠: {content}')
        return content, tool_calls
    print('🧠: ', end='', flush=True)
    content, tool_calls = "", None
    for chunk in response:
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)
            content += delta.content
        if delta.tool_calls:
            if tool_calls is None:
                tool_calls = []
            for tc_chunk in delta.tool_calls:
                # 同一次响应可能并行请求多个工具。index 指明当前增量属于第几个调用；
                # 某些兼容服务不提供 index，此时就把它视为下一个新调用。
                idx = tc_chunk.index if tc_chunk.index is not None else len(tool_calls)
                # 流式块可能先抵达较大的 index，先补占位项再安全写入对应位置。
                while len(tool_calls) <= idx:
                    tool_calls.append({
                        "id": "",
                        "function": {"name": "", "arguments": ""}
                    })
                # ID、函数名和 JSON 参数都可能被拆成若干字符串片段，必须按到达顺序累加。
                if tc_chunk.id:
                    tool_calls[idx]["id"] += tc_chunk.id
                if tc_chunk.function:
                    if tc_chunk.function.name:
                        tool_calls[idx]["function"]["name"] += tc_chunk.function.name
                    if tc_chunk.function.arguments:
                        tool_calls[idx]["function"]["arguments"] += tc_chunk.function.arguments
    print()
    if not tool_calls:
        # 流式接口同样保留标签文本的兼容解析路径。
        tool_calls = parse_tool_call_from_text(content)
    return content, tool_calls


def run_case(prompt, tools, args, model=None, tokenizer=None, client=None):
    """运行一个测试问题，循环处理工具调用，直到模型给出最终回答。

    对话历史 ``messages`` 是这个函数的核心状态。每轮依次追加 assistant 消息和若干 tool
    消息，下一轮再把完整历史交给模型。函数没有显式最大轮数；正常情况下模型读到工具
    结果后会停止调用并回答，若模型持续请求工具，则循环也会持续运行。
    """
    # 每个测试用例都从一条全新的用户消息开始，不与前一个用例共享上下文。
    messages = [{"role": "user", "content": prompt}]
    while True:
        if args.backend == 'local':
            content = generate(model, tokenizer, messages, tools, args)
            tool_calls = parse_tool_calls(content)
        else:
            content, tool_calls = chat_api(client, messages, tools, args, stream=bool(args.stream))
        # 没有工具调用说明 content 已是最终回答（或调用格式无法解析），当前用例结束。
        if not tool_calls:
            break
        # OpenAI SDK 非流式返回的是对象，兼容解析和流式分支返回的则可能是字典。
        # 这里把 API 后端的不同形态都压平成后续所需的 id/name/arguments 三个字段。
        tool_calls = [{
            "id": tc.id if hasattr(tc, 'id') else tc.get("id", ""),
            "name": tc.function.name if hasattr(tc, 'function') else tc["function"]["name"],
            "arguments": tc.function.arguments if hasattr(tc, 'function') else tc["function"]["arguments"]
        } for tc in tool_calls] if args.backend == 'api' else tool_calls
        # 必须先保存助手的调用请求，随后再追加对应工具结果，消息顺序才能符合聊天模板/API
        # 协议。本地模板能从 content 中自行识别标签；API 协议则要求显式提供 tool_calls。
        messages.append({"role": "assistant", "content": content} if args.backend == 'local' else {"role": "assistant", "content": content, "tool_calls": [{"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}} for tc in tool_calls]})
        for tc in tool_calls:
            name = tc["name"]
            arguments = tc["arguments"]
            print(f'📞 [Tool Calling]: {name} | args={arguments}')
            # 本地调用传整个字典，API 调用传“名称 + JSON 参数串”；execute_tool 会统一处理。
            result = execute_tool(tc if args.backend == 'local' else name, arguments)
            print(f'✅ [Tool Called]: {json.dumps(result, ensure_ascii=False)}')
            # API 必须携带 tool_call_id，服务端才能把结果与并行调用中的原请求对应起来；
            # 本地聊天模板只依赖消息次序，会将 role=tool 包装为 <tool_response> 标签。
            messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)} if args.backend == 'local' else {"role": "tool", "content": json.dumps(result, ensure_ascii=False), "tool_call_id": tc["id"]})


def main():
    """解析命令行参数、初始化后端，并进入自动或手动评测入口。"""
    parser = argparse.ArgumentParser(description="MiniMind ToolCall评测")
    parser.add_argument('--backend', default='local', choices=['local', 'api'], type=str, help="推理后端（local=本地模型，api=OpenAI兼容接口）")
    parser.add_argument('--load_from', default='../model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='../out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_new_tokens', default=512, type=int, help="最大生成长度")
    parser.add_argument('--temperature', default=0.9, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.9, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--show_speed', default=0, type=int, help="显示decode速度（tokens/s）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    parser.add_argument('--api_base_url', default="http://localhost:11434/v1", type=str, help="OpenAI兼容接口的base_url")
    parser.add_argument('--api_key', default='sk-123', type=str, help="OpenAI兼容接口的api_key")
    parser.add_argument('--api_model', default='jingyaogong/minimind-3:latest', type=str, help="API请求时使用的模型名称")
    parser.add_argument('--stream', default=1, type=int, help="API模式下是否流式输出（0=否，1=是）")
    args = parser.parse_args()

    # 只初始化所选后端，避免 API 模式无谓占用显存，也避免本地模式要求 API 服务在线。
    model = tokenizer = client = None
    if args.backend == 'local': model, tokenizer = init_model(args)
    else: client = OpenAI(api_key=args.api_key, base_url=args.api_base_url)

    # 0 会依次执行 TEST_CASES；1 会不断读取用户输入，直接回车空字符串即可结束。
    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))

    # 手动模式用 iter(callable, sentinel) 构造迭代器：每轮调用 lambda 读取一次输入，直到
    # 返回值与 sentinel 字典相等。手动测试会向模型开放全部工具。
    cases = [{"prompt": case["prompt"], "tools": get_tools(case["tools"]), "tool_names": case["tools"]} for case in TEST_CASES] if input_mode == 0 else iter(lambda: {"prompt": input('💬: '), "tools": TOOLS, "tool_names": [t["function"]["name"] for t in TOOLS]}, {"prompt": "", "tools": TOOLS, "tool_names": []})
    for case in cases:
        if not case["prompt"]: break
        # 每个用例使用一个新的随机种子；它同时影响采样生成和 random_number 模拟工具。
        setup_seed(random.randint(0, 31415926))
        if input_mode == 0:
            print(f'📦 可用工具: {case["tool_names"]}\n')
            print(f'💬: {case["prompt"]}')
        run_case(case["prompt"], case["tools"], args, model=model, tokenizer=tokenizer, client=client)
        print('\n' + '-' * 50 + '\n')


if __name__ == "__main__":
    # 只有直接运行本文件时才启动交互；被其他模块导入时只暴露上面的函数和常量。
    main()
