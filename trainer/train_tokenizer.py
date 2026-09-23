"""从 JSONL 语料训练 ByteLevel BPE 分词器，并检查生成的 Hugging Face 配置。

建议先阅读（有助于理解本文件）：
    1. ``README.md`` 的“数据介绍 / Tokenizer”：了解 token、词表大小以及项目为何
       统一使用仓库自带的 ``model/`` 分词器。
    2. ``dataset/lm_dataset.py`` 中的 ``PretrainDataset.__getitem__`` 和
       ``SFTDataset.create_chat_prompt``：分别看纯文本如何编码、对话如何套用聊天模板。
    3. ``model/tokenizer_config.json`` 开头的特殊 token 配置：对照本文件最终写出的配置。

读完本文件后，推荐继续阅读：
    1. ``dataset/lm_dataset.py`` 中的 ``SFTDataset.generate_labels``：理解模板里的
       ``<|im_start|>`` / ``<|im_end|>`` 如何界定需要训练的 assistant 回复。
    2. ``trainer/trainer_utils.py`` 中的 ``init_model`` 和 ``trainer/train_pretrain.py``：
       看训练入口如何加载 ``model/`` 中的分词器、把 token id 交给模型。
    3. ``model/model_minimind.py`` 中的 ``MiniMindConfig``：了解词表大小、BOS/EOS
       token id 必须与模型配置相匹配。

注意：MiniMind 已提供 ``model/`` 下的分词器。本脚本仅用于学习和参考；重新训练会
改变文本与 token id 的对应关系，已有模型权重和其他人使用的词表便不能直接兼容。
脚本中的相对路径以运行命令时的工作目录为基准，默认从 ``trainer/`` 目录运行。
"""

import os
import json
from tokenizers import decoders, models, pre_tokenizers, trainers, Tokenizer

# 输入是逐行一个 JSON 对象的语料；默认取 SFT 示例数据。训练结果放在独立目录，
# 避免覆盖项目正式使用的 model/tokenizer.json 和 model/tokenizer_config.json。
DATA_PATH = '../dataset/sft_t2t_mini.jsonl'
TOKENIZER_DIR = '../model_learn_tokenizer/'
# BPE 的目标词表容量，包含特殊 token；MiniMind 模型默认也按 6400 个词表项配置。
VOCAB_SIZE = 6400
# 给协议标记、工具/思考标记和未来扩展预留的 token 总数。
SPECIAL_TOKENS_NUM = 36
MAX_LINES = 0  # 0 表示读取全部；设为正数则只取前 N 条（快速试跑）

def get_texts(data_path, max_lines=MAX_LINES):
    """逐行读取 JSONL，按需产出用于学习 BPE 合并规则的非空文本。

    ``pretrain_t2t(_mini).jsonl`` 的每行是 ``{"text": ...}``；
    ``sft_t2t(_mini).jsonl`` 则是 ``{"conversations": [{"role": ..., "content": ...}, ...]}``。
    对话数据只提取各消息的 content，以换行连接；role 等字段不参与 BPE 训练。
    这是生成器：读取到一条有效样本才产出一条文本，不会先把全部语料放进内存。
    ``max_lines`` 限制的是有效文本条数，空行、坏 JSON 和空内容都不计数。

    若没有有效文本就抛错；否则训练器可能仅根据初始字节字母表生成一个没有
    BPE 合并规则的词表，却看起来像是训练成功了。
    """
    used = 0
    with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if max_lines and used >= max_lines:
                break
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                # 一行损坏不妨碍继续使用后续样本。
                continue
            if 'text' in data:
                text = str(data['text'])
            else:
                contents = [item.get('content') for item in data.get('conversations', []) if item.get('content')]
                text = "\n".join(contents) if contents else ''
            if text.strip():
                used += 1
                yield text
    if used == 0:
        raise ValueError(
            f'{data_path} 中没有可用文本：每行应为 {{"text": ...}} 或 '
            f'{{"conversations": [{{"role":..., "content":...}}, ...]}}。'
            f'继续训练只会得到一个空词表。'
        )

def train_tokenizer(data_path, tokenizer_dir, vocab_size, special_tokens_num=SPECIAL_TOKENS_NUM):
    """训练底层 BPE 词表，保存文件，并写出供 AutoTokenizer 使用的配置。"""
    # BPE 从基础符号开始，统计语料中常见的相邻符号组合并逐步合并成新 token。
    # ByteLevel 先把文本映射到字节级表示；False 表示不会在输入前凭空补一个空格。
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # 这些是对话边界、多模态占位等协议标记。即使训练语料中没有出现，也要在词表中
    # 占据稳定的 id；其中 <|im_start|> / <|im_end|> 分别用作 BOS / EOS。
    special_tokens_list = [
        "<|endoftext|>", "<|im_start|>", "<|im_end|>", 
        "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>", 
        "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>", 
        "<|audio_start|>", "<|audio_end|>", "<|audio_pad|>", "<tts_pad>", "<tts_text_bos>", "<tts_text_eod>", "<tts_text_bos_single>"
    ]
    
    # 工具调用与思考标签要保持完整的 token，但下文会把其 special 标志改为 False：
    # 后续 decode(..., skip_special_tokens=True) 不会自动把这些可见标签删除。
    additional_tokens_list = [
        "<tool_call>", "</tool_call>",
        "<tool_response>", "</tool_response>",
        "<think>", "</think>"
    ]
    # 补齐到 special_tokens_num 个保留位置，便于以后扩展而不挪动已有标记的 id。
    num_buffer = special_tokens_num - len(special_tokens_list + additional_tokens_list)
    buffer_tokens = [f"<|buffer{i}|>" for i in range(1, num_buffer + 1)]  # 预留未来扩展位置
    all_special_tokens = special_tokens_list + additional_tokens_list + buffer_tokens
    # initial_alphabet 预置 ByteLevel 所需的完整字节字母表，让未见过的字符仍能
    # 拆成字节编码；special_tokens 则先于普通 BPE 合并词进入词表。
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=all_special_tokens
    )
    # train_from_iterator 会逐条消费 get_texts 产出的文本。
    texts = get_texts(data_path)
    tokenizer.train_from_iterator(texts, trainer=trainer)
    # 编码采用 ByteLevel，解码也需对应的 ByteLevel 解码器以还原原始文本。
    tokenizer.decoder = decoders.ByteLevel()
    # 显式登记真正的特殊 token；它们已在 BpeTrainer 的列表中，通常不会新增 id。
    tokenizer.add_special_tokens(special_tokens_list)

    # tokenizer.json 是完整的 fast tokenizer；tokenizer.model.save 另存 BPE 的
    # vocab.json / merges.txt，便于单独查看词表以及“把哪些片段合并”的规则。
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    tokenizer.model.save(tokenizer_dir)
    tokenizer_json_path = os.path.join(tokenizer_dir, "tokenizer.json")
    with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)
    # BpeTrainer 接收的 all_special_tokens 都会先以 special=True 保存。
    # 这里仅保留协议标记的 special 属性；工具/思考标签及 buffer 仍是独立 token，
    # 但不会因为调用 skip_special_tokens=True 而被省略。
    for token_info in tokenizer_data.get('added_tokens', []):
        if token_info['content'] not in special_tokens_list:
            token_info['special'] = False
    with open(tokenizer_json_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
    
    # Hugging Face 的 tokenizer_config.json 也要记录每个保留 id 的文本及属性，
    # 与上面对 tokenizer.json 的 special 标志保持一致。
    added_tokens_decoder = {}
    for i, token in enumerate(all_special_tokens):
        idx = tokenizer.token_to_id(token)
        added_tokens_decoder[str(idx)] = {
            "content": token,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True if token in special_tokens_list else False
        }

    # 以下配置由 transformers.AutoTokenizer.from_pretrained(tokenizer_dir) 读取。
    # add_bos/eos_token=False：普通 encode 不自动插入边界；预训练数据集自行添加，
    # 对话数据则由 chat_template 插入。PAD/UNK 共用 <|endoftext|> 的 id。
    # model_max_length 是 tokenizer 的输入长度元数据，不会改变模型实际支持的上下文长度。
    # chat_template 是 Jinja 模板，不是 Python 代码：apply_chat_template 会按消息 role
    # 拼出 <|im_start|>角色\n正文<|im_end|>，并在需要时追加等待 assistant 续写的开头。
    # 若提供 tools，模板先在 system 消息中列出工具定义，再把 assistant 的 tool_calls
    # 包在 <tool_call> 中，把连续的 tool 消息包在 <tool_response> 中。
    # assistant 回复包含 <think> 块：优先读取 reasoning_content，否则尝试从正文拆出；
    # add_generation_prompt=True 时，open_thinking 决定生成从思考块内部还是正文开始。
    # 这些文字也是模型训练/推理所见的协议，改动模板需同步考虑已有训练数据与模型权重。
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": added_tokens_decoder,
        "additional_special_tokens": [t for t in special_tokens_list if t not in ["<|endoftext|>"]],
        "bos_token": "<|im_start|>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|im_end|>",
        "legacy": True,
        "model_max_length": 131072,
        "pad_token": "<|endoftext|>",
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "unk_token": "<|endoftext|>",
        "image_token": "<|image_pad|>",
        "audio_token": "<|audio_pad|>",
        "video_token": "<|video_pad|>",
        "vision_bos_token": "<|vision_start|>",
        "vision_eos_token": "<|vision_end|>",
        "audio_bos_token": "<|audio_start|>",
        "audio_eos_token": "<|audio_end|>",
        "chat_template": "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is string %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in content %}\n                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n        {%- if true %}\n            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if open_thinking is defined and open_thinking is true %}\n        {{- '<think>\\n' }}\n    {%- else %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}",
        "tokenizer_class": "PreTrainedTokenizerFast"
    }

    # 配置单独保存后，AutoTokenizer 才知道 BOS/EOS/PAD 等角色及聊天模板。
    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    print("Tokenizer training completed.")

def eval_tokenizer(tokenizer_dir):
    """重新加载磁盘产物，直观检查模板、编解码、压缩率和逐 token 解码。"""
    from transformers import AutoTokenizer
    # 用下游训练代码同样的入口加载，顺便检查文件能否被 transformers 识别。
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    # 构造多轮对话，以便看到 system/user/assistant 的边界是否按模板输出。
    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": '你来自哪里？'},
        {"role": "assistant", "content": '我来自月球'},
        {"role": "user", "content": '你到底来自哪里？'},
        {"role": "assistant", "content": '我来自地球'}
    ]
    # tokenize=False 只渲染模板文本；下面再单独编码，便于查看两步各自产物。
    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )
    print('-'*100)
    print(new_prompt)
    print('-'*100)
    print('tokenizer词表长度：', len(tokenizer))
    model_inputs = tokenizer(new_prompt)
    print('encoder长度：', len(model_inputs['input_ids']))
    # 保留特殊 token 再解码，才能核对原始模板文本是否完整还原。
    response = tokenizer.decode(model_inputs['input_ids'], skip_special_tokens=False)
    print('decoder一致性：', response == new_prompt, "\n")
    print('-'*100)
    print('压缩率测试（Chars/Tokens）：')
    # 中、英及混合文本的字符/token 比值只用于观察切分密度；比值越高，
    # 平均一个 token 承载的字符越多，并不等同于模型回答质量。
    test_texts = [
        # 中文样本 (约200字)
        "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器，该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大，可以设想，未来人工智能带来的科技产品，将会是人类智慧的“容器”。人工智能可以对人的意识、思维的信息过程的模拟。人工智能不是人的智能，但能像人那样思考、也可能超过人的智能。",
        "星际航行是指在星系内甚至星系间的空间中进行的航行。由于宇宙空间极其广阔，传统的化学火箭动力在恒星间航行时显得力不从心。科学家们提出了多种方案，包括离子推进器、核热火箭、甚至是利用反物质作为能源的设想。此外，曲率驱动和虫洞旅行等科幻概念也在理论物理研究中被反复探讨。尽管目前人类的足迹仅限于月球，但随着核聚变技术和材料科学的突破，前往火星乃至更遥远的太阳系边缘将成为可能。",
        # 英语样本（长度约数百字符）
        "Large language models (LLMs) are a type of artificial intelligence (AI) trained on vast amounts of text data to understand and generate human-like language. These models use deep learning techniques, specifically transformers, to process and predict the next word in a sequence. LLMs like GPT-4, Llama, and Claude have demonstrated remarkable capabilities in coding, translation, and creative writing. However, they also face challenges such as hallucinations, where the model generates factually incorrect information, and the need for significant computational resources.",
        "The development of sustainable energy is crucial for the future of our planet. As climate change continues to impact global weather patterns, transitioning from fossil fuels to renewable sources like solar, wind, and hydroelectric power has become an urgent priority. Innovations in battery storage technology and smart grid management are essential to ensure a reliable energy supply. International cooperation and policy frameworks are also necessary to drive the global shift towards a greener economy and reduce carbon emissions.",
        # 混合样本
        "Python 是一种高级编程语言，以其简洁的语法和强大的生态系统而闻名。It is widely used in data science, machine learning, and web development. 开发者可以利用 NumPy, Pandas, and PyTorch 等库快速构建复杂的应用。学习 Python 的过程非常愉快，因为它的代码读起来就像英语一样。Whether you are a beginner or an expert, Python offers something for everyone.",
    ]
    
    total_compression = 0
    for i, text in enumerate(test_texts):
        # len(text) 是 Python 字符数，不是 UTF-8 字节数；中英文结果不宜直接当作
        # 相同单位的压缩效率比较。这里仍能快速观察新词表是否出现异常切分。
        encoded = tokenizer.encode(text)
        token_count = len(encoded)
        char_count = len(text)
        compression_ratio = char_count / token_count
        total_compression += compression_ratio
        print(f"样本 {i+1} | 字符数: {char_count:4} | Tokens: {token_count:3} | 压缩率: {compression_ratio:.2f}")
    
    print(f"平均压缩率: {total_compression / len(test_texts):.2f}")
    print('-'*100)
    print('流式解码（字节缓冲）测试：')
    # 单个 ByteLevel token 可能只含某个汉字 UTF-8 编码的一部分；立即解码可能得到
    # 替代字符 \ufffd。先累积 id，直到能形成完整字符，再打印该段的原始 token 和文本。
    input_ids = model_inputs['input_ids']
    token_cache = []
    for tid in input_ids:
        token_cache.append(tid)
        current_decode = tokenizer.decode(token_cache)
        if current_decode and '\ufffd' not in current_decode:
            display_ids = token_cache[0] if len(token_cache) == 1 else token_cache
            raw_tokens = [tokenizer.convert_ids_to_tokens(int(t)) for t in (token_cache if isinstance(token_cache, list) else [token_cache])]
            print(f'Token ID: {str(display_ids):15} -> Raw: {str(raw_tokens):20} -> Decode Str: {current_decode}')
            token_cache = []

if __name__ == '__main__':
    # 直接运行脚本时先训练到独立目录，再用刚写出的文件做冒烟检查；
    # 被其他模块导入时不会自动训练。
    train_tokenizer(DATA_PATH, TOKENIZER_DIR, VOCAB_SIZE)
    eval_tokenizer(TOKENIZER_DIR)
