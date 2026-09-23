"""
MiniMind 的命令行推理与多轮对话入口。

这个脚本负责把“加载模型 → 整理提示词 → 分词 → 自回归生成 → 解码输出”串成一条
完整的推理链路，既可以依次运行内置问题，也可以进入终端交互模式。

建议先阅读（有助于理解本文件）：
1. ``README.md`` 中的“快速开始/模型推理”“监督微调（SFT）”“LoRA”和
   “RoPE 长度外推”章节：了解权重从哪里来、各命令行参数应如何搭配。
2. ``model/model_minimind.py``：重点阅读 ``MiniMindConfig``、
   ``MiniMindForCausalLM`` 及其 ``generate`` 方法，理解模型结构和逐 token 生成过程。
3. ``model/tokenizer_config.json``：关注 ``chat_template``，理解多轮消息如何被拼成模型
   真正接收的文本，以及 ``open_thinking`` 如何影响思考标签。

进一步推荐阅读：
1. ``model/model_lora.py``：理解 ``apply_lora`` 如何给线性层挂载低秩分支，以及
   ``load_lora`` 如何载入增量权重。
2. ``trainer/trainer_utils.py``：查看 ``setup_seed``、``get_model_params``，并对比训练侧的
   模型初始化方式。
3. ``dataset/lm_dataset.py`` 与 ``trainer/train_full_sft.py``：理解对话数据如何编码、
   如何构造标签，以及本脚本所加载的 SFT 权重是怎样训练出来的。
4. ``scripts/serve_openai_api.py`` 和 ``scripts/web_demo.py``：了解同一套模型如何被封装为
   OpenAI 风格 API 和 Web 界面；它们与本文件的核心推理流程基本一致。

常用示例：
    # 加载仓库原生 PyTorch 权重 out/full_sft_768.pth
    python eval_llm.py --load_from model --weight full_sft

    # 加载已经转换好的 Transformers 格式模型目录
    python eval_llm.py --load_from ./minimind-3

说明：本文件里的“原生权重”指仅保存 ``state_dict`` 的 ``.pth`` 文件；“Transformers
格式”则指包含 config、模型权重和 tokenizer 文件的完整目录。
"""

import time
import argparse
import random
import warnings

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import *
from trainer.trainer_utils import setup_seed, get_model_params

# 屏蔽第三方库的非致命警告，让终端中的模型回答更易阅读。调试兼容性问题时可暂时注释掉。
warnings.filterwarnings('ignore')


def init_model(args):
    """根据命令行参数加载 tokenizer 和模型，并将模型切换到推理状态。

    本函数支持两种互斥的加载方式：

    - ``args.load_from`` 中含有字符串 ``"model"``：把该路径当作 tokenizer 目录，
      同时在代码中实例化 MiniMind，再从 ``save_dir`` 读取原生 ``.pth`` 权重。
    - 其他情况：把 ``args.load_from`` 当作标准 Transformers 模型目录，一次性加载
      目录中的配置与模型权重。

    参数：
        args: ``argparse.Namespace``，包含模型结构、权重路径、设备和 LoRA 等参数。

    返回：
        ``(model, tokenizer)``：模型已转为半精度、切换至评估模式并移动到目标设备；
        tokenizer 保持加载时的配置。

    注意：原生 ``.pth`` 只保存参数张量，不保存模型结构。因此 ``hidden_size``、
    ``num_hidden_layers`` 和 ``use_moe`` 必须与训练该权重时完全一致，否则严格加载会失败。
    """
    # 两种加载路径都共用同一个 tokenizer 入口。默认的 ``model`` 目录包含仓库自带分词器。
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)

    # 这是仓库约定的原生模型判定方式，例如默认值 ``model`` 或 ``./model`` 都会进入此分支。
    if 'model' in args.load_from:
        # 先用命令行给出的结构参数搭出“空模型”，随后再把 .pth 中的张量填入各层。
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))

        # MoE 权重文件比稠密模型多 ``_moe`` 后缀，例如 full_sft_768_moe.pth。
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'

        # map_location 允许在 CPU 上读取 GPU 保存的权重（或直接加载到指定 GPU）。
        # strict=True 会校验每个参数名及形状，能尽早发现“模型配置与权重不匹配”。
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)

        # LoRA 是叠加在基础模型上的增量权重：必须先创建 LoRA 分支，再向分支中载入参数。
        # 字符串 ``None`` 是该命令行参数约定的“不启用”，并非 Python 的 None 对象。
        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, f'./{args.save_dir}/{args.lora_weight}_{args.hidden_size}{moe_suffix}.pth')
    else:
        # Transformers 格式目录自带 config，因此不再使用上面的结构参数手动构建模型。
        # trust_remote_code=True 允许目录中注册的自定义模型实现被 AutoModel 加载。
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)

    # 输出总参数量等信息，便于确认实际加载的是预期规格的模型。
    get_model_params(model, model.config)

    # half() 使用 FP16 减少显存；eval() 关闭 Dropout；to() 把参数移动到所选设备。
    # 三个方法都会返回模型本身，因此可以链式调用。
    return model.half().eval().to(args.device), tokenizer


def main():
    """解析命令行参数，并运行自动测试或交互式多轮对话。"""
    parser = argparse.ArgumentParser(description="MiniMind模型推理与对话")

    # ---------- 模型与权重 ----------
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")

    # 以下三个参数只参与“原生 .pth 权重”分支的模型构建，必须与训练配置匹配。
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")

    # RoPE 外推只扩展位置编码的可用范围，并不意味着模型自动获得了同等长度的理解能力。
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")

    # ---------- 生成策略 ----------
    # max_new_tokens 只限制“新生成”部分；总序列长度还要加上输入提示词的 token 数。
    parser.add_argument('--max_new_tokens', default=8192, type=int, help="最大生成长度（注意：并非模型实际长文本能力）")

    # temperature 调整概率分布的平滑程度；top_p 只保留累计概率达到阈值的候选 token。
    # 二者与下方 do_sample=True 共同构成随机采样，因此相同问题不一定每次得到相同回答。
    parser.add_argument('--temperature', default=0.85, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.95, type=float, help="nucleus采样阈值（0-1）")

    # open_thinking 会传给 tokenizer 的 chat_template，由模板决定是否预填充 <think> 标签。
    parser.add_argument('--open_thinking', default=0, type=int, help="是否开启自适应思考（0=否，1=是）")

    # 实际含义是“保留多少条历史消息”。一轮对话通常包含 user、assistant 两条消息，
    # 所以取偶数才能避免只保留某一轮的半边；0 表示每个问题都作为独立会话。
    parser.add_argument('--historys', default=0, type=int, help="携带历史对话轮数（需为偶数，0表示不携带历史）")
    parser.add_argument('--show_speed', default=1, type=int, help="显示decode速度（tokens/s）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    args = parser.parse_args()

    # 自动测试模式会按顺序询问这些覆盖常识、代码、解释和建议任务的示例问题。
    prompts = [
        '你有什么特长？',
        '为什么天空是蓝色的',
        '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程',
        '如果明天下雨，我应该如何出门',
        '比较一下猫和狗作为宠物的优缺点',
        '解释什么是机器学习',
        '推荐一些中国的美食'
    ]

    # conversation 始终采用 Transformers 通用的消息格式：
    # [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]。
    conversation = []
    model, tokenizer = init_model(args)

    # 0：无需继续输入，跑完上面的 prompts 后退出；1：不断读取终端输入，空行结束。
    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))

    # TextStreamer 会在 generate 每产出一些可解码 token 时立即打印，避免等待整段回答结束。
    # skip_prompt 避免重复打印用户输入；skip_special_tokens 隐藏 BOS/EOS 等控制 token。
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # iter(callable, sentinel) 会反复调用 input；当返回空字符串（用户直接回车）时停止迭代。
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')
    for prompt in prompt_iter:
        # 每个问题使用一个随机种子。它能同步 Python/PyTorch 等随机源，却故意不固定为同一个值，
        # 因而仍会展现采样生成的多样性。
        setup_seed(random.randint(0, 31415926))

        # 手动模式中的问题已经由 input 提示符显示；自动模式需要在这里额外打印当前问题。
        if input_mode == 0:
            print(f'💬: {prompt}')

        # 只保留末尾 N 条历史消息，再追加本轮用户输入。切片发生在追加之前，因此最终送入
        # 模板的消息数最多为 historys + 1。historys=0 时，每轮都会清空上下文。
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})

        # 预训练模型只学习“根据前文预测下一个 token”，尚未学习 user/assistant 对话协议，
        # 所以直接拼接 BOS（句首标记）和问题。经过 SFT 等对齐训练的权重则必须使用与训练
        # 阶段相同的 chat_template，补齐角色标记和“轮到 assistant 回答”的生成提示。
        if 'pretrain' in args.weight:
            inputs = tokenizer.bos_token + prompt
        else:
            inputs = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, open_thinking=bool(args.open_thinking))

        # 把字符串编码成形状为 [batch_size=1, sequence_length] 的 input_ids 和
        # attention_mask。truncation=True 会在超过 tokenizer 上限时截断输入；随后整体移至设备。
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        print('🧠: ', end='')
        st = time.time()

        # 自回归生成：模型每次根据已有 token 预测下一个 token，直到遇到 EOS 或达到上限。
        # streamer 负责边生成边显示；generate 的返回值仍包含“原输入 + 新生成”的完整 token。
        generated_ids = model.generate(
            inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p, temperature=args.temperature, repetition_penalty=1
        )

        # 跳过返回序列开头与输入等长的部分，只解码模型新生成的回答。虽然 streamer 已经
        # 打印过回答，仍需再次解码并保存文本，后续轮次才能把 assistant 回复放回历史上下文。
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": response})

        # token/s 只统计生成 token，不包含提示词；计时则覆盖完整 generate 调用。
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n') if args.show_speed else print('\n\n')


if __name__ == "__main__":
    # 只有直接执行 ``python eval_llm.py`` 时才进入 main；被其他模块 import 时不会启动交互。
    try:
        main()
    # Ctrl+C 或输入流结束都属于正常退出，不展示冗长的异常堆栈。
    except (EOFError, KeyboardInterrupt):
        print()
