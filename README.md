# 微信聊天角色扮演

使用 Qwen3 + LoRA 微调，基于微信聊天记录让模型模仿目标人物的说话风格。

## 核心思路

微信聊天记录数量有限，直接用于微调往往数据不足。本项目通过**风格迁移**扩充训练数据：

1. **真实对话**（`train_data.json`）：少量真实聊天记录，定义目标风格
2. **通用问答**（`replay_data.json`）：内容丰富但风格不符，需要改写
3. **风格改写**（`rewrite_style.py`）：LLM 参考真实对话的风格，将通用回答改写为目标风格 → `replay_data_styled.json`
4. **联合微调**：将真实对话 + 风格改写后的数据合并，在 Colab 上做 QLoRA 微调

风格改写的关键在于**保留内容、迁移语气**：LLM 从真实对话中学习口头禅、句式、语气，将结构化的通用回答改写为自然口语。

## 数据流

```
data/train_data.json          ← 真实聊天记录（少量，定义目标风格）
data/replay_data.json         ← 通用问答数据（内容丰富，风格不符）
        │
        ▼ scripts/rewrite_style.py（LLM 风格改写）
        │
data/replay_data_styled.json  ← 改写后的数据（内容不变，风格对齐）
        │
        ▼ train/finetune.ipynb（Colab QLoRA 微调）
        │
lora_adapter/                 ← LoRA 权重（约 150MB）
```

## 训练数据格式（ShareGPT）

```json
[
  {
    "conversations": [
      {"role": "user", "content": "在干嘛"},
      {"role": "assistant", "content": "刚下班回来"},
      {"role": "user", "content": "今天累不累"},
      {"role": "assistant", "content": "还好吧，开了一天会"}
    ]
  }
]
```

- `user`：你发的消息；`assistant`：对方的回复（模型学习目标）
- 每组对话必须以 `user` 开头、`assistant` 结尾

## 微调方案

| 组件 | 选型 |
|------|------|
| 基座模型 | Qwen3（默认 4B，可换 8B/14B） |
| 微调方法 | LoRA（r=16, alpha=32，覆盖注意力层+FFN） |
| 量化 | QLoRA 4-bit NF4，免费 Colab T4 可跑 |
| 训练框架 | TRL SFTTrainer + PEFT |

**关键超参**：`epochs=2`，`lr=1e-4`，`max_seq_length=512`，`batch_size=1 × grad_accum=16`

## 推理

```bash
# LoRA 适配器（推荐）
python inference/chat.py --model Qwen/Qwen3-4B --lora ./lora_adapter

# 合并后的完整模型
python inference/chat.py --model ./merged_model

# llama.cpp（CPU 更快）
python inference/chat.py --backend llama.cpp --gguf ./model.gguf
```

## 项目结构

```
roleplay/
├── data/
│   ├── train_data.json          # 真实聊天记录（定义目标风格）
│   ├── replay_data.json         # 通用问答原始数据
│   └── replay_data_styled.json  # 风格改写后的数据（由脚本生成）
├── scripts/
│   └── rewrite_style.py         # 风格改写脚本（调用 LLM API）
├── train/
│   └── finetune.ipynb           # Colab 微调 notebook
├── inference/
│   └── chat.py                  # 本地推理脚本
└── .env                         # API 配置（OPENAI_API_KEY 等）
```

## 参考

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [Qwen3 技术报告](https://qwenlm.github.io/blog/qwen3/)
