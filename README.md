# LMFS Assignment 1

## 安装依赖

需要 Python 3.13 和 uv。在项目根目录执行：

```bash
uv sync
```

## 命令行训练、验证及 W&B 记录

训练和验证语料均为 UTF-8 文本文件。首次使用 W&B 可执行 `uv run wandb login`。
在项目根目录执行：

```bash
uv run python -m src.train_bpe \
  --train-file data/owt_train.txt \
  --valid-file data/owt_valid.txt \
  --vocab-size 32000 \
  --special-token '<|endoftext|>' \
  --wandb-project lmfs-assignment1
```

训练完成后保存到 `outputs/<run-id>/tokenizer.pkl`。一个 pickle 文件包含
`vocab`、`merges` 和完整的 `special_tokens` 清单，直接保留整数 ID、bytes
及合并顺序，不保存查找索引或编码缓存。

入口重新加载模型后读取完整验证文本，再计时全文 encode、全文 decode 和逐字符比较。
读取文件、加载模型、前 200 个 token 表格及 W&B 记录均在计时区间之外。
W&B Table 保存前 200 个 token 的 ID、hex、bytes、解码结果、UTF-8 合法性、
字节位置及原文匹配情况；Summary 保存 `bpe/valid_seconds`、
`bpe/valid_token_count`、`bpe/roundtrip_passed` 和 `bpe/vocab_size`。
源码保存为 `code` 类型的 artifact，完整特殊 token 清单记录在 config 中。

| 参数 | 用途 | 默认值 |
| --- | --- | --- |
| `--train-file` | 训练语料路径 | 课程数据目录存在时使用该目录，否则使用本项目 `data/owt_train.txt` |
| `--valid-file` | 完整验证语料路径 | 同上，文件名为 `owt_valid.txt` |
| `--vocab-size` | 目标词表大小 | `32000` |
| `--special-token` | 保留的特殊 token；可重复传入以指定多个 | `<\|endoftext\|>` |
| `--load-tokenizer` | 加载已有模型并跳过训练 | 不指定 |
| `--output-file` | 保存完整模型的 pickle 文件路径 | `outputs/<run-id>/tokenizer.pkl` |
| `--wandb-project` | W&B 项目名 | `lmfs-assignment1` |
| `--wandb-entity` | W&B 用户或团队 | 使用 W&B 默认设置 |
| `--wandb-mode` | `online`、`offline` 或 `disabled` | 使用 W&B 默认设置 |

已有 `tokenizer.pkl` 可以直接进行正式全文验证并记录 W&B，无需重新训练：

```bash
uv run python -m src.train_bpe \
  --load-tokenizer tokenizer.pkl \
  --valid-file data/owt_valid.txt \
  --wandb-project lmfs-assignment1
```

加载已有模型时，config 使用模型中保存的特殊 token 清单；若另外传入
`--special-token`，其清单必须与模型一致。每次调用都会重新加载模型再计时。
仅做本地验证可加 `--wandb-mode disabled`；这种模式不会上传评分记录。
`offline` 模式保留本地 W&B 数据，但需要后续同步才能在网页查看。

指定多个特殊 token：

```bash
uv run python -m src.train_bpe \
  --train-file data/owt_train.txt \
  --valid-file data/owt_valid.txt \
  --special-token '<|endoftext|>' \
  --special-token '<|padding|>'
```

此示例中的 padding 标记用于显式填充位置；加入清单仅保留其独立 ID，
不会自动插入标记。请传入本人实际使用的完整清单，任务二沿用同一配置。

查看全部参数：

```bash
uv run python -m src.train_bpe --help
```

## 编码与解码

在项目根目录加载训练结果并使用 tokenizer：

```python
from src import bpe

tokenizer = bpe.BPE()
tokenizer.load("tokenizer.pkl")

text = "你好，BPE！<|endoftext|>"
ids = tokenizer.encode(text)
assert tokenizer.decode(ids) == text

with open("data/corpus.txt", encoding="utf-8", newline="") as stream:
    for token_id in tokenizer.encode_iterable(stream):
        print(token_id)
```

`encode_iterable` 将输入视为同一文本的连续片段，缓存尚未完整的预词、
特殊 token 和空白边界，输出与 `encode("".join(chunks))` 完全一致。
空片段不会结束流；不能完成的长预词需要保留到其边界确定。
`decode` 保留特殊 token，先拼接 token 字节再解码 UTF-8；
非法字节序列替换为 `�`，未知 ID 抛出 `ValueError`。

`save("outputs/tokenizer.pkl")` 和 `load("outputs/tokenizer.pkl")` 使用同一个
文件路径；文件名没有扩展名时也按 pickle 保存和加载，不自动添加后缀。
任务二通过 `load()` 加载该文件，恢复完整的特殊 token 配置。
课程接口 `BPE.from_files("vocab.pkl", "merges.pkl", special_tokens)` 用于
读取分别 pickle 保存的 `dict[int, bytes]` 词表和有序 merge 列表，特殊 token
清单由调用方传入。日常保存完整模型使用 `save()`，恢复使用 `load()`。

## 项目结构与职责

```text
assignment1/
├── src/
│   ├── __init__.py
│   ├── bpe.py                  # BPE 类、模型索引、缓存及 CS336 适配函数
│   ├── evaluation.py           # 全文计时、还原检查、token 明细
│   ├── wandb_record.py         # W&B 生命周期及源码记录
│   ├── train_bpe.py            # CLI 配置及训练、保存、重载、验证流程
│   ├── training/
│   │   ├── __init__.py
│   │   ├── words.py            # TrainingWords：训练语料与字节序列
│   │   └── pairs.py            # PairState、HeapEntry、频次与候选堆
│   └── utils/
│       ├── __init__.py
│       ├── pretokenization.py  # 共享 PreTokenizer：正则、特殊 token 和流式边界
│       ├── indexes.py          # 特殊 token 词表补齐与编码索引构建
│       ├── serializer.py       # pickle 文件读写
│       └── encoding.py         # 预词编码、按排名 merge 和缓存操作
├── tests/                      # 训练、编码、持久化及验证测试
├── data/                       # 训练与验证数据，不提交版本库
└── outputs/                    # 实验输出及模型文件，不提交版本库
```

实现与运行入口统一放在 `src/`，根目录不保留同名转发文件。
在项目根目录用 `uv run python -m src.train_bpe` 运行，
用 `from src.bpe import BPE` 导入模型；内部代码统一使用 `src.*` 完整包路径导入。
课程适配函数 `run_train_bpe()` 和 `get_tokenizer()` 位于 `src/bpe.py`。

`BPE` 管理词表、查找索引、编码缓存及训练和编码主流程。训练专用的数据结构
放在 `training/`，其中两个较小的 pair 状态类与候选操作共同位于 `pairs.py`。
预分词和编码逻辑放在 `utils/`，不依赖 `BPE` 实例。`PreTokenizer` 统一管理
课程给定的 `PAT`、特殊 token 校验及匹配，提供完整文本、连续片段和文件流入口。
训练通过 `iter_stream()` 获取普通预分词，编码通过 `iter_text()` 或
`iter_chunks()` 获取带特殊 token 标记的预分词；三者共享相同的匹配规则。
`utils/indexes.py` 根据词表、merges 和特殊 token 构建查找表，并补齐缺失的
特殊 token；`BPE` 在初始化、训练和加载后接收这些查找表并清空旧缓存。
`BPE` 也负责 `save()`、`load()` 和
`from_files()` 的模型持久化流程，包括文件路径、配置恢复和状态更新；
`utils/serializer.py` 只提供 pickle 文件读写。

`train_bpe.py` 调用模型、验证及 W&B 记录模块，模型、训练模块和工具函数不依赖
W&B 或运行入口；序列化仅保存模型数据，不保存训练中间状态或编码缓存。
源码记录递归收集整个 `src/`，包含运行入口和各子包的实现。

验证实现：

```bash
uv run python -m pytest -q
```
