# LMFS Assignment 1

## 手写 Transformer 组件库

任务二第一部分按职责分布在 `src/kernels/`（前向算子）、`src/models/`（模型）、
`src/training/lm/`（优化工具）和 `src/utils/`（公共辅助计算）。已实现基础层、
RoPE、因果多头注意力、Pre-norm Transformer 语言模型、交叉熵、AdamW、
warmup + cosine 学习率和全局梯度裁剪。所有核心公式使用 PyTorch 基础张量运算自行实现，
CPU/CUDA 共用一套公式，基础运算的设备分发由 PyTorch 负责。

按职责导入：

```python
from src.models import TransformerLM
from src.training.lm.optim import AdamW, clip_grad_norm_, cosine_learning_rate
from src.utils.losses import cross_entropy
```

`src/training/bpe/` 放分词器训练结构，`src/training/lm/` 放语言模型训练工具。
`kernels/functional.py` 只包含模型前向算子。
完整组件列表、BPE 接入、参数更新、PPL 汇总和保存恢复示例见
[kernels 使用文档](src/kernels/README.md)。语言模型训练入口已实现，见
[训练、验证和续训说明](src/training/lm/README.md)。

## 下载 TinyStories

在项目根目录执行，下载 TinyStories V2 GPT-4 训练集和验证集到 `data/`：

```bash
bash download_tinystories.sh
```

使用 `wget -c`，中断后可再次运行以继续下载。

## 语言模型训练

默认模型：context length 256、d_model 512、d_ff 1344、RoPE θ 10000、4 层、16 头。
词表大小使用任务一 BPE 的实际词表大小，模型从随机权重初始化。
在 4060 上先检查训练流程：

```bash
uv run python -m src.train_lm \
  --tokenizer tokenizer.pkl \
  --train-file data/owt_train.txt \
  --valid-file data/owt_valid.txt \
  --device cuda --batch-size 1 --max-steps 300 \
  --output-dir outputs/lm-4060
```

`--tokenizer` 替换成任务一保存的实际路径。脚本使用自己的 BPE 流式编码并缓存，
每 100 次更新记录训练 PPL，最后保存模型并对完整验证文本计算 PPL。
启动时自动打印和保存源码，默认关闭 W&B；`--wandb-mode online` 开启源码和标量记录。
默认累计预算 6 小时，包含初始化、数据准备、训练、保存和最终验证。
缓存准备、A100 运行、预算预留和 checkpoint 续训命令见
[训练说明](src/training/lm/README.md)。

## 安装依赖

需要 Python 3.13、uv 和 Rust 工具链（Cargo/rustc）。在项目根目录执行：

```bash
uv sync
```

`uv sync` 会构建并安装 `rust/` 中的本地 PyO3 扩展 `_lmfs_bpe`。
修改 Rust 源码后执行 `uv sync --reinstall-package lmfs-bpe-native` 重新构建。

`BPE.encode()`、`BPE.encode_iterable()` 和 `BPE.encode_blocks()` 将完整文本块
交给 Rust。Rust 释放 GIL，执行 Unicode 预分词、特殊 token 匹配、跨块尾部处理
及 Rayon 并行 BPE 合并；不足 1024 个 pre-token 时串行合并。
预词使用 UTF-8 缓冲区中的区间，不逐词创建 Python bytes 或 tuple。
两边的预分词 Unicode 数据固定为 **17.0.0**：Python 使用
`regex==2026.9.10`，Rust 使用 `unicode-properties==0.1.4` 的字符分类。
直接使用依赖自带的 Unicode 数据，无需生成或传递分类表。
Rust `regex` 固定为 `1.13.1`，仅匹配特殊 token 字面量，自带 Unicode
分类表已禁用。`train()` 的 Python 预分词和训练流程继续作为参考实现。
可在启动前设置 `RAYON_NUM_THREADS=8` 指定线程数。Rust 模型索引在初始化、
训练和加载后重建。工作线程复用合并缓冲区，词缓存保留最近两代；总条目预算
为 65536，键和 ID 数据的总预算为 32 MiB（哈希表和分配器开销另计）。
语言模型每次启动都用 `encode_blocks()` 重新编码语料，打包 ID 保存在内存中，
不生成磁盘 token 缓存。NumPy 直接借用打包缓冲区，无逐 token 的 Python 往返。
日志和配置记录编码耗时；恢复训练时核对语料、词表和编码结果的身份。

更新 Unicode 版本时，同时选择 Unicode 数据版本一致的 Python `regex`
和 Rust `unicode-properties` 版本，更新依赖锁文件并重编译。
一致性测试覆盖 Python 字符分类的全部区间边界及流式编码。

## BPE 命令行训练、验证及 W&B 记录

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
每次输出当前输入块中已确定边界的预词，不会把整个输入流读进内存。
`encode()` 在 Rust 内按 UTF-8 边界分块，限制临时预词和 ID 数组的大小。
准备训练数据时使用块接口，避免逐 token 迭代：

```python
import numpy as np

packed = bytearray()
with open("data/corpus.txt", encoding="utf-8", newline="") as stream:
    chunks = iter(lambda: stream.read(65536), "")
    for block in tokenizer.encode_blocks(chunks, dtype="uint16"):
        packed.extend(block)
ids_array = np.frombuffer(packed, dtype="<u2")
```

`encode_blocks()` 返回拥有独立生命周期的不可变 `bytes`，内容为小端
`uint16` 或 `uint32`。所有词表 ID 必须能由指定类型表示，否则抛出
`ValueError`，不截断或溢出。Rust 到 Python 有一次按块复制，
`numpy.frombuffer(block, dtype="<u2")` 可直接创建只读视图，无额外复制。
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
│   ├── train_lm.py             # LM CLI、训练循环、记录及预算管理
│   ├── kernels/                # 手写前向算子、基础层、RoPE 和 MHA
│   ├── models/
│   │   ├── __init__.py
│   │   └── transformer.py      # Transformer 配置、block 和完整语言模型
│   ├── training/
│   │   ├── __init__.py
│   │   ├── bpe/
│   │   │   ├── __init__.py
│   │   │   ├── words.py        # TrainingWords：BPE 训练语料与字节序列
│   │   │   └── pairs.py        # PairState、HeapEntry、频次与候选堆
│   │   └── lm/
│   │       ├── __init__.py
│   │       ├── optim.py        # AdamW、学习率和全局梯度裁剪
│   │       ├── data.py         # 内存 BPE token 数组、采样和验证窗口
│   │       ├── checkpoint.py   # 模型、优化器、随机状态与计时保存恢复
│   │       └── validation.py   # 全量真实文本验证与 PPL 汇总
│   └── utils/
│       ├── __init__.py
│       ├── pretokenization.py  # 共享 PreTokenizer：正则、特殊 token 和流式边界
│       ├── indexes.py          # 特殊 token 词表补齐与编码索引构建
│       ├── serializer.py       # pickle 文件读写
│       ├── losses.py           # 手写稳定交叉熵和 NLL 计算
│       ├── metrics.py          # 总 NLL / 目标数换算 PPL
│       ├── source.py           # 启动时打印并保存源码快照
│       └── tensors.py          # 张量累积精度和设备一致性检查
├── rust/                       # 独立构建的 Rust 编码扩展
│   ├── pyproject.toml          # maturin 构建配置
│   ├── Cargo.toml
│   ├── Cargo.lock
│   └── src/
│       ├── lib.rs              # Python 绑定、GIL 释放与异常转换
│       ├── bpe.rs              # 模型索引、缓存和有序并行批次
│       ├── pretokenization.rs  # Unicode 区间、特殊 token 和流式尾部
│       ├── unicode.rs         # 固定 Unicode 17.0.0 分类表
│       └── encoding.rs         # 按 merge rank 合并字节片段
├── tests/                      # 训练、编码、持久化及验证测试
├── data/                       # 训练与验证数据，不提交版本库
└── outputs/                    # 实验输出及模型文件，不提交版本库
```

Python 实现与运行入口放在 `src/`，Rust 编码内核放在 `rust/src/`。
在项目根目录用 `uv run python -m src.train_bpe` 运行，
用 `from src.bpe import BPE` 导入模型；内部代码统一使用 `src.*` 完整包路径导入。
课程适配函数 `run_train_bpe()` 和 `get_tokenizer()` 位于 `src/bpe.py`。

`BPE` 管理词表、查找索引、编码缓存及训练和编码主流程。训练专用的数据结构
放在 `training/bpe/`，其中两个较小的 pair 状态类与候选操作共同位于 `pairs.py`。
语言模型训练工具单独放在 `training/lm/`。BPE 训练不导入语言模型或 PyTorch，
语言模型训练沿用已训练的 BPE 编码，不依赖 BPE 的词频表或 merge 候选堆。
Python 预分词参考实现在 `utils/`，原生预分词和编码内核在 `rust/src/`。
Python `PreTokenizer` 管理
课程给定的 `PAT`、特殊 token 校验及匹配，提供完整文本、连续片段和文件流入口。
训练通过 `iter_stream()` 获取普通预分词，`iter_text()` 和 `iter_chunks()`
用于验证 Rust 预分词和流式边界一致性。Rust 使用同一组 Unicode 分类规则。
`utils/indexes.py` 根据词表、merges 和特殊 token 构建查找表，并补齐缺失的
特殊 token；`BPE` 在初始化、训练和加载后接收这些查找表、清空旧缓存，
并重建 Rust 编码器。
`BPE` 也负责 `save()`、`load()` 和
`from_files()` 的模型持久化流程，包括文件路径、配置恢复和状态更新；
`utils/serializer.py` 只提供 pickle 文件读写。

`train_bpe.py` 调用模型、验证及 W&B 记录模块，模型、训练模块和工具函数不依赖
W&B 或运行入口；序列化仅保存模型数据，不保存训练中间状态或编码缓存。
源码记录包含 `src/` 的 Python 实现、`rust/src/` 的 Rust 实现及构建配置。

验证实现：

```bash
uv run python -m pytest -q
```
