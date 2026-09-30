# LMFS Assignment 1

## 安装依赖

需要 Python 3.13 和 uv。在项目根目录执行：

```bash
uv sync
```

## 命令行训练 BPE

训练语料为 UTF-8 文本文件。在项目根目录执行：

```bash
uv run python src/train_bpe.py \
  --train-file /path/to/corpus.txt \
  --vocab-size 32000 \
  --special-token '<|endoftext|>'
```

训练完成后自动保存到当前目录的 `tokenizer.pkl`，并输出保存路径、训练时间、词表大小、merge 数量和前 10 个 merge。同名文件会被覆盖。

| 参数 | 用途 | 默认值 |
| --- | --- | --- |
| `--train-file` | 训练语料路径 | `/data/nlp_course/1-basic-data/owt_train.txt` |
| `--vocab-size` | 目标词表大小 | `32000` |
| `--special-token` | 保留的特殊 token；可重复传入以指定多个 | `<\|endoftext\|>` |

指定多个特殊 token：

```bash
uv run python src/train_bpe.py \
  --train-file /path/to/corpus.txt \
  --special-token '<|endoftext|>' \
  --special-token '<|padding|>'
```

查看全部参数：

```bash
uv run python src/train_bpe.py --help
```
