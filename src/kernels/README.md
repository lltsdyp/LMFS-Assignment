# kernels：手写前向算子库及模型使用说明

`src.kernels` 负责模型前向算子、基础层、RoPE 和 MHA。模型组装位于
`src/models/`，训练工具位于 `src/training/lm/`，公共辅助计算位于 `src/utils/`。
用 PyTorch 基础张量运算手写组件公式，CPU/CUDA 共用一套实现，保留自动微分。
基础运算的设备分发由 PyTorch 负责；CUDA 张量的运算执行 PyTorch 的 CUDA
内核。正式语料采样、完整训练/验证入口、W&B 曲线和 6 小时预算管理位于
`src/train_lm.py`；使用方式见 [语言模型训练说明](../training/lm/README.md)。

## 已实现的内容

| 文件 | 内容 |
| --- | --- |
| `functional.py` | 模型前向公式：Linear、Embedding、RMSNorm、SiLU、softmax、RoPE、注意力 |
| `layers.py` | `Linear`、`Embedding`、`RMSNorm`、`SiLU`、`SwiGLU` |
| `attention.py` | `RotaryPositionalEmbedding`、带 RoPE 的因果 `MultiHeadSelfAttention` |
| `__init__.py` | 常用类和函数的统一导入入口 |

相关模块位于算子库外：

| 文件 | 内容 |
| --- | --- |
| `../models/transformer.py` | `TransformerBlock`、`TransformerLM`、`TransformerConfig` |
| `../training/lm/optim.py` | 手写 AdamW、warmup + cosine 学习率、全局梯度裁剪 |
| `../utils/losses.py` | 手写稳定交叉熵，支持逐 token NLL 和总和 |
| `../utils/tensors.py` | 共用 dtype 和设备校验 |

BPE 训练的数据结构放在 `src/training/bpe/`，与语言模型训练分开。

调用关系：

```text
TransformerLM / Linear / RMSNorm 等上层对象
                          ↓
                    functional.py
                          ↓
                PyTorch 基础张量运算与设备分发
                     ↙          ↘
               PyTorch CPU   PyTorch CUDA 内核
```

例如 `Linear.forward()` 调用 `functional.linear()`，执行 `x @ weight.T`。
PyTorch 根据张量设备选择矩阵乘法的底层实现。RoPE 模块管理位置和缓存，调用
其中的旋转公式。`AdamW` 的参数组、step、动量状态和更新公式位于
`src/training/lm/optim.py`。
模型参数名和 checkpoint 结构保持兼容。

交叉熵用 `from src.utils.losses import cross_entropy` 导入，模型用
`from src.models import TransformerLM` 导入。PPL 的验证集遍历和 NLL 累计属于
`src/training/lm/validation.py`，`functional.py` 只包含模型前向算子。

所有输入，包括权重、目标、mask 和显式位置，都必须在同一设备；不同 CUDA
编号也视为不同设备。当前测试覆盖 CPU，并提供可选 CUDA 测试；其他设备的
算子支持由 PyTorch 决定，尚未验证。

迁移模型和输入即可选择 CUDA：

```python
model = model.to("cuda")
x = x.to("cuda")
targets = targets.to("cuda")
logits = model(x)
```

应在迁移模型之后创建优化器；GPU 验证需在有可用 CUDA 设备的环境运行。
公式是基础张量运算组合。梯度裁剪将阈值选择留在张量设备上；CUDA 上的非有限
检查仍需要一次主机同步，以保留异常处理语义。

所有前向和参数更新规则自行计算。仅使用 `nn.Module`、`nn.Parameter`、
`nn.ModuleList`、`torch.optim.Optimizer` 基类及允许的初始化函数；没有调用
`torch.nn.functional`，也没有调用现成层、激活、softmax、交叉熵、优化器、
调度器或梯度裁剪。

### 模型结构与初始化

完整模型：token embedding → 多个 Pre-norm block → 最终 RMSNorm → 词表投影。

每个 block 执行：

```text
x = x + Attention(RMSNorm(x))
x = x + SwiGLU(RMSNorm(x))
```

- 所有 Linear 无 bias，权重形状为 `(out_features, in_features)`。
- Linear 使用标准差 `sqrt(2 / (in_features + out_features))` 的截断正态，
  截断范围为正负 3 倍标准差。
- Embedding 使用标准差为 1、截断范围为 `[-3, 3]` 的正态分布。
- RMSNorm 的 gain 初始化为 1，默认 `eps=1e-5`。
- SwiGLU 为 `w2(SiLU(w1(x)) * w3(x))`。未显式传入 `d_ff` 时，选择最接近
  `8/3 * d_model` 的正的 64 倍数；`d_model=512` 时得到 `d_ff=1344`。
- 使用标准多头自注意力，Q/K/V 头数相同，每头维度必须为偶数。
- RoPE 旋转相邻特征对，只作用于 Q/K，默认 `theta=10000`。
- Embedding 与输出权重独立；模型从随机权重开始。
- 模型始终输出所有位置的 logits，不在前向中计算 loss 或插入特殊 token。

## 使用基础算子和层

在项目根目录运行或导入，内部使用 `src.*` 完整包路径。

```python
import torch
from src.kernels import Linear, RMSNorm, SiLU, SwiGLU
from src.kernels import functional as F

x = torch.randn(2, 8, 32)
projection = Linear(32, 64)
norm = RMSNorm(64)
activation = SiLU()
ffn = SwiGLU(64, d_ff=128)
y = ffn(activation(norm(projection(x))))  # (2, 8, 64)

scores = torch.randn(2, 4, 8, 8)
probabilities = F.softmax(scores, dim=-1)
```

模块支持 `device`、`dtype` 和 PyTorch 的 `parameters()`、`state_dict()`、
`train()`、`eval()`、`.to()`。RMSNorm、SiLU、softmax、注意力分数和交叉熵对
float16/bfloat16 输入使用 float32 中间计算；float64 保留双精度。
RoPE 缓存在转换模型 dtype 后仍保留至少 float32 精度，随模型迁移设备，
恢复模型时由配置重新生成，不写入 checkpoint。

## 使用任务一 BPE 创建模型

```python
import torch
from src.bpe import BPE
from src.models import TransformerLM
from src.utils.losses import cross_entropy

bpe = BPE()
bpe.load("tokenizer.pkl")  # 使用任务一实际输出的完整文件路径
if sorted(bpe.vocab) != list(range(len(bpe.vocab))):
    raise ValueError("词表 ID 必须从 0 连续编号")

model = TransformerLM(
    vocab_size=len(bpe.vocab),
    context_length=256,
    d_model=512,
    num_layers=4,
    num_heads=16,
    d_ff=1344,
)

# 此处仅演示接口；正式数据采样由 src.training.lm.data 负责。
# 不额外插入 BOS/EOS，已有特殊 token 由任务一编码逻辑处理。
ids = bpe.encode("Hello, Transformer! This is a training example.")
if len(ids) < 2:
    raise ValueError("示例文本至少需要两个 token")
tokens = torch.tensor(ids[:257], dtype=torch.long).unsqueeze(0)
x, targets = tokens[:, :-1], tokens[:, 1:]
logits = model(x)  # (1, sequence_length, len(bpe.vocab))
loss = cross_entropy(logits, targets)
loss.backward()
```

训练集与验证集必须调用同一个已加载 BPE 的编码方法。该库不会修改词表、merges
或特殊 token 配置。NumPy `uint16` 数据送入模型时需转换为 `torch.long`；
交叉熵的目标也必须是 `torch.long`。

## 单次参数更新与学习率

```python
from src.training.lm.optim import AdamW, clip_grad_norm_, cosine_learning_rate

optimizer = AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=0.1,
)

step = 1  # 调用方维护参数更新次数
lr = cosine_learning_rate(
    step=step,
    max_lr=3e-4,
    min_lr=3e-5,
    warmup_steps=100,
    cosine_steps=10000,
)
for group in optimizer.param_groups:
    group["lr"] = lr

optimizer.zero_grad(set_to_none=True)
loss = cross_entropy(model(x), targets)
loss.backward()
original_grad_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

`cosine_steps` 是 cosine 衰减结束的**绝对 step**，包含 warmup 区间，不是 warmup
之后再运行的步数。warmup 大于零时 step 0 的学习率为零，warmup 结束时达到
`max_lr`，cosine 结束时达到 `min_lr`；后续保持 `min_lr`。

AdamW 严格使用 CS336 讲义中的规则：

```text
p *= 1 - lr * weight_decay
m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * grad**2
adjusted_lr = lr * sqrt(1 - beta2**t) / (1 - beta1**t)
p -= adjusted_lr * m / (sqrt(v) + eps)
```

`t` 对每个有梯度的参数从 1 开始。这里 `eps` 加在未修正的 `sqrt(v)` 上，与
PyTorch AdamW 的 epsilon 约定略有不同；数值核对应使用讲义公式。支持参数组、
closure、`zero_grad()` 和状态保存恢复。float16/bfloat16 参数的动量和二阶矩
保存在 float32 中；不会自动创建模型参数的 float32 主副本，也不会自动开启 AMP。

梯度裁剪返回裁剪前全局 L2 范数，只在超过阈值时缩放，默认 `eps=1e-6`。
没有梯度的参数被跳过；非有限梯度会在修改前报错。优化器和裁剪仅支持稠密实数梯度。

## 交叉熵、掩码与验证

`cross_entropy(logits, targets, reduction="mean", ignore_index=-100)` 的词表维度
始终在最后，目标形状必须与 logits 的前导维度相同。支持 `none`、`sum`、`mean`：

- `none`：返回每个目标位置的 NLL，被忽略位置返回零。
- `sum`：返回有效目标 NLL 总和。
- `mean`：除以有效目标数；全被忽略或空批次返回可微的零。

验证 PPL 的汇总方式：

```python
import math

model.eval()
total_nll = 0.0
total_targets = 0
with torch.no_grad():
    # valid_batches 由 src.training.lm.data 提供，遍历真实目标。
    for valid_x, valid_targets in valid_batches:
        total_nll += cross_entropy(
            model(valid_x), valid_targets, reduction="sum"
        ).item()
        total_targets += (valid_targets != -100).sum().item()

if total_targets == 0:
    raise ValueError("验证集没有有效目标 token")
ppl = math.exp(total_nll / total_targets)
model.train()
```

不能平均 batch 的 PPL，也不能用生成样例替代验证文本。这里展示汇总接口，
完整验证集的分块和目标覆盖规则见 `src/training/lm/data.py` 和训练说明。

`scaled_dot_product_attention` 接收 `(..., query_length, head_dim)` 的 Q 和
`(..., key_length, head_dim/value_dim)` 的 K/V。`mask` 可以广播到注意力分数：
布尔值 `True` 表示**允许关注**；浮点 mask 会加到分数上。`is_causal=True`
自动遮掉未来位置。全被遮掉的行输出零。`MultiHeadSelfAttention` 默认始终因果，
不需要传 mask。

`RotaryPositionalEmbedding` 接收 `(..., seq_len, d_k)`，可传一维位置，或每个
batch 各自的位置；缺少的 head 维度自动广播。默认从位置零开始，传入的位置
必须落在初始化指定的缓存范围内。

## 模型和优化器保存恢复

```python
from dataclasses import asdict

torch.save(
    {
        "model_config": asdict(model.config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "tokenizer_path": "tokenizer.pkl",
    },
    "checkpoint.pt",
)

saved = torch.load("checkpoint.pt", map_location="cpu", weights_only=True)
restored_model = TransformerLM(**saved["model_config"])
restored_model.load_state_dict(saved["model"])
restored_optimizer = AdamW(restored_model.parameters())
restored_optimizer.load_state_dict(saved["optimizer"])
restored_step = saved["step"]
```

配置记录模型结构，不包含设备或 dtype；恢复低精度模型时应显式指定相同 dtype。
GPU 恢复时先把模型移动到目标设备，再创建优化器和加载其状态。续训还需沿用
同一 tokenizer；`src/train_lm.py` 会记录并核对 BPE 配置、原文和训练配置。

## 测试

```bash
uv run python -m pytest -q tests/test_kernels_functional.py tests/test_kernels_transformer.py tests/test_kernels_optim.py tests/test_kernels_devices.py
uv run python -m pytest -q
```

测试包含独立 NumPy 数值参考、有限差分梯度检查、半精度稳定性、完整模型的
NumPy 前向对照、因果性、小模型学习、模型和优化器恢复后的更新一致性，以及
生产源码中禁止 API 的递归检查。还包含设备一致性检查，以及同一套公式的 CPU/CUDA
数值和梯度对照测试；没有 CUDA 设备时 GPU 测试自动跳过。测试不依赖预训练模型
或现成神经网络组件。
