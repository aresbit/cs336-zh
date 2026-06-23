layout: default
title: PyTorch 与资源核算
nav_order: 2
permalink: /docs/ch02
nav:
  prev: ch01
  next: ch03
---

## 第02章 PyTorch 与资源核算

现代大语言模型（LLM）的训练与推理涉及大规模的张量运算。要高效地设计与训练模型，我们必须理解三个层次的问题：代码表达（如何清晰地描述张量操作）、计算代价（模型消耗了多少 FLOPs）以及硬件瓶颈（GPU 如何制约实际运行速度）。本章以 einops 为起点建立清晰的张量操作思维，随后系统性地展开 FLOPs 计算、GPU 内存模型、算术强度与 Roofline 分析，最后将这套方法论应用于 Transformer 各组件的资源核算。

### 2.1 einops：比原生形状操作更清晰的张量重排

PyTorch 提供了 `reshape`、`permute`、`transpose`、`unsqueeze`、`squeeze` 等丰富的形状操作原语。然而在实际工程中，仅仅用这些原语组合出来的代码往往难以阅读——读代码的人需要在脑海中追踪每个维度的大小变化，才能理解"这个操作到底在做什么"。einops 库通过维度命名的方式解决了这个问题。

**核心操作一：rearrange**

`rearrange` 将输入张量按命名维度重新排列。以多头注意力中的头拆分（head split）为例：

```python
from einops import rearrange

# x: (batch, seq_len, d_model)
# 拆分为 (batch, num_heads, seq_len, head_dim)
x = rearrange(x, 'b s (h d) -> b h s d', h=num_heads)
```

这里的 `(h d)` 表示将 `d_model` 这个维度切分为 `num_heads * head_dim`。相比之下，原生写法需要先 reshape 再 permute：

```python
# 等价的 PyTorch 原生写法——意图被操作顺序掩盖
x = x.reshape(batch, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
```

rearrange 的优势在于：维度名称出现在目标模式中，读者一眼就能看到输出张量的语义结构。

**核心操作二：repeat**

`repeat` 用于沿指定维度复制张量。典型场景是 MQA（Multi-Query Attention）中将 KV 头广播到与 Q 头数量对齐：

```python
# kv: (batch, seq_len, num_kv_heads, head_dim)
# 复制为与 num_heads 个查询头对齐
kv = repeat(kv, 'b s h d -> b s (h copy) d', copy=num_heads // num_kv_heads)
```

**核心操作三：reduce**

`reduce` 在命名维度上进行规约操作，例如对注意力权重进行 softmax 之前需要指定规约轴：

```python
from einops import reduce

# attn_scores: (batch, heads, seq_len, seq_len)
# softmax 在最后一个维度上做，einops 的表达方式是：
# （实际上 softmax 通常用 PyTorch 原生函数，reduce 更多用于 mean/sum/max）
avg_head = reduce(x, 'b h s d -> b s d', 'mean')
```

**为什么 einops 优于原生写法**：
1. 维度命名消除了"魔法数字"——`'b s (h d) -> b h s d'` 告诉你这里有一个头拆分，而 `reshape(...).permute(0, 2, 1, 3)` 只告诉你维度大小如何变化。
2. 错误信息友好——einops 在遇到维度不匹配时会报告具体的维度名称和期望大小。
3. 代码意图自文档化——维度名称本身即是注释。

### 2.2 矩阵乘法的 FLOPs 计算

FLOPs（Floating Point Operations，浮点运算次数）是衡量模型计算量的基本单位。正确计算 FLOPs 是进行资源估算和优化决策的前提。

**矩阵乘法的通用公式**

对于矩阵乘法 $$C = A \times B$$，其中 $$A \in \mathbb{R}^{m \times k}$$，$$B \in \mathbb{R}^{k \times n}$$，结果 $$C \in \mathbb{R}^{m \times n}$$：

- 输出矩阵 $$C$$ 有 $$m \times n$$ 个元素。
- 每个元素的计算需要 $$k$$ 次乘法与 $$k-1$$ 次加法，共计 $$2k-1$$ 次浮点运算。
- 因此总 FLOPs = $$m \times n \times (2k-1) \approx 2mnk$$。

工程实践中，我们通常使用近似 $$2mnk$$（相当精确，因为 $$k$$ 在 Transformer 中通常很大）。

**线性层（Linear）**

Transformer 中一次全连接变换 $$Y = XW^T$$，其中 $$X \in \mathbb{R}^{b \times n \times d}$$，$$W \in \mathbb{R}^{d_{\text{out}} \times d}$$：

$$ \text{FLOPs} = 2 \cdot b \cdot n \cdot d \cdot d_{\text{out}} $$

**前馈网络（FFN）**

标准 FFN 包含两次全连接与一次激活：

$$ \text{FFN}(x) = \sigma(xW_1)W_2 $$

其中 $$W_1 \in \mathbb{R}^{d_{\text{ff}} \times d}$$，$$W_2 \in \mathbb{R}^{d \times d_{\text{ff}}}$$。FLOPs 为：

$$ \text{FLOPs}_{\text{FFN}} = 2 \cdot b \cdot n \cdot d \cdot d_{\text{ff}} + 2 \cdot b \cdot n \cdot d_{\text{ff}} \cdot d = 4 \cdot b \cdot n \cdot d \cdot d_{\text{ff}} $$

当使用 SwiGLU 等门控变体时，额外的门控投影增加了一组权重矩阵。回顾 Lecture 3 的分析，$$\text{FF}_{\text{SwiGLU}}(x) = (\text{Swish}(xW_g) \otimes xW_1)W_2$$，其中 $$W_g, W_1 \in \mathbb{R}^{d_{\text{ff}} \times d}$$。此时总 FLOPs 为：

$$ \text{FLOPs}_{\text{FFN-gated}} = 2 \cdot b \cdot n \cdot d \cdot d_{\text{ff}} \cdot 3 $$

这也是为什么门控 FFN 通常将 $$d_{\text{ff}}$$ 缩小到约 $$8/3 \cdot d$$（非门控的 $$4d$$ 对应 $$8/3 \cdot d$$），以保持总计算量大致不变。

### 2.3 注意力机制的 FLOPs 分析

自注意力机制的计算可以分解为四个阶段。设 $$d$$ 为模型维度，$$b$$ 为批次大小，$$n$$ 为序列长度，$$h$$ 为头数，$$k = d/h$$ 为头维度（$$k \ll d$$，通常 $$k < n$$）。

**阶段一：QKV 投影**

$$Q = XW_Q,\ K = XW_K,\ V = XW_V$$，每个投影将 $$(b, n, d)$$ 映射到 $$(b, n, d)$$：

$$ \text{FLOPs}_{\text{QKV}} = 3 \times 2 \cdot b \cdot n \cdot d \cdot d = 6bnd^2 $$

**阶段二：注意力分数计算**

$$S = QK^T / \sqrt{k}$$，其中 $$Q, K \in \mathbb{R}^{(b \cdot h) \times n \times k}$$（多头被合并到批次中乘积后 reshape 回头形式）：

将 $$QK^T$$ 看作两个形状为 $$(bh, n, k)$$ 的张量相乘，输出形状为 $$(bh, n, n)$$：

$$ \text{FLOPs}_{\text{scores}} = 2 \cdot (bh) \cdot n \cdot n \cdot k = 2bhn^2k = 2bn^2d $$

**阶段三：Softmax 加权求和**

$$\text{Output} = \text{softmax}(S) \cdot V$$，其中 $$S \in \mathbb{R}^{bh \times n \times n}$$，$$V \in \mathbb{R}^{bh \times n \times k}$$：

$$ \text{FLOPs}_{\text{attn-apply}} = 2 \cdot (bh) \cdot n \cdot k \cdot n = 2bn^2d $$

**阶段四：输出投影**

$$Y = \text{Output} \cdot W_O$$，将 $$(b, n, d)$$ 映射回 $$(b, n, d)$$：

$$ \text{FLOPs}_{\text{out-proj}} = 2bnd^2 $$

**汇总**：总注意力 FLOPs 为

$$ \text{FLOPs}_{\text{attention}} = 6bnd^2 + 2bn^2d + 2bn^2d + 2bnd^2 = 8bnd^2 + 4bn^2d $$

其中投影部分 $$8bnd^2$$ 占比最大，除非 $$n$$ 远超 $$d$$。自注意力在 $$n$$ 足够大时表现为二次复杂度——这正是长序列推理的主要瓶颈。

### 2.4 GPU 内存模型

理解 GPU 内存层次是理解 Transformer 训练性能的关键。以 NVIDIA GPU 为例，存储单元从快到慢分为：

**SRAM（静态随机存取存储器）**

- 位置：流式多处理器（SM）内部，包括 L1 Cache 与 Shared Memory。
- 容量：非常小（例如 A100 每 SM 约 192 KB），但极快。
- 作用：GPU 运行的终极目标是让数据尽量停留在 SRAM 内完成多次运算——这正是 FlashAttention 的核心设计哲学。

**L2 Cache**

- 位置：SM 之间共享，全局。
- 容量：中等（例如 A100 为 40 MB）。
- 作用：作为 SRAM 和 HBM 之间的二级缓存，减少对 HBM 的零散访问。

**HBM（高带宽存储器）**

- 容量：大（例如 A100-80GB 有 80 GB HBM2e），但速率远低于 SRAM。
- 作用：存储模型参数、激活、优化器状态等。HBM 是主要的"状态存储器"。

**全局内存（Global Memory）**

- 即 GPU DRAM（现代 GPU 即 HBM），是 GPU 与 CPU 之间 PCIe/NVLink 传输的源/目标。

**关键数字**（以 NVIDIA A100 为例）：

| 层级 | 容量 | 带宽 |
|------|------|------|
| L1/SRAM | 192 KB / SM (共 ~20 MB) | ~19 TB/s |
| L2 Cache | 40 MB | ~4 TB/s |
| HBM2e | 40/80 GB | ~2 TB/s |
| NVLink (GPU-GPU) | — | ~600 GB/s |
| PCIe 4.0 (CPU-GPU) | — | ~32 GB/s |

数据移动的成本远超计算：A100 的理论峰值算力约 312 TFLOPS（FP16），而 HBM 带宽为 2 TB/s。这意味着每从 HBM 读取一个 2 字节（FP16）数值，GPU 可以执行约 $$312 \times 10^{12} / (2 \times 10^{12} / 2) \approx 312$$ 次浮点运算。换而言之，如果算法每读入一个数只做少量运算，GPU 将因等待数据传输而空转——这就是 **内存带宽成为瓶颈** 的根本原因。

### 2.5 算术强度与 Roofline 模型

**算术强度（Arithmetic Intensity）**

算术强度定义为 FLOPS 与数据传输量（bytes）之比：

$$ I = \frac{\text{FLOPS}}{\text{Bytes Moved}} $$

它衡量每次数据移动能"养活"多少次计算。高算术强度的算子（如大矩阵乘法）受限于 GPU 的峰值算力（**Compute-Bound**）；低算术强度的算子（如逐元素激活函数、LayerNorm）受限于内存带宽（**Memory-Bound**）。

**Roofline 模型**

Roofline 模型将算子的可实现性能 $$P$$（单位 TFLOPS）作为算术强度 $$I$$ 的函数：

$$ P = \min(\text{Peak Compute}, \ I \times \text{Peak Memory Bandwidth}) $$

用中文来说：当算术强度较低时，实际性能受限于"内存的搬运速度"（$$I \times \text{带宽}$$）；当算术强度足够高时，"屋顶"由芯片峰值算力决定。这个"屋顶"形状使得任何算法的性能上限一目了然。

**注意力机制的算术强度**

回忆 Lecture 3 中 GQA/MQA 分析中的算术强度估算。考虑 Transformer 训练阶段的自注意力——将批量数据并行处理时，总计算量为 $$bn^2d$$ 量级，总内存访问量约为 $$bnd + bhn^2 + d^2$$。算术强度大致为：

$$ I_{\text{attention}} \approx O\left(\frac{1}{k} + \frac{1}{bn}\right)^{-1} $$

其中 $$k = d/h$$ 为头维度。当 $$k$$ 较大（如 $$k=128$$）且 $$bn$$ 较大时，算术强度足够高，注意力是 Compute-Bound 的。

**生成阶段的算术强度骤降**

问题出在自回归生成阶段——每次仅处理一个 token（batch 不可并行）。此时内存访问为 $$bnd + nd^2$$（需要反复读写 KV Cache），而 FLOPs 仅为 $$bnd^2$$。算术强度变为：

$$ I_{\text{inference}} \approx O\left(\frac{n}{d} + \frac{1}{b}\right)^{-1} $$

这个值通常很小（因为 $$n/d$$ 在短序列时不占优势，而 batch 受限于在线服务的实时性要求），使得自回归生成变为 **Memory-Bound**。这就是为什么 KV Cache 优化对推理如此重要——每次生成一个 token 都需要重新扫描全部 KV Cache。

这也解释了 MQA（Multi-Query Attention）和 GQA（Grouped-Query Attention）的动机：通过减少 KV 头的维度来降低 KV Cache 的大小，从而减少内存访问量，缓解 Memory-Bound 瓶颈。

### 2.6 Transformer 组件参数量计算

精确计算模型参数量是评估显存需求和训练可行性的基础。以 LLaMA 风格（无偏置项）的 Transformer 为例：

**嵌入层**

词嵌入矩阵：$$E \in \mathbb{R}^{V \times d_{\text{model}}}$$（$$V$$ 为词汇表大小）：

$$ \text{Params}_{\text{embed}} = V \cdot d_{\text{model}} $$

RMSNorm 嵌入层后（如 LLaMA）：包含 $$d_{\text{model}}$$ 个可学习缩放参数（如果使用）。

**自注意力（每层）**

Q、K、V、O 四个投影矩阵，每个形状为 $$d_{\text{model}} \times d_{\text{model}}$$：

$$ \text{Params}_{\text{attn}} = 4 \cdot d_{\text{model}}^2 $$

对于 GQA（Grouped-Query Attention），设查询头数 $$h_q$$，键值头数 $$h_{kv}$$（$$h_{kv} \leq h_q$$，通常 $$h_{kv} = 8$$）：

$$ \text{Params}_{\text{GQA}} = \underbrace{d_{\text{model}}^2}_{Q} + \underbrace{2 \cdot h_{kv} \cdot k \cdot d_{\text{model}}}_{K, V} + \underbrace{d_{\text{model}}^2}_{O} $$

其中 $$k = d_{\text{model}} / h_q$$ 为头维度。

**前馈网络（每层）**

标准 FFN（无偏置）：$$W_1 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$$，$$W_2 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$$：

$$ \text{Params}_{\text{FFN}} = 2 \cdot d_{\text{model}} \cdot d_{\text{ff}} $$

对于 SwiGLU（含三个投影矩阵 $$W_g, W_1, W_2$$）：

$$ \text{Params}_{\text{FFN-SwiGLU}} = 3 \cdot d_{\text{model}} \cdot d_{\text{ff}} $$

如 Lecture 3 所述，SwiGLU 通常取 $$d_{\text{ff}} \approx \frac{8}{3} \cdot d_{\text{model}}$$，使得总参数量与标准 FFN（$$d_{\text{ff}} = 4 \cdot d_{\text{model}}$$）大致持平：

$$ 3 \cdot d_{\text{model}} \cdot \frac{8}{3}d_{\text{model}} = 8 d_{\text{model}}^2 \quad \text{vs} \quad 2 \cdot d_{\text{model}} \cdot 4 d_{\text{model}} = 8 d_{\text{model}}^2 $$

**RMSNorm / LayerNorm**

每个 Norm 层包含 $$d_{\text{model}}$$ 个缩放参数（RMSNorm 只有 $$\gamma$$，无 $$\beta$$ 偏置项）。每层 Transformer Block 通常有两个 Norm 层：

$$ \text{Params}_{\text{norm per layer}} = 2 d_{\text{model}} $$

**LM Head（输出层）**

通常与嵌入矩阵共享权重（tied weights），或独立为一层：

$$ \text{Params}_{\text{lm-head}} = V \cdot d_{\text{model}} \quad (\text{若不共享}) $$

**完整模型总参数量（$$n_{\text{layers}}$$ 层）**

$$ \text{Params}_{\text{total}} \approx Vd + n_{\text{layers}} \cdot (4d^2 + 2d d_{\text{ff}} + 2d) + Vd $$

对于 LLaMA-7B（$$d=4096$$，$$d_{\text{ff}}=11008$$，$$n_{\text{layers}}=32$$，$$V=32000$$），省略 Norm 与嵌入的小项后，主力来自每层的 $$4d^2 + 2d \cdot d_{\text{ff}} \approx 4 \cdot 4096^2 + 2 \cdot 4096 \cdot 11008 \approx 157M$$ 每层，32 层共约 5B + 嵌入约 131M + LM Head 约 131M，总计约 6.7B 参数（实际 LLaMA-7B 含约 6.7B 参数）。

### 2.7 激活内存与优化器状态

训练时 GPU 显存的占用远不止模型参数本身。总显存需求由三大部分组成：

**模型参数（Parameters）**

以 FP16 混合精度训练为例，模型参数占用：$$\text{Params} \times 2$$ 字节。

**梯度（Gradients）**

与参数同形状，同样占用：$$\text{Params} \times 2$$ 字节。

**优化器状态（Optimizer States）**

AdamW 优化器为每个参数维护两个状态变量：

- 一阶动量 $$m$$（指数移动平均的梯度）
- 二阶动量 $$v$$（指数移动平均的梯度平方）

在混合精度训练中，优化器状态通常以 FP32 精度存储（数值稳定性需求），因此：

$$ \text{Optimizer States Memory} = \text{Params} \times 4 \times 2 = \text{Params} \times 8 \; \text{bytes} $$

参数、梯度、优化器状态三者合计：

| 组件 | 精度 | 每参数字节数 |
|------|------|-------------|
| Model Params (master copy) | FP32 | 4 |
| Gradients | FP32 | 4 |
| Optimizer m | FP32 | 4 |
| Optimizer v | FP32 | 4 |
| **小计** | | **16** |

这意味着：一个 $$P$$ 参数的模型在训练时至少需要 $$16P$$ 字节的显存来存放基础训练状态。对于一个 7B 参数的模型，仅这部分就接近

$$ 7 \times 10^9 \times 16 = 112 \; \text{GB} $$

**激活内存（Activation Memory）**

前向传播产生的中间激活（Activations）也必须保留到反向传播计算梯度。激活内存的估算公式为（以每个 Transformer 层为基本单位）：

$$ \text{Activation Memory}_{\text{per layer}} \approx b \cdot n \cdot d \cdot (34 + 5 \cdot \frac{n \cdot h}{d}) $$

更简洁的经验法则：在不启用激活检查点（Activation Checkpointing）的情况下，激活内存与模型参数在同一数量级，甚至更高（取决于批次大小和序列长度）。

以 LLaMA-7B 为例，$$b=1, n=2048, d=4096, h=32$$：

- 每层激活约 $$1 \times 2048 \times 4096 \times (34 + 5 \times 2048 \times 32 / 4096) \approx 954$$ MB
- 32 层合并约 30 GB 的激活内存。

**总训练显存（无梯度检查点）**

$$ M_{\text{total}} = \underbrace{16P}_{\text{params + grads + optimizer}} + \underbrace{M_{\text{activations}}}_{\text{每层激活}} + \underbrace{M_{\text{scratch}}}_{\text{临时缓冲区}} $$

对于一个 7B 模型：基础训练状态约 112 GB + 激活约 30 GB = 约 142 GB。这已经远超单张 A100-80GB，说明了为什么需要使用多 GPU 分布式训练（模型并行、流水线并行或 FSDP/ZeRO）。

### 2.8 梯度累积

梯度累积（Gradient Accumulation）是缓解显存压力的关键技术。当单 GPU 无法容纳足够大的批次大小时，可以将"大批次"拆分为多个"小批次"（micro-batches），分别在 GPU 上前向和反向传播，累积梯度但不更新参数，直到等效批次大小达到目标值。

**算法描述**：

```
设 global_batch_size = 512
设 micro_batch_size   = 8   （受限于单 GPU 显存）
设 accumulation_steps  = 512 / 8 = 64

for step in range(total_steps):
    optimizer.zero_grad()
    for micro_step in range(accumulation_steps):
        x, y = next_batch(micro_batch_size)    # 加载一个 micro-batch
        loss = model(x, y)                     # 前向传播
        loss.backward()                        # 反向传播，累积梯度
    optimizer.step()                           # 累积完成后更新参数
```

**注意事项**：

1. **Loss 放缩**：每个 micro-batch 的 loss 应除以 `accumulation_steps`（或直接在 loss 上做归一化），否则梯度将被放大。
2. **BatchNorm / LayerNorm**：RMSNorm（常用于现代 LLM）在每个 micro-batch 内部独立计算统计量，不依赖跨 micro-batch 的同步，因此与梯度累积天然兼容。这与普通 BatchNorm 不同（后者需要跨设备同步统计量）。
3. **通信开销**：梯度累积通过减少参数更新的频率来降低跨 GPU 的梯度同步通信次数（在分布式训练中至关重要），但增加了每个 micro-batch 的前向/反向计算量。

**梯度累积与算术强度的关系**：梯度累积实际上是一种"时间换空间"的策略——通过在单个 GPU 上串行处理多个 micro-batch 来模拟更大的有效批次。这与 Lecture 3 讨论的思想相呼应：注意力机制的算术强度在高批量时更高，因此使用梯度累积保持大的有效批次不仅节省显存，也有助于保持较高的 GPU 利用率。

### 2.9 小结

本章建立了一套完整的资源核算框架：

1. **einops** 以维度命名的风格让张量操作代码自文档化，在多头注意力等复杂形状变换中尤为清晰。
2. **FLOPs 计算**为模型设计提供了量化的"算力账单"——核心公式为 $$2mnk$$ 形式的矩阵乘法和 $$2bn^2d$$ 形式的注意力。
3. **GPU 内存模型**揭示了数据移动才是现代深度学习的主要瓶颈——HBM 的带宽远低于 SRAM，而算力却远超访存能力。
4. **算术强度与 Roofline 模型**给出了判断一个算子是 Compute-Bound 还是 Memory-Bound 的分析工具——这正是 FlashAttention 和 GQA 等优化存在的根本理由。
5. **参数量计算**使我们可以预先估计模型规模，而**激活内存与优化器状态**的核算则解释了为什么 7B 模型的训练需要数百 GB 显存。
6. **梯度累积**是训练中的关键技术，通过时间换空间使得大模型在有限硬件上也能训练。

这些知识不是分散的琐碎——它们构成了一个逻辑闭环：代码表达（einops）决定计算图，计算图产生 FLOPs 和内存访问，硬件特性通过 Roofline 模型决定实际性能，资源核算则告诉我们需要多少 GPU 和多少时间。
