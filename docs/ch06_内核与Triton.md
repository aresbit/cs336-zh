layout: default
title: 第6章 内核与Triton编程
prev: ch05_数据与分词.md
next: ch07_分布式训练.md
---

# 第6章 内核与Triton编程

> 原课程: CS 336 Language Modeling from Scratch, Stanford, Spring 2025
> 主讲: Tatsu Hashimoto
> 中译稿 — 深度系统编程扩展版

**导航**: [上一章: 数据与分词](ch05_数据与分词.md) | [下一章: 分布式训练](ch07_分布式训练.md)

## 一、本章概要

**核心问题**: 现代深度学习框架(PyTorch, JAX)提供了高度抽象的算子接口，为什么我们仍然需要手写 GPU 内核？答案在于一个根本性的张力: **通用框架的算子组合无法充分利用 GPU 的内存层次结构**。当一个操作可以由多个框架算子组合实现时，每个算子都会独立地从 HBM(高带宽显存)读取输入、写回输出——这种"往返"产生了巨大的内存带宽浪费。手写融合内核(fused kernel)正是为了消除这种浪费。

**动机**: 大语言模型的训练和推理本质上是 GPU 上的大规模矩阵计算。理解从 CUDA 编程模型到 Triton 语言的抽象层次，不仅让你能够手写高性能内核，更重要的是让你在阅读模型代码时能够**预测每个操作的性能行为**——为什么 attention 是访存密集的？为什么 FlashAttention 比标准 attention 快？为什么 kernel fusion 能减少 30-50% 的延迟？本章将回答这些问题。

**两条知识主线**:
- **CUDA 基础**: 理解 GPU 的物理执行模型——grid/block/thread 层次结构、共享内存、warp 调度、同步原语。这些概念构成了我们思考 GPU 性能的"底层语言"。
- **Triton 编程**: 在 CUDA 之上，Triton 提供了块级(block-level)编程抽象，让你无需手动管理线程和共享内存就能写出接近手写 CUDA 性能的内核。我们将用它实现矩阵乘法和 FlashAttention。

**深度推理主线**: 本章将从 GPU 内存层次的第一性原理出发，推导出算子融合的收益边界，然后用 Triton 代码逐步构建从简单内核到 FlashAttention 的完整实现链。每一步都附带形状分析(shapes analysis)和访存量推导。

---

## 二、为什么需要手写内核

### 2.1 框架算子的隐藏成本

考虑一个看似简单的操作: 对 Transformer 中的残差连接做 dropout 后再加回主干。在 PyTorch 中你可能会写:

```python
# 朴素的 PyTorch 实现
x = x + F.dropout(sublayer_output, p=0.1, training=True)
```

这行代码触发了什么？`F.dropout` 的内部实现大致是:

1. 生成一个与输入形状相同的随机掩码矩阵(从 HBM 分配，写入)
2. 将输入乘以掩码并除以 `(1-p)`(读输入，写输出)
3. `x + ...` 触发一次加法(读两个输入，写输出)

每一步都涉及**显式的 HBM 读写**。对于一个 `[B, S, D]` 形状的张量(例如 `[8, 2048, 4096]`, 共 256 MB), 三次读写就是 768 MB 的数据搬运——而这仅仅是 dropout + 残差连接这一个小操作。

更糟糕的是，`F.dropout` 本身在 PyTorch 中是一个**离散的 CUDA kernel launch**。每次 kernel launch 都有固定的调度开销(约 5-10 微秒)，当模型中有数百个这样的小操作时，kernel launch overhead 本身就能占据 10-20% 的总运行时间。

### 2.2 算子融合: 消除往返

**算子融合 (operator fusion)** 的核心思想是: 将多个逻辑操作合并到**单个 GPU kernel** 中执行，使得中间结果只驻留在寄存器或共享内存中，无需写回 HBM。

以上面的例子为例，融合后的内核:
1. 从 HBM 读取 `x` 和 `sublayer_output` 到寄存器
2. 在寄存器中生成随机数并应用 dropout
3. 在寄存器中执行加法
4. 将最终结果写回 HBM

**数据搬运从 3 次读 + 3 次写减少到 2 次读 + 1 次写**，减少了 50% 的 HBM 带宽消耗。

更一般地，对于 LLM 中常见的 **pointwise 操作链**(LayerNorm, GeLU, Dropout, Residual Add 等)，这些操作都是逐元素(element-wise)的，计算量极小但访存量巨大。将它们融合成一个内核后，性能瓶颈从内存带宽转移到了计算吞吐量——这正是我们希望的方向。

### 2.3 访存受限 vs. 计算受限的再审视

回顾第 2 章引入的 Roofline 模型。每个 GPU 操作可以用其**算术强度**来刻画:

$$
\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes}}
$$

逐元素操作(如 ReLU, Dropout, Add)的算术强度极低——每个元素只做 1-2 次浮点运算，但需要读取和写入若干字节。例如对于 bf16 精度的 `y = x + z`:
- FLOPs: 每个元素 1 次加法
- Bytes: 读 `x`(2 bytes), 读 `z`(2 bytes), 写 `y`(2 bytes) = 6 bytes
- 算术强度: $$1/6 \approx 0.17$$ FLOPs/Byte

而 H100 的临界算术强度约为 $$295$$ FLOPs/Byte(HBM 带宽 3.35 TB/s, bf16 算力 989.5 TFLOPS)。这意味着逐元素操作距离计算受限差了三个数量级——它们是**深度访存受限**的。

矩阵乘法恰恰相反: 对于 $$[M, K] \times [K, N]$$，算术强度约为 $$\min(M, N, K) / 2$$，当维度较大时很容易进入计算受限区域。

**核心洞察**: LLM 的一次前向传播中，矩阵乘法占约 90% 以上的 FLOPs，但只占约 10-20% 的运行时间；pointwise 操作及归一化层占不到 5% 的 FLOPs，却占据 30-40% 的运行时间。这就是算子融合的收益来源——将访存受限的操作"接入"计算受限的操作中，让昂贵的 HBM 带宽尽可能只被使用一次。

---

## 三、CUDA 编程基础

Triton 是对 CUDA 的抽象，但要理解 Triton 为什么这样设计，必须先理解 CUDA 的物理执行模型。

### 3.1 GPU 的物理架构

一块 NVIDIA GPU(以 H100 为例)由以下层次结构组成:

| 层次 | 数量(H100) | 关键特性 |
|:---|:---:|:---|
| GPC (Graphics Processing Cluster) | 8 | 顶层组织单元 |
| TPC (Texture Processing Cluster) | 每个 GPC 含 9 个 | 包含 2 个 SM |
| SM (Streaming Multiprocessor) | 132 | **调度的基本单位** |
| CUDA Core / Tensor Core | 每 SM 128 FP32 cores + 4 Tensor Cores | 执行单元 |
| L1 Cache / Shared Memory | 每 SM 256 KB(可配置) | 片上高速存储 |
| HBM (High Bandwidth Memory) | 80 GB | 片外显存 |

**关键比例**: HBM 的带宽约 3.35 TB/s，延迟约 300-800 个时钟周期；而共享内存的带宽约 20 TB/s，延迟约 20-30 个时钟周期。共享内存比 HBM 快约 6 倍带宽、20 倍延迟——这解释了为什么**将数据留在片上**(寄存器/共享内存)是一切内核优化的核心原理。

### 3.2 线程层次: Grid, Block, Thread

CUDA 的编程模型将计算组织为三个层次:

```
Grid (网格)
├── Block 0 (线程块)
│   ├── Thread 0
│   ├── Thread 1
│   └── ...
├── Block 1
│   └── ...
└── Block N-1
```

**Thread (线程)**:
- GPU 上执行的最小单元
- 拥有自己的寄存器(每 SM 65536 个 32 位寄存器)
- 同一 Block 内的线程可通过共享内存通信

**Block (线程块)**:
- 一组线程(通常 128-1024 个)，被调度到**同一个 SM** 上执行
- Block 内的线程共享一块**共享内存**(shared memory)，这是快速片上存储
- Block 之间**不能直接通信**——每个 Block 独立执行，执行顺序不确定
- Block 的维度可以是 1D, 2D 或 3D(通过 `dim3` 指定)

**Grid (网格)**:
- 一组 Block，组成一次 kernel launch
- Block 被分配到不同的 SM 上并行执行
- 如果 Block 数多于 SM 数，剩余的 Block 排队等待

**Warp 的重要性**:
在 Block 内部，线程以 32 个为一组进行调度，称为 **warp**。同一 warp 内的所有线程在每个时钟周期执行**相同的指令**(SIMT 模型——单指令多线程)。这意味着:
- 如果一个 warp 内出现分支(if-else)，两条路径会**串行执行**(warp divergence)，导致有效算力减半
- 内存访问的合并(coalesced access)至关重要——同一 warp 的 32 个线程应该访问连续的内存地址，否则一次 warp 级的内存请求会被拆分成多次请求

### 3.3 共享内存 (Shared Memory)

共享内存是程序员手动管理的片上 SRAM。它的特点是:
- 极低延迟(~20 cycles vs. ~300+ cycles for HBM)
- 极高带宽(~20 TB/s per SM)
- 容量有限(H100 每 SM 最多 256 KB，可配置为 128 KB L1 + 128 KB shared)

**编程模式**: 将需要重复访问的数据从 HBM 加载到共享内存，Block 内的所有线程协作完成加载，然后在计算中重复使用共享内存中的数据。这正是矩阵乘法分块(tiling)的基础。

```cuda
// CUDA: 使用共享内存的向量加法 (简化示例)
__global__ void vector_add_shared(float* a, float* b, float* c, int n) {
    extern __shared__ float shared_data[];  // 动态分配的共享内存
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // 协作加载到共享内存
    if (idx < n) {
        shared_data[tid] = a[idx] + b[idx];
    }
    __syncthreads();  // 确保所有线程加载完成
    
    // 写回
    if (idx < n) {
        c[idx] = shared_data[tid];
    }
}
```

### 3.4 同步原语

CUDA 提供两级同步:

**Block 内同步 (`__syncthreads()`)**:
- 作为 barrier: 直到 Block 内**所有**线程都到达该点，才继续执行
- 确保共享内存的写入对所有线程可见
- 不能用于 Block 之间

**Grid 级同步**: CUDA 没有内建的 grid 级同步。如果需要 Block 之间协调，必须通过**拆分 kernel launch**(第一个 kernel 写回 HBM, 第二个 kernel 再读取)。这是一种粗粒度的同步方式。

在 Triton 中，同步被抽象掉了——Triton 的编程模型保证每个 **program instance** 独立运行，不需要用户手动插入 barrier。

---

## 四、Triton 语言入门

### 4.1 为什么是 Triton

写 CUDA C++ 是痛苦的: 你需要手动管理线程索引、共享内存分配、bank conflict、warp divergence、寄存器压力……一个高性能的矩阵乘法 CUDA kernel 可以轻松超过 200 行，而且每次更换 GPU 架构都需要重新调优。

Triton(OpenAI 开发的开源项目)提供了一种折中: **块级编程**。你不再为单个线程编写代码，而是为 **program instance**(一个 Block)编写代码。在每个 program instance 内部，Triton 的编译器自动处理线程级的并行化和共享内存管理。

Triton 的关键抽象:

| CUDA 概念 | Triton 概念 |
|:---|:---|
| Thread | 不可见(编译器管理) |
| Block | Program instance |
| Grid | Program(由 `grid` 参数定义) |
| Shared Memory | 编译器自动管理 |
| `__syncthreads()` | 不需要(编译器保证正确性) |

### 4.2 Hello World: Triton 向量加法

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,       # 指向输入张量 x 的指针
    y_ptr,       # 指向输出张量 y 的指针
    n_elements,  # 元素总数
    BLOCK_SIZE: tl.constexpr,  # 每个 program instance 处理的元素数
):
    # 获取当前 program instance 的索引
    pid = tl.program_id(axis=0)
    
    # 计算这个 program instance 负责的起始偏移量
    block_start = pid * BLOCK_SIZE
    
    # 生成偏移量序列: [block_start, block_start+1, ..., block_start+BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # 创建掩码: 防止越界访问
    mask = offsets < n_elements
    
    # 从 HBM 加载数据
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # 计算
    output = x + 1.0
    
    # 写回 HBM
    tl.store(y_ptr + offsets, output, mask=mask)
```

**调用方式**:

```python
x = torch.randn(1000, device='cuda')
y = torch.empty_like(x)

# 计算需要多少个 program instance
grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']),)

# 启动 kernel
add_kernel[grid](x, y, x.numel(), BLOCK_SIZE=256)
```

**逐行解析**:

1. `@triton.jit`: 将 Python 函数编译为 GPU 可执行代码。注意这个装饰器标记的函数会在**编译时**被 Triton 编译器处理。

2. `tl.program_id(axis=0)`: 返回当前 program instance 在维度 0 上的索引(类比 CUDA 中的 `blockIdx.x`)。

3. `tl.arange(0, BLOCK_SIZE)`: 生成一个从 0 到 `BLOCK_SIZE-1` 的整数序列 `[0, 1, 2, ..., BLOCK_SIZE-1]`。这是 Triton 中构造索引的基础操作。

4. `offsets = block_start + tl.arange(...)`: 通过将基础偏移量广播到 `arange` 上，得到这个 program instance 要处理的全局索引列表。

5. `mask = offsets < n_elements`: 创建布尔掩码。当 `n_elements` 不能被 `BLOCK_SIZE` 整除时，最后一个 program instance 需要处理少于 `BLOCK_SIZE` 个元素——掩码确保不会越界读写。

6. `tl.load(x_ptr + offsets, mask=mask)`: 从 HBM 的指定地址加载数据。`mask` 参数确保只有有效位置的地址被访问。无效位置返回未定义值(但不读取——避免 segfault)。

7. `tl.store(y_ptr + offsets, output, mask=mask)`: 将计算结果写回 HBM。`mask` 同样保证只写入有效位置。

### 4.3 指针算术与内存模型

在 Triton 中，张量被表示为**裸指针**: 你需要传入张量的 `.data_ptr()` 以及各个维度的步长(stride)。例如，对于一个 `[M, N]` 的行主序(row-major)矩阵 `A`:
- `A[i, j]` 的地址 = `A_ptr + i * A_stride_0 + j * A_stride_1`

Triton 提供了 `tl.make_block_ptr` 和 `tl.advance` 等工具函数来简化多维指针操作，但对于初学者，手写偏移量计算是最好的理解方式。

```python
@triton.jit
def load_2d_tile(
    A_ptr,           # 矩阵 A 的数据指针
    M, N,            # 矩阵维度(行数, 列数)
    stride_am,       # A 在行方向上的步长(stride_0)
    stride_an,       # A 在列方向上的步长(stride_1)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    # 当前 tile 的起始行和列
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # 将 1D 行索引和 1D 列索引转为 2D 偏移量
    # A_ptrs 的形状: [BLOCK_M, BLOCK_N]
    A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an
    
    # 创建掩码
    mask_m = offs_m[:, None] < M
    mask_n = offs_n[None, :] < N
    
    # 加载 tile
    a = tl.load(A_ptrs, mask=(mask_m & mask_n), other=0.0)
    return a
```

关键细节: `offs_m[:, None]` 创建一个 `[BLOCK_M, 1]` 的列向量，`offs_n[None, :]` 创建一个 `[1, BLOCK_N]` 的行向量。广播后得到 `[BLOCK_M, BLOCK_N]` 的偏移量矩阵——这正是矩阵 tiling 的核心索引技巧。

### 4.4 tl.dot: Tensor Core 加速的矩阵乘法

`tl.dot` 是 Triton 中最重要的操作之一——它调用 GPU 的 **Tensor Core** 来执行小型矩阵乘法。Tensor Core 是 NVIDIA Volta 架构后引入的专用硬件单元，能在单个时钟周期内完成 $$4 \times 4$$ 的矩阵乘加运算(H100 上为 $$8 \times 4 \times 16$$ 的 mma 指令)。

```python
# C = A @ B, 其中 A: [M, K], B: [K, N], C: [M, N]
c = tl.dot(a, b)
# a: [BLOCK_M, BLOCK_K] — 从共享内存或寄存器加载的 tile
# b: [BLOCK_K, BLOCK_N] — 从共享内存或寄存器加载的 tile
# c: [BLOCK_M, BLOCK_N] — 输出 tile(累加到已有值)
```

**Tensor Core 的使用约束**:
- 输入必须是 bf16 或 fp16(精度取决于 GPU 架构)
- 输出/累加器通常是 fp32(高精度累加)
- H100 上的最佳 tile 大小: `16 x 16`(wmma) 或 `128 x 128 x 32`(mma)

`tl.dot` 的语义是 `c = c + a @ b`(累加模式)。如果你需要重置而非累加，需要先将 `c` 初始化为零:

```python
c = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
c = tl.dot(a, b, c)  # c 作为累加器传入
```

---

## 五、在 Triton 中实现矩阵乘法

### 5.1 朴素矩阵乘法及其瓶颈

标准矩阵乘法 $$C = A \times B$$ 的朴素实现直接使用了三重循环:

```python
for i in range(M):
    for j in range(N):
        acc = 0
        for k in range(K):
            acc += A[i, k] * B[k, j]
        C[i, j] = acc
```

在 GPU 上，这对应着每个线程计算 $$C$$ 的一个元素。问题在于数据复用: $$A$$ 的每个元素被读取 $$N$$ 次，$$B$$ 的每个元素被读取 $$M$$ 次——而每次读取都是从 HBM 进行的(如果没有缓存的帮助)。算术强度约为:

$$
AI_{\text{naive}} \approx \frac{2MNK}{2MK + 2KN + 2MN} \approx \frac{K}{2}
$$

当 $$K$$ 较小时(例如 attention 中的 `head_dim = 128`)，算术强度只有约 64 FLOPs/Byte——远低于 H100 的临界值，这意味着朴素矩阵乘法是**访存受限**的。

### 5.2 分块 (Tiling): 核心优化思想

分块的核心思想是将矩阵划分为小块(tile)，每个 tile 小到可以放入共享内存，从而在 tile 内部充分利用片上带宽进行高强度的计算复用。

**分块矩阵乘法的步骤**:

1. 将输出矩阵 $$C$$ 划分为 `[BLOCK_M, BLOCK_N]` 的 tile
2. 对于每个输出 tile，沿 K 维迭代:
   a. 加载 $$A$$ 的一个 `[BLOCK_M, BLOCK_K]` tile 到共享内存
   b. 加载 $$B$$ 的一个 `[BLOCK_K, BLOCK_N]` tile 到共享内存
   c. 在共享内存中的数据上执行 `tl.dot`
   d. 累加到输出累加器
3. 将累加器写回 HBM

**分块的收益量化**:

每个 $$A$$ tile 被加载一次，但对其中的每个元素，与 $$B$$ tile 中 $$BLOCK_N$$ 个元素各做一次乘加。因此:

$$
\text{每个 } A \text{ 元素的计算} = 2 \times BLOCK\_N \text{ FLOPs}
$$

而读取成本仅 $$2$$ bytes(bf16)。所以分块后的算术强度:

$$
AI_{\text{tiled}} \approx \frac{2 \times BLOCK\_N}{2} = BLOCK\_N
$$

取 $$BLOCK\_N = 128$$，算术强度约为 128 FLOPs/Byte——比朴素实现提升了一倍。如果再加上对 $$B$$ 的复用，实际算术强度更高，接近:

$$
AI_{\text{tiled}} \approx \frac{BLOCK\_M \times BLOCK\_N}{2} \quad (\text{当 } K \gg BLOCK\_K \text{ 时})
$$

### 5.3 完整的 Triton 矩阵乘法

```python
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,    # A 的步长
    stride_bk, stride_bn,    # B 的步长
    stride_cm, stride_cn,    # C 的步长
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Triton 分块矩阵乘法: C = A @ B"""
    # Program ID: 输出 tile 在 (M, N) 网格中的位置
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    # 输出 tile 的起始位置
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # A 和 B tile 在 K 维上的偏移量
    offs_k = tl.arange(0, BLOCK_K)
    
    # 指针指向当前 tile
    A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    # 累加器(高精度)
    accumulator = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    # 掩码
    mask_m = offs_m[:, None] < M
    mask_n = offs_n[None, :] < N
    
    # 沿 K 维迭代
    for k in range(0, K, BLOCK_K):
        # 加载 A tile (HBM -> 寄存器, Triton 编译器管理)
        mask_k = offs_k[None, :] < (K - k)
        a = tl.load(A_ptrs, mask=mask_m & mask_k, other=0.0)
        
        # 加载 B tile
        mask_k_col = offs_k[:, None] < (K - k)
        b = tl.load(B_ptrs, mask=mask_k_col & mask_n, other=0.0)
        
        # Tensor Core 矩阵乘法
        accumulator = tl.dot(a, b, accumulator)
        
        # 推进指针(而不是重新计算偏移量)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk
    
    # 掩码
    c_mask = mask_m & mask_n
    
    # 写回 HBM (可选: 应用激活函数后再写入)
    C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(C_ptrs, accumulator.to(tl.bfloat16), mask=c_mask)
```

**调用函数**:

```python
def matmul(a: torch.Tensor, b: torch.Tensor):
    """封装 Triton 矩阵乘法为 PyTorch 兼容接口"""
    assert a.shape[1] == b.shape[0], "维度不匹配"
    assert a.is_cuda and b.is_cuda
    
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # 网格大小: (ceil(M/BLOCK_M), ceil(N/BLOCK_N))
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
    )
    return c
```

### 5.4 性能分析与 tile 大小选取

tile 大小的选择需要在以下约束间权衡:

| 约束 | 说明 |
|:---|:---|
| 共享内存容量 | 每个 SM 的共享内存有限(H100 上约 228 KB)，A tile 和 B tile 必须能同时放入 |
| 寄存器压力 | 较大的 tile 需要更多寄存器来维护累加器。每线程寄存器数有限(255 for H100) |
| 占用率 (Occupancy) | 较大的 tile = 每个 Block 使用更多资源 = 每个 SM 上能并行执行的 Block 数减少 |
| Tensor Core 形状 | `tl.dot` 的 tile 形状应匹配 Tensor Core 的输入要求(推荐 16 的倍数) |
| 全局内存合并 | tile 的维度应使 warp 内的线程访问连续的内存地址 |

**经验法则**(对于 H100 + bf16):
- $$BLOCK\_M = BLOCK\_N = 128$$
- $$BLOCK\_K = 32$$ 或 $$64$$
- 累加器 ```[128, 128]``` 总共 16384 个 fp32 元素 = 64 KB，可放入寄存器

---

## 六、在 Triton 中实现 FlashAttention

### 6.1 标准注意力的计算与内存困境

标准缩放点积注意力的定义:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

对于序列长度 $$S$$ 和头维度 $$d$$，朴素实现的计算与内存特征:

**步骤 1**: $$S = QK^T$$，形状 `[S, S]`
- FLOPs: $$2 S^2 d$$
- 内存: 需要分配 $$O(S^2)$$ 的中间矩阵 `[S, S]`

**步骤 2**: $$P = \text{softmax}(S)$$
- FLOPs: $$O(S^2)$$
- 内存: 需要分配另一个 `[S, S]`(softmax 输出)

**步骤 3**: $$O = P V$$
- FLOPs: $$2 S^2 d$$
- 内存: 输出 `[S, d]`

**核心问题**: 中间矩阵 $$S$$ 和 $$P$$ 的大小是 $$O(S^2)$$(每个都是 $$S \times S$$)，需要从 HBM 写入再读出。当 $$S = 8192$$ 时，每个 `[S, S]` 矩阵占 $$8192^2 \times 2 \text{ bytes} = 128 \text{ MB}$$。加上读写过程，仅这两个中间矩阵就要占用 GB 级的 HBM 带宽。

传统 attention 的算术强度:

$$
AI_{\text{attention}} \approx \frac{4 S^2 d}{2 S^2 + 2 S d} \approx \frac{4 d}{2 + 2d/S} \approx \frac{4 S d}{S + d} \quad (\text{当 } S \gg d)
$$

当 $$S \gg d$$(长序列)时，$$AI \approx 4d$$——对于 $$d = 128$$，仅约 512 FLOPs/Byte。尽管比 pointwise 操作好，但仍然不是计算受限(需要 > 295 才能到达 H100 的 Ridge Point)。

### 6.2 FlashAttention 的核心洞察: 分块 + 重计算

FlashAttention (Dao et al., 2022) 的核心思想是 **IO-aware 算法设计**: 既然 $$S^2$$ 中间矩阵是内存瓶颈，那就**永不在 HBM 中物化完整的 $$S$$ 和 $$P$$ 矩阵**。

**实现方案**: 将 $$Q$$ 沿序列维分块，对每个 $$Q$$ tile，迭代加载 $$K$$ 和 $$V$$ 的 tile 到片上，在片上完成 softmax 和加权求和。关键技巧是 **online softmax**——一种可以在一次遍历中完成 softmax 归一化的算法。

**Online Softmax 的原理**:

标准 softmax 需要两次遍历: 第一次找最大值(用于数值稳定性)，第二次计算指数和归一化。问题在于，对于长度为 $$S$$ 的向量，我们无法一次性将所有元素加载到片上。

Online softmax 解决这个问题的思路是: **在遍历过程中维护最大值和归一化因子的运行估计，并在发现新的最大值时修正之前的累加和**。

设我们已处理了前 $$t$$ 个元素，维护:
- $$m_t$$: 前 $$t$$ 个元素的最大值
- $$d_t$$: 前 $$t$$ 个元素未归一化的指数和 $$\sum_{i=1}^{t} e^{x_i - m_t}$$

当遇到一个新元素 $$x_{t+1}$$ 时，有两种情况:

**情况 1**: $$x_{t+1} \leq m_t$$(新的最大值不更新)
$$
\begin{aligned}
d_{t+1} &= d_t + e^{x_{t+1} - m_t}
\end{aligned}
$$

**情况 2**: $$x_{t+1} > m_t$$(发现新的最大值)
$$
\begin{aligned}
m_{t+1} &= x_{t+1} \\
d_{t+1} &= d_t \cdot e^{m_t - m_{t+1}} + 1
\end{aligned}
$$

注意 $$d_t \cdot e^{m_t - m_{t+1}}$$ 这一项——它把之前所有元素的指数值按新的最大值**重新缩放**。这是一个 $$O(1)$$ 的修正，不需要重新访问之前的元素。

最终 softmax 结果: $$\text{softmax}(x)_i = e^{x_i - m_S} / d_S$$。

**扩展到矩阵**: 在 attention 中，我们对 $$QK^T$$ 的每一行独立计算 softmax(result 的每一行对应一个 query)。Online softmax 的矩阵版本意味着: 对 $$Q$$ 的每个 tile，沿 $$K$$ 维分块迭代，维持行级的 $$m$$ 和 $$d$$。

### 6.3 FlashAttention 的 Triton 实现

```python
@triton.jit
def _fwd_kernel(
    Q, K, V,                     # 输入张量指针
    sm_scale,                    # 1/sqrt(d_k) 缩放因子
    L,                           # logsumexp 输出(用于反向传播)
    O,                           # 输出指针
    stride_qb, stride_qh, stride_qm, stride_qk,  # Q 的步长
    stride_kb, stride_kh, stride_kn, stride_kk,  # K 的步长
    stride_vb, stride_vh, stride_vk, stride_vn,  # V 的步长
    stride_ob, stride_oh, stride_om, stride_on,  # O 的步长
    BATCH, N_HEADS,              # 批大小，头数
    N_CTX,                       # 序列长度
    BLOCK_M: tl.constexpr,       # Q tile 大小(沿序列维)
    BLOCK_DMODEL: tl.constexpr,  # 头维度(固定)
    BLOCK_N: tl.constexpr,       # K/V tile 大小(沿序列维)
):
    # 获取当前 program 的批索引和头索引
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    # 偏移量
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Q 指针: [BLOCK_M, BLOCK_DMODEL]
    q_ptrs = Q + off_hz * stride_qh + \
             offs_m[:, None] * stride_qm + \
             offs_d[None, :] * stride_qk
    
    # 加载 Q tile
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    
    # 初始化累加器和状态
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')  # 运行最大值
    d_i = tl.zeros([BLOCK_M], dtype=tl.float32)                  # 运行归一化因子
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)    # 输出累加器
    
    # 沿 K/V 序列维迭代
    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n_curr = start_n + offs_n
        
        # 加载 K tile: [BLOCK_N, BLOCK_DMODEL]
        k_ptrs = K + off_hz * stride_kh + \
                 offs_n_curr[:, None] * stride_kn + \
                 offs_d[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=offs_n_curr[:, None] < N_CTX, other=0.0)
        
        # 加载 V tile: [BLOCK_N, BLOCK_DMODEL]
        v_ptrs = V + off_hz * stride_vh + \
                 offs_n_curr[:, None] * stride_vk + \
                 offs_d[None, :] * stride_vn
        v = tl.load(v_ptrs, mask=offs_n_curr[:, None] < N_CTX, other=0.0)
        
        # 计算 Q @ K^T: [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, k.T)  # k.T 是转置
        
        # 缩放 + 因果掩码(如果需要)
        qk = qk * sm_scale
        # 因果掩码: 只允许 query position >= key position
        causal_mask = offs_m[:, None] >= offs_n_curr[None, :]
        qk = qk + tl.where(causal_mask, 0.0, float('-inf'))
        
        # Online softmax 更新
        m_ij = tl.max(qk, axis=1)  # 每行的最大值
        p = tl.exp(qk - m_ij[:, None])  # 减法稳定数值
        
        # 更新状态
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)      # 旧累加器的缩放因子
        beta = tl.exp(m_ij - m_new)       # 新行的缩放因子
        
        # 重新缩放累加器并累加新的行
        d_i = d_i * alpha + tl.sum(p * beta, axis=1)
        acc = acc * alpha[:, None] + tl.dot((p * beta[:, None]).to(tl.bfloat16), v)
        
        m_i = m_new
    
    # 最终归一化
    acc = acc / d_i[:, None]
    
    # 写回 HBM
    o_ptrs = O + off_hz * stride_oh + \
             offs_m[:, None] * stride_om + \
             offs_d[None, :] * stride_on
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=offs_m[:, None] < N_CTX)
    
    # 保存 logsumexp 用于反向传播
    l_ptrs = L + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, m_i + tl.log(d_i), mask=offs_m < N_CTX)
```

**逐段解析**:

1. **Tile 划分**: Q 沿序列维分块(`BLOCK_M`)，K 和 V 沿序列维分块(`BLOCK_N`)。外层循环遍历 Q tile，内层循环遍历 K/V tile。

2. **Online Softmax 的三个状态变量**:
   - `m_i`: 当前行中已见元素的最大值(`[BLOCK_M]` 向量，每个 query 位置一个最大值)
   - `d_i`: 当前行的未归一化指数和(`[BLOCK_M]` 向量)
   - `acc`: 输出加权和(`[BLOCK_M, BLOCK_DMODEL]` 矩阵)

3. **状态更新数学**: 当处理新的 K/V tile 时:
   - 计算 $$S_{ij} = Q_i K_j^T / \sqrt{d_k}$$
   - 找到新的行最大值: $$m_{\text{new}} = \max(m_i, m_{ij})$$
   - 计算缩放因子: $$\alpha = \exp(m_i - m_{\text{new}})$$(旧值的缩放), $$\beta = \exp(m_{ij} - m_{\text{new}})$$(新值的缩放)
   - 更新累加和: $$d_{\text{new}} = d_{\text{old}} \cdot \alpha + \sum \exp(S_{ij} - m_{ij}) \cdot \beta$$
   - 更新加权和: $$O_{\text{new}} = O_{\text{old}} \cdot \alpha + P_{ij} \cdot \beta \cdot V_j$$

### 6.4 重计算 vs. 存储的权衡

FlashAttention 在反向传播中采用**重计算**策略。由于在前向传播中没有保存完整的 softmax 矩阵 $$P$$(这恰好是 FlashAttention 加速的来源)，反向传播需要重新计算 attention 权重。

**前向存储**: 仅需保存:
- $$Q, K, V$$: 总大小 $$O(S \cdot d)$$
- Logsumexp $$L$$: 大小为 $$O(S)$$
- 伪随机数种子(用于 dropout)

**反向计算**: 使用保存的 $$Q, K, V$$ 和 $$L$$ 重新执行 FlashAttention 的前向逻辑，在重计算 softmax 权重的同时计算梯度。这额外使用了约 $$O(S)$$ 的显存，但避免了 $$O(S^2)$$ 的中间矩阵存储。

**为什么重计算是值得的**:

| | 标准 Attention | FlashAttention |
|:---|:---|:---|
| 前向中间结果 | $$O(S^2)$$ | $$O(S)$$ |
| 反向重计算 | 不重计算 | 重计算 attention weights |
| 总内存 | $$O(S^2)$$ | $$O(S)$$ |
| 速度(训练) | 1x | 2-4x 快 |
| 速度(推理) | 1x | 2-3x 快 |

对于 $$S=4096, d=128$$:
- 标准 attention: 中间矩阵 $$4096^2 \cdot 2 = 32 \text{ MB}$$ (bf16)
- FlashAttention: logsumexp $$4096 \cdot 4 = 16 \text{ KB}$$ (fp32)

**差异超过 2000 倍**。

---

## 七、内核融合与自动调优

### 7.1 常见融合模式

在 Transformer 模型中，以下内核融合模式带来了显著的端到端加速:

**模式 1: MLP 融合**

```python
@triton.jit
def fused_mlp_kernel(
    x_ptr, w1_ptr, w2_ptr, out_ptr,
    # ... strides and dimensions ...
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """融合: x @ w1 -> SiLU -> x @ w2 (SwiGLU 的简化版本)"""
    pid = tl.program_id(0)
    
    # 阶段 1: x @ w1 (gate projection)
    gate = matmul_tile(x_ptr, w1_ptr, ...)
    
    # Pointwise 激活(在寄存器中, 零额外访存)
    gate = gate * tl.sigmoid(gate)  # SiLU
    
    # 阶段 2: gate @ w2 (down projection)
    # ... 在同一个 kernel 中完成 ...
```

收益: 消除 gate projection 输出写回 HBM + 重新读取的开销。对 Llama 的 MLP 块，这可以节省约 20% 的延迟。

**模式 2: Attention + 残差 + LayerNorm 融合**

将 attention 输出、残差连接和 LayerNorm 融合到一个 kernel 中。这在推理中特别有效，因为 Decode 阶段这些操作全部是访存受限的。

**模式 3: KV Cache 更新融合**

在 Decode 阶段，新生成的 token 的 K 和 V 需要写入 KV Cache。传统实现需要: 计算 K/V -> 写入 KV Cache -> 读取 KV Cache(下一个迭代)。融合方案直接在生成 K/V 的 kernel 中将其写入正确的 KV Cache 位置，消除一次读写往返。

### 7.2 Autotune: 自动搜索最优 tile 配置

tile 大小的选择对性能影响巨大，但最优值依赖于具体硬件(GPU 型号、SM 数量、共享内存大小)和数据形状。Triton 通过 `@triton.autotune` 装饰器提供自动调优:

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
    ],
    key=['M', 'N', 'K'],  # 根据这些参数的值来缓存最佳配置
)
@triton.jit
def matmul_kernel(A_ptr, B_ptr, C_ptr, M, N, K, ...):
    ...
```

**Autotune 的工作机制**:

1. 在**第一次调用**时，Triton 对 `configs` 列表中的每个配置执行一次 benchmark(运行数次 kernel launch 取平均)
2. 选择最快的配置，并将 `(key 值 -> 最佳配置)` 的映射缓存到磁盘
3. 后续调用直接使用缓存的最佳配置

**参数说明**:
- `num_stages`: 软件流水线级数。较大的值可以利用更多的异步拷贝指令隐藏延迟，但使用更多共享内存
- `num_warps`: 每个 program instance 使用的 warp 数。4 warps = 128 threads, 8 warps = 256 threads
- `key`: 用于决定何时重新 benchmark 的参数列表。例如 `['M', 'N', 'K']` 意味着不同矩阵形状会各自生成最佳配置

### 7.3 软件流水线(Software Pipelining)

`num_stages` 参数控制 **异步拷贝流水线** 的深度。工作原理:

```
没有流水线:
  load A_tile -> compute -> load B_tile -> compute -> ... 

有流水线 (num_stages=2):
  load A_tile_1 | compute tile_1
                | load A_tile_2    | compute tile_2
                                   | load A_tile_3   | ...
```

通过提前发出下一次迭代的 load 指令，将内存访问与计算重叠，可以隐藏 HBM 的访存延迟(约 300-800 cycles)。`num_stages=3` 或 `4` 在长 $$K$$ 维度的矩阵乘法中特别有效。

---

## 八、调试与性能分析

### 8.1 Triton Debugger

Triton 提供了 `TRITON_INTERPRET=1` 环境变量来启用解释器模式，在 CPU 上逐行执行内核代码，方便调试:

```bash
TRITON_INTERPRET=1 python my_kernel_test.py
```

在解释器模式下，`tl.load` 和 `tl.store` 表现为普通的数组访问，`tl.dot` 展开为标准矩阵乘法。这允许你使用常规的 Python 调试器检查中间值。

### 8.2 NVIDIA Nsight Systems

Nsight Systems 是 NVIDIA 的 GPU 系统级性能分析工具，提供时间线上的 kernel 执行视图:

```bash
nsys profile -o profile_output python my_training_script.py
```

**关键分析指标**:
- **Kernel duration**: 每个 kernel 的执行时间。如果有很多微小的 kernel(每个 < 10 us)，考虑融合
- **Memory operations**: HBM 读写量。如果内存操作占比 > 60%，说明是访存受限
- **SM utilization**: Tensor Core 利用率。如果 < 80%，可能有 kernel launch overhead 或线程调度问题

### 8.3 PyTorch Profiler

PyTorch Profiler 提供了更易用的 Python 接口:

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    output = model(input_data)

# 打印关键统计
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# 导出 Chrome trace(在 chrome://tracing 中查看)
prof.export_chrome_trace("trace.json")
```

**常见性能问题的诊断指南**:

| 症状 | 可能原因 | 修复方向 |
|:---|:---|:---|
| 大量 < 10us 的 CUDA kernel | 过多的小 kernel launch | 内核融合 |
| GPU 利用率 < 60% | CPU 成为瓶颈(数据加载) | 增加 DataLoader workers, 预取 |
| Tensor Core 利用率 < 50% | 矩阵形状不利于 Tensor Core | 调整 tile 大小或 padding |
| HBM 带宽接近上限但算力很低 | 访存受限 | 增加批量, 融合, 使用 fp8 |
| 算力接近上限但 HBM 带宽低 | 计算受限(理想状态) | 考虑更高效的算法或增加数据并行 |

### 8.4 精度调试

Triton 内核与 PyTorch 参考实现之间的数值差异是常见的调试挑战。推荐策略:

```python
# 对比 Triton 内核与 PyTorch 参考实现
def check_accuracy(triton_fn, torch_fn, *inputs, atol=1e-3, rtol=1e-3):
    triton_out = triton_fn(*inputs)
    torch_out = torch_fn(*inputs)
    
    abs_diff = (triton_out - torch_out).abs()
    rel_diff = abs_diff / (torch_out.abs() + 1e-6)
    
    max_abs = abs_diff.max().item()
    max_rel = rel_diff.max().item()
    
    print(f"Max absolute difference: {max_abs:.6e}")
    print(f"Max relative difference: {max_rel:.6e}")
    
    if max_rel > rtol and max_abs > atol:
        print("WARNING: Significant numerical differences detected!")
        # 定位差异最大的位置
        worst_idx = abs_diff.argmax().item()
        print(f"  Worst at index {worst_idx}: "
              f"triton={triton_out.flatten()[worst_idx].item():.6f}, "
              f"torch={torch_out.flatten()[worst_idx].item():.6f}")
```

注意: bf16 的精度限制(约 3 位有效数字)意味着一些差异是固有的。关键是确保差异在合理的数值误差范围内(通常 $$< 10^{-3}$$ 相对误差)，且不会在多层传播中累积。

---

## 九、内核开发的最佳实践

### 9.1 开发流程

一个可靠的内核开发流程:

1. **在 PyTorch 中编写参考实现**: 确保算法正确性，建立数值基准
2. **移植到 Triton(解释器模式)**: 使用 `TRITON_INTERPRET=1` 在 CPU 上验证逻辑
3. **小规模正确性验证**: 对小输入对比 Triton GPU 输出与 PyTorch 参考输出
4. **性能基准测试**: 使用 `triton.testing.Benchmark` 测量吞吐量
5. **Autotune**: 搜索最优 tile 配置
6. **大规模压力测试**: 对不同形状和 dtype 组合验证正确性和性能

### 9.2 常见陷阱

**陷阱 1: 未对齐的内存访问**
- 确保矩阵维度和步长是 16 或 32 的倍数(对齐到 GPU 的缓存行)
- 对于不规则维度，考虑 padding

**陷阱 2: 共享内存 Bank Conflict**
- 当同一 warp 的多个线程访问同一 bank(每 32-bit 为一个 bank)的不同地址时，访问被序列化
- Triton 编译器自动处理大部分 bank conflict，但特定访问模式(如跨步访问)可能仍需手动调整

**陷阱 3: 寄存器溢出 (Register Spilling)**
- 当 local 变量超过每线程的寄存器限制时(255 for H100)，变量被"溢出"到 L1 缓存
- 症状: 性能突然下降 30-50%
- 修复: 减小 tile 大小或减少 `num_warps`

**陷阱 4: 过度优化**
- 在投入大量时间手写 Triton 内核前，先检查 PyTorch 的 `torch.compile` 是否已经能生成足够好的融合内核
- 手写内核只在以下情况中真正需要: (1) 涉及 `S^2` 内存的 attention 类操作, (2) 需要特殊的融合模式, (3) `torch.compile` 无法处理的非常规形状

---

## 十、总结与展望

### 10.1 本章核心要点

1. **算子融合是填充 Roofline 间隙的关键**: 逐元素操作是深度访存受限的(算术强度 < 1)，通过将它们融合到矩阵乘法中，可以大幅减少 HBM 带宽消耗。

2. **CUDA 的执行模型(Grid/Block/Thread/Warp)构成了 GPU 编程的底层语言**: 即使使用 Triton 这样的高级抽象，理解物理执行模型仍然是诊断性能问题的基础。

3. **Triton 通过块级编程抽象出线程管理**: `tl.program_id`, `tl.arange`, `tl.load`, `tl.store`, `tl.dot` 是五个基本原语，掌握了它们就能写出 90% 的常用内核。

4. **分块(Tiling)是 GPU 上一切高性能计算的基石**: 无论是矩阵乘法还是 FlashAttention，分块的目的都是将数据保留在片上(寄存器/共享内存)，最大化计算与访存的比值。

5. **FlashAttention 的 IO-aware 设计将内存复杂度从 $$O(S^2)$$ 降到 $$O(S)$$**: 其技术核心是 online softmax——一个可以在单次遍历中完成 softmax 归一化的数学技巧——以及反向传播中的重计算策略。

6. **Autotune 是工程化的必要条件**: tile 大小的最优选择依赖于硬件和数据形状，手动调优不可扩展。`@triton.autotune` 提供了自动化的解决方案。

### 10.2 与课程整体的连接

本章在 CS 336 课程体系中扮演着 **"从理论到实现"** 的桥接角色:
- **前序章节**: 第 2-5 章建立了 Transformer 架构、数据管道和分词——这些告诉我们需要计算什么
- **本章**: 讲授如何高效地执行这些计算——GPU 内核是 Transformer 的物理实现基础
- **后续章节**: 第 7 章(分布式训练)将本章的单 GPU 内核扩展到多 GPU/多节点——如何在更大的规模上组织计算

在手写内核时，始终记住我们在第 2 章中分析的算术强度和 Roofline 模型——它们不是抽象的理论，而是指导你选择 tile 大小、决定是否需要融合的直接工具。

---

**参考阅读**:

- *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (Dao et al., NeurIPS 2022)
- *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning* (Dao, 2023)
- *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations* (Tillet et al., MAPS 2019)
- NVIDIA CUDA C++ Programming Guide — 第 3 章 (Programming Model), 第 5 章 (Performance Guidelines)
- *Roofline: An Insightful Visual Performance Model for Multicore Architectures* (Williams et al., CACM 2009)

---

**导航**: [上一章: 数据与分词](ch05_数据与分词.md) | [下一章: 分布式训练](ch07_分布式训练.md)
