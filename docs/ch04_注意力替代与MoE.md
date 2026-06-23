<div style="display:flex; justify-content:space-between; align-items:center; padding:1em 0;">
  <div><a href="ch03_Transformer架构演进.md">← 第3章 Transformer架构演进</a></div>
  <div><a href="index.md">↑ 目录</a></div>
  <div><a href="ch05_训练优化与微调.md">第5章 训练优化与微调 →</a></div>
</div>

# 第4章: 注意力替代方案与混合专家模型

> 原课程: CS 336 Language Modeling from Scratch, Stanford, Spring 2025
> 主讲: Tatsu Hashimoto
> 译注: 本章覆盖注意力机制的复杂度瓶颈、稀疏与线性注意力、状态空间模型（Mamba/S6/GDN）以及混合专家模型（MoE）的完整技术栈。

---

## 一、本章概要

Transformer 的自注意力机制赋予了模型捕捉长程依赖的能力，但其计算复杂度与序列长度 $$n$$ 呈二次关系——$$O(n^2)$$。当上下文窗口从几千 token 扩展到百万 token 时，这一瓶颈变得不可承受。本章系统性地考察两类解决路径: **注意力替代方案**（从算法层面降低注意力成本）和**混合专家模型**（从架构层面用稀疏激活换取更大容量）。

学完本章后，你应当能够:

1. 从矩阵乘法顺序的角度推导线性注意力的核心技巧——$$QK^\top V = Q(K^\top V)$$——并解释其从二次到线性的复杂度转变。
2. 区分滑动窗口、膨胀、全局+局部三种稀疏注意力模式的计算图差异，理解 DSA（DeepSeek Sparse Attention）的"事后适配"策略。
3. 写出线性注意力的循环形式 $$S_t = S_{t-1} + k_t v_t^\top$$，并解释其训练并行/推理串行的对偶性。
4. 描绘从线性注意力到 Mamba-2 再到 Gated Delta Net 的演化路径，特别是门控和选择性擦除带来的表达能力提升。
5. 手写 MoE 层的 top-$$k$$ 路由公式，解释负载均衡损失的梯度机制，以及专家容量（expert capacity）概念。
6. 对比 DeepSeek MoE v1 到 v3 三代架构的演进——细粒度专家、共享专家、无辅助损失路由、MLA 和 MTP。
7. 根据应用场景（长文档 vs 低延迟生成 vs 大规模训练）选择合适的替代方案。

> 核心主题: 当 $$O(n^2)$$ 不可接受时，我们如何在稀疏性、表达力和工程复杂度之间做出权衡？

---

## 二、注意力复杂度瓶颈

### 2.1 二次复杂度从何而来

标准的缩放点积注意力定义为:

$$
\text{Attn}(Q, K, V) = \rho(QK^\top) V
$$

其中 $$\rho(\cdot)$$ 通常是 softmax（按行），$$Q, K \in \mathbb{R}^{n \times d_k}$$，$$V \in \mathbb{R}^{n \times d_v}$$。矩阵乘法 $$QK^\top$$ 产生一个 $$n \times n$$ 的注意力分数矩阵，计算代价为 $$O(n^2 d_k)$$。

对于典型的 LLM 推理场景:
- 短上下文（$$n \approx 2\text{K}$$）: $$n^2 = 4\text{M}$$，与 FFN 的计算量相比尚可接受
- 中等上下文（$$n \approx 32\text{K}$$）: $$n^2 \approx 1\text{B}$$，注意力开始成为瓶颈
- 长上下文（$$n \approx 1\text{M}$$）: $$n^2 = 1\text{T}$$，单层注意力的浮点运算量就超过了整个稠密模型的 FFN 计算

此外，注意力分数矩阵 $$n \times n$$ 需要显存存储，在长上下文下直接超出 GPU 显存容量。

### 2.2 基本工具箱: 局部+全局混合

在实际系统工程中，最简单的缓解策略是将**局部注意力**（每个 token 只关注其邻域窗口内的 token）与**全局注意力**（少数特殊 token 关注全体序列）结合使用。这一思想的典型代表包括:

- **Longformer** (Beltagy et al., 2020): 滑动窗口局部注意力 + 选定的全局 token
- **BigBird** (Zaheer et al., 2020): 滑动窗口 + 随机稀疏 + 全局 token 的组合
- **Gemini 1.5**: 在实践中大量使用混合注意力模式

这种方式将复杂度从 $$O(n^2)$$ 降至 $$O(n \cdot w)$$（$$w$$ 为窗口大小），工程实现简单——只需在注意力掩码中置零即可。但它仍然依赖系统工程的仔细调优，且对于极端长上下文（$$n > 100\text{K}$$），即使 $$O(n \cdot w)$$ 也可能成为负担。

> 关键权衡: 混合注意力可以解决大部分实际问题，但如果我们追求更根本的、可能更大的收益，就需要考察更激进的方案。

---

## 三、线性注意力

### 3.1 核心洞察: 矩阵乘法顺序的交换

考虑一个"去掉 softmax"的简化注意力操作（即 $$\rho$$ 是恒等映射）:

$$
\text{Attn}(Q, K, V) = QK^\top V
$$

标准计算路径: 先算 $$QK^\top$$（$$O(n^2 d_k)$$），再乘以 $$V$$（$$O(n^2 d_v)$$），总计 $$O(n^2 d_k + n^2 d_v)$$。

但矩阵乘法是结合的:

$$
QK^\top V = Q(K^\top V)
$$

先算 $$K^\top V$$: $$K \in \mathbb{R}^{n \times d_k}$$ 转置后与 $$V \in \mathbb{R}^{n \times d_v}$$ 相乘，得到 $$d_k \times d_v$$ 的矩阵，代价为 $$O(n d_k d_v)$$。再乘以 $$Q$$（$$n \times d_k$$），代价为 $$O(n d_k d_v)$$。总计 $$O(2 n d_k d_v)$$——与序列长度呈**线性**关系。

> 这看似是一个平凡的代数恒等式，但其意义深远: 我们将复杂度从 $$O(n^2(d_k + d_v))$$ 降到了 $$O(n d_k d_v)$$。

这一技巧最早由 Shen et al. (2018) 提出，Katharopoulos et al. (2020) 将其推广到核函数版本。它也与 Schmidhuber 的"快速权重编程器"（Fast Weight Programmers）存在深刻的理论联系。

### 3.2 核函数视角: 从 Softmax 到特征映射

上述推导假设 $$\rho$$ 是恒等映射——但真实的注意力使用 softmax，而 softmax 是非线性的。如何将线性注意力推广到带 softmax 的情形？

**核技巧（Kernel Trick）**: 将 softmax 分解为特征映射的内积。具体而言，如果我们能找到特征映射 $$\phi: \mathbb{R}^{d_k} \to \mathbb{R}^{m}$$，使得:

$$
\text{softmax}(q_i^\top k_j) \approx \phi(q_i)^\top \phi(k_j)
$$

那么:

$$
QK^\top V \approx (\phi(Q) \phi(K)^\top) V = \phi(Q) (\phi(K)^\top V)
$$

经典选择包括:
- **Performer** (Choromanski et al., 2021): 使用正随机特征（Positive Random Features, PRF）来近似 softmax，$$\phi(x) = \frac{1}{\sqrt{m}} \exp(Wx - \|x\|^2/2)$$，其中 $$W$$ 是随机投影矩阵
- **Linear Transformer** (Katharopoulos et al., 2020): 使用 $$\phi(x) = \text{elu}(x) + 1$$ 作为特征映射

线性注意力的核心代价: 特征映射的维度 $$m$$ 需要足够大以保证近似质量（实践中 $$m \approx d_k$$ 或更大），而 $$K^\top V$$ 的中间状态大小为 $$d_k \times d_v$$——与标准注意力需要存储的 $$n \times n$$ 矩阵相比仍然小得多（当 $$n \gg d_k$$ 时）。

### 3.3 循环形式与对偶性

线性注意力在串行计算时自然地呈现为**循环神经网络（RNN）** 的形式。定义状态矩阵 $$S_t \in \mathbb{R}^{d_k \times d_v}$$:

$$
S_t = S_{t-1} + k_t v_t^\top, \quad y_t = q_t^\top S_t
$$

其中 $$k_t, v_t, q_t$$ 分别是第 $$t$$ 个 token 的 key、value、query 向量（行向量）。最终输出 $$y_t$$ 仅依赖当前 query 和累积的状态 $$S_t$$。

这揭示了线性注意力的一种深刻**对偶性**:

| 阶段 | 推荐形式 | 计算模式 | 原因 |
|------|---------|---------|------|
| **训练** | 并行形式 $$Q(K^\top V)$$ | 并行矩阵乘法 | 利用 GPU 并行性，虽然复杂度为 $$O(n d_k d_v)$$ 但可高度并行 |
| **推理（prefill）** | 并行形式 | 并行矩阵乘法 | 输入序列一次性可用 |
| **推理（decode）** | 循环形式 $$S_t = S_{t-1} + k_t v_t^\top$$ | 逐 token 串行更新 | 每次只需 $$O(d_k d_v)$$ 更新状态，$$O(d_k d_v)$$ 计算输出 |

这种对偶性使得线性注意力模型可以在训练时享受并行化的效率，而在推理时享受恒定内存和线性时间的优势。

> 注: 如果给状态更新加上衰减因子 $$\gamma$$，即 $$S_t = \gamma S_{t-1} + k_t v_t^\top$$，就得到了 **RetNet** (Sun et al., 2023) 的基本形式。

### 3.4 工业应用: MiniMax M1 与混合比例

MiniMax M1（以及 minimax-text-01）采用了 **7:1 混合线性注意力**: 每 8 个注意力层中，7 层使用线性注意力，1 层使用标准全注意力。性能表现强劲，且在上下文长度上展现出线性缩放特性。

这种"以少量全注意力保底，大量线性注意力提速"的混合策略正在成为工业界的共识范式。

---

## 四、从线性注意力到状态空间模型

### 4.1 Mamba-2: 引入门控

线性注意力的状态更新是纯加性的——旧状态与新信息同等对待。Mamba-2 (Dao & Gu, 2024) 的核心创新是为状态更新引入**数据依赖的门控**:

**线性注意力**:
$$
S_t = S_{t-1} + k_t v_t^\top, \quad y_t = q_t^\top S_t
$$

**Mamba-2**:
$$
S_t = \gamma_t S_{t-1} + k_t v_t^\top, \quad y_t = q_t^\top S_t + v_t^\top D, \quad \gamma_t = f(x_t)
$$

其中:
- $$\gamma_t$$ 是一个**输入依赖的标量门控**（由当前 token $$x_t$$ 通过一个小型神经网络 $$f$$ 计算得到），控制旧状态的保留程度
- $$D$$ 是一个可学习的对角矩阵，提供从 $$v_t$$ 到 $$y_t$$ 的直接跳跃连接（skip connection），确保即使状态更新缓慢，当前 token 的信息也不会丢失

门控的关键作用: $$\gamma_t$$ 允许模型根据输入内容**选择性遗忘**——在处理句子边界、话题转换或无关填充词时，可以衰减旧状态；在需要长程依赖时，保持 $$\gamma_t \approx 1$$ 以保留历史信息。

Mamba-2 同样保持了对偶性: 门控因子 $$\gamma_t$$ 可以在训练时并行计算，然后使用向量化扫描（parallel scan）高效更新状态。

### 4.2 Gated Delta Net: 选择性擦除

Gated Delta Net (GDN) 在 Mamba-2 的基础上进一步推广——不仅门控输入，还门控对旧状态的**选择性擦除**:

**Mamba-2**:
$$
S_t = \gamma_t S_{t-1} + k_t v_t^\top
$$

**Gated Delta Net**:
$$
S_t = \gamma_t (I - \beta_t k_t k_t^\top) S_{t-1} + \beta_t k_t v_t^\top, \quad \gamma_t = f(x_t), \quad \beta_t = f(x_t)
$$

其中:
- $$\beta_t \in [0, 1]$$ 是另一个输入依赖的门控，控制将新键 $$k_t$$ 写入状态的程度
- 当 $$\beta_t = 0$$ 时，模型执行"无输入操作"——完全保留旧状态，不写入新信息
- 擦除项 $$(I - \beta_t k_t k_t^\top)$$ 的作用是: **在写入新信息之前，先擦除状态中与当前 key 方向 $$k_t$$ 对齐的旧内容**。这实现了比简单衰减更精确的记忆管理——不是均匀遗忘，而是有针对性的覆盖

GDN 与"快速权重编程"和"测试时训练"（test-time training）的思想有密切联系——状态更新可以被解释为对联想记忆的一次在线梯度更新。

### 4.3 混合架构的工业验证

| 模型 | 混合方案 | 性能评估 |
|------|---------|---------|
| **Nemotron 3** (NVIDIA) | Mamba-Attention 约 3:1 混合 | 与同类稠密模型相当或更好的 prefill 性能 |
| **Qwen 3.5 / Qwen Next** (Alibaba) | GDN-Attention 约 3:1 混合 | 良好的推理特性，与稠密基线竞争 |

虽然缺乏大规模受控消融实验，但有限证据表明: **较低的混合比例（如 1/4 到 1/8 的全注意力层）即可实现较低的损失**，这表明大部分 token 间交互可以用线性复杂度机制有效捕获，只有少数情况需要全注意力。

---

## 五、稀疏注意力与稀疏适配

### 5.1 DSA: DeepSeek Sparse Attention

与混合注意力（修改模型架构）不同，另一条路径是**保持模型不变，但在推理时做稀疏化**——即稀疏适配（Sparse Adaptation）。

**DeepSeek Sparse Attention (DSA)** (DeepSeek V3.2, GLM-5 中使用) 的核心思想:

- 训练一个**轻量级索引器（Indexer）**，为每个 query token 预测它应该关注哪些 key token
- 在推理时，每个 token 只计算索引器选出的 top-$$k$$ 个 key 的注意力分数
- 索引器的计算开销远小于全注意力，因此总体获得显著加速

关键优势:
- **"事后适配"（Post Hoc Adaptation）**: 可以在短上下文稠密预训练完成后，再适配到长上下文——无需从头训练
- **索引器极轻量**: 通常是一个小的 MLP 或甚至一个哈希函数，计算量可忽略不计
- **与稠密模型兼容**: 已有的大规模稠密 checkpoint 可以直接利用，通过添加索引器切换到稀疏模式

### 5.2 稀疏注意力的经典模式

| 模式 | 机制 | 复杂度 | 代表工作 |
|------|------|--------|---------|
| **滑动窗口** | 每个 token 只看前后 $$w$$ 个 token | $$O(n w)$$ | Longformer, Mistral |
| **膨胀窗口** | 滑动窗口的间隔以指数增长 | $$O(n \log n)$$ | Longformer, Sparse Transformer |
| **全局+局部** | 特殊 token 看全部 + 普通 token 看局部 | $$O(n w + g n)$$ | BigBird, Longformer |
| **随机稀疏** | 每个 token 随机选择一部分 key | $$O(n k)$$ | BigBird |
| **可学习稀疏** | 训练索引器/哈希函数来选择 key | $$O(n k)$$ | DSA, Reformer (LSH), Routing Transformer |

---

## 六、混合专家模型 (MoE)

### 6.1 MoE 的基本思想

稠密 Transformer 中，每个 token 经过 FFN 层的全部参数。混合专家模型（Mixture of Experts, MoE）的核心思想是: **用稀疏激活换取更多总参数**。

将 FFN 层替换为 $$E$$ 个并行的"专家"子网络（每个专家是一个独立的 FFN），并引入一个**路由器（Router/Gate）**来决定每个 token 激活哪些专家:

$$
y = \sum_{i \in \text{top-}k(G(x))} G(x)_i \cdot E_i(x)
$$

其中 $$G(x) = \text{softmax}(x W_g)$$ 是路由器的输出（在 $$E$$ 个专家上的概率分布），$$E_i(\cdot)$$ 是第 $$i$$ 个专家的 FFN。

关键性质:
- **总参数量** $$\propto E$$，但**每个 token 的计算量** $$\propto k$$（$$k \ll E$$）
- 在训练和推理中都保持稀疏计算——可以增加专家数量而不增加单 token FLOPs
- 这一思想源于 1991 年 Jacobs et al. 的"自适应混合局部专家"和 2017 年 Shazeer et al. 的"稀疏门控 MoE"在大规模深度学习中的首次成功应用

### 6.2 为什么 MoE 正在流行

**理由一: 同样的 FLOPs，更多的参数 → 更好的性能**

当训练计算预算固定时，MoE 可以用相同的 FLOPs 训练一个总参数量大得多的模型。来自 Fedus et al. (2022) 和 OlMoE 的实验一致表明: MoE 在相同训练 FLOPs 下的 loss 显著低于稠密等价模型。

**理由二: 训练更快**

由于每个 token 只激活一小部分参数，MoE 在相同的硬件上可以达到更高的训练吞吐量——尤其是当专家分布在多设备上时，可以利用额外的并行维度。

**理由三: 与稠密模型高度竞争**

无论是西方的前沿开源模型（Mixtral 8x7B 匹配 Llama 2 70B 的性能）还是中国团队的工作（Qwen MoE、DeepSeek 系列），MoE 已成为性能最强的开源模型的主流架构选择。

**理由四: 天然适合多设备并行**

FFN 专家天然可以跨设备分布——每个设备持有若干专家，token 按路由决策在设备间传输。这引入了专家并行（Expert Parallelism），成为模型并行的又一维度。

### 6.3 MoE 的主要挑战: 路由与负载均衡

#### 6.3.1 离散路由的非可微问题

MoE 的核心操作——"选择 top-$$k$$ 专家"——是一个离散决策，不可微。训练时，我们需要对稀疏路由进行梯度估计。三类解决方案:

| 方法 | 机制 | 现状 |
|------|------|------|
| **强化学习 (REINFORCE)** | 将路由视为策略，用 REINFORCE 梯度优化 | "正确的方案"，但梯度方差大，工程复杂，未被广泛采用 |
| **随机扰动** | 在路由 logits 上施加高斯噪声或均匀乘法扰动，使路由软化 | Shazeer et al. 2017 使用高斯噪声；Fedus et al. 2022 使用乘法扰动但后被 Zoph et al. 2022 移除 |
| **启发式负载均衡损失** | 添加辅助损失项惩罚专家使用不均 | 实际中最广泛使用的方案 |

#### 6.3.2 负载均衡损失的梯度机制

大多数 MoE 使用某种形式的**辅助负载均衡损失**。以 Switch Transformer (Fedus et al., 2022) 为例，目标是鼓励每个专家处理的 token 数量尽量均衡。

损失形式为:

$$
\mathcal{L}_{\text{balance}} = \alpha \cdot N \cdot \sum_{i=1}^{E} f_i \cdot P_i
$$

其中 $$f_i$$ 是路由到专家 $$i$$ 的 token 比例，$$P_i$$ 是路由器分配给专家 $$i$$ 的平均概率。该损失对路由器概率 $$p_i(x)$$ 的导数为:

$$
\frac{\partial \mathcal{L}_{\text{balance}}}{\partial p_i(x)} \propto \frac{\alpha N}{T^2} \cdot \mathbb{1}[\text{argmax } p(x) = i]
$$

这意味着: **被更频繁选中的专家，其路由概率受到更强的向下惩罚**——一个自调节的负反馈循环。

#### 6.3.3 负载均衡的变体

**Per-Expert Balancing** (Switch Transformer, DeepSeek v1-v2): 如上所述，惩罚专家使用频率的差异。

**Per-Device Balancing** (DeepSeek v1-v2): 将上述目标按设备聚合——当多个专家分布在同一设备上时，更关心设备级别的负载均衡而非单专家级别。

**Auxiliary-Loss-Free Balancing** (DeepSeek v3): 为每个专家维护一个可学习的偏置项 $$b_i$$，通过在线学习动态调整——被过度使用的专家降低偏置，被忽略的专家提高偏置。DeepSeek 称之为"无辅助损失负载均衡"，但实际上仍使用了序列级别的辅助损失作为辅助。

**Device-level Top-M Routing** (DeepSeek v2): 在设备层面先做 top-$$M$$ 选择，再在设备内做专家级别的 top-$$k$$ 选择，减少跨设备通信。

### 6.4 路由类型详解

#### 6.4.1 Token Choice Top-$$k$$（主流方案）

几乎所有现代 MoE 都使用"token 选择专家"的 top-$$k$$ 路由:

- 路由器为每个 token 输出在所有专家上的分数
- 选择分数最高的 $$k$$ 个专家
- 被选中的专家处理该 token，输出按路由权重加权求和

不同模型的 $$k$$ 值:

| 模型 | $$k$$ | 备注 |
|------|-------|------|
| Switch Transformer | 1 | 极简设计 |
| GShard | 2 | 早期大规模 MoE |
| Mixtral 8x7B | 2 | 开源标杆 |
| Grok | 2 | xAI |
| DBRX | 4 | Databricks |
| Qwen MoE | 4 | 阿里 |
| DeepSeek v1-v2 | 6 | 含共享专家 |
| DeepSeek v3 | 8 | 含共享专家 |
| OlMoE | 8 | AI2 开源 |

路由器的两种软最大变体:
- **DeepSeek v1-v2 / Grok / Qwen**: sigmoid 门控 ——每个专家独立打分，选取 top-$$k$$（更灵活，允许不同专家有不同的激活阈值）
- **Mixtral / DBRX / DeepSeek v3**: softmax 后取 top-$$k$$ ——概率分布归一化（更标准的概率解释，但与 sigmoid 在实践中差异不大）

#### 6.4.2 其他路由方法

- **哈希路由**: 使用哈希函数将 token 映射到专家——简单、确定性强，常用作基线
- **强化学习路由**: Bengio et al. (2013) 的早期工作使用 RL 学习路由策略，但因工程复杂度未被广泛继承
- **线性指派路由**: Clark et al. (2022) 将路由建模为线性指派问题（Linear Assignment），保证负载绝对均衡，但计算开销较大

### 6.5 专家的设计

#### 6.5.1 细粒度专家 (Fine-Grained Experts)

DeepSeek MoE 提出的关键创新——将标准的"大"专家拆分为更多、更小的"细粒度"专家:

- 标准 MoE: 每个专家是一个完整的 FFN（例如 8 个专家）
- 细粒度 MoE: 将每个大专家拆分为 $$r$$ 个小专家（例如 64 个细粒度专家 = 8 个标准专家 $$\times$$ 8 倍拆分）

细粒度专家比例（Fine-Grained Ratio）定义为: 细粒度专家的参数量 / 标准专家的参数量。例如 DeepSeek v1 的比例为 1/4，意味着 64 个细粒度专家总共拥有相当于 16 个标准专家的参数量。

**优势**: 更灵活的专家组合——模型可以对不同 token 组合更多样的专家子集，提高参数利用效率。OlMoE 的消融实验确认了细粒度专家带来的增益，但对共享专家的增益持保留态度。

#### 6.5.2 共享专家 (Shared Experts)

另一个重要设计是**始终激活的共享专家**——无论路由决策如何，某些专家对所有 token 都生效。这保证了即使路由失败（例如对所有专家都产生低置信度），token 仍能获得基本的 FFN 计算。

DeepSeek v1-v2 使用 2 个共享专家，DeepSeek v3 精简到 1 个。Qwen MoE 使用 4 个共享专家。

#### 6.5.3 专家路由配置总览

| 模型 | 总专家数 | 激活专家 | 共享专家 | 细粒度比例 |
|------|---------|---------|---------|-----------|
| GShard | 2048 | 2 | 0 | — |
| Switch Transformer | 64 | 1 | 0 | — |
| ST-MoE | 64 | 2 | 0 | — |
| Mixtral 8x7B | 8 | 2 | 0 | — |
| DBRX | 16 | 4 | 0 | — |
| Grok | 8 | 2 | 0 | — |
| DeepSeek v1 | 64 | 6 | 2 | 1/4 |
| Qwen 1.5 MoE | 60 | 4 | 4 | 1/8 |
| DeepSeek v3 | 256 | 8 | 1 | 1/14 |
| OlMoE | 64 | 8 | 0 | 1/8 |
| MiniMax | 32 | 2 | 0 | ~1/4 |
| Llama 4 (Maverick) | 128 | 1 | 1 | 1/2 |

### 6.6 MoE 的训练稳定性

#### 6.6.1 数值稳定性问题

Zoph et al. (2022) 发现 MoE 训练中路由器容易出现数值不稳定——路由 logits 的剧烈波动导致专家选择频繁切换，进而引发训练崩溃。解决方案是: **对路由器使用 Float32 精度**（即使模型其余部分使用 BF16/FP16），并可选地添加 **Z-loss** 以抑制过大的 logits。

Z-loss 的形式为:

$$
\mathcal{L}_z = \lambda \cdot \frac{1}{B} \sum_{b=1}^{B} \left(\log \sum_{e=1}^{E} \exp(z_{b, e})\right)^2
$$

它惩罚 log-sum-exp 的大值，本质上是防止路由器 logits 的绝对值过大。

#### 6.6.2 Token 丢弃的随机性

MoE 中存在一个微妙的随机性来源: 当专家容量（expert capacity）被设限时，超额的 token 会**按批次随机丢弃**。这意味着同一批次中**其他用户的请求可能影响你的 token 是否被丢弃**——这在多租户推理服务中是一个潜在的公允性问题。

#### 6.6.3 微调中的过拟合

稀疏 MoE 在小规模微调数据上更容易过拟合——因为每个专家只看到数据的子集，有效训练样本被稀疏路由稀释。两种解决方案:

- **Zoph et al. 的方案**: 微调时将 MoE 层替换为稠密 MLP（对非 MoE 部分进行微调）
- **DeepSeek 的方案**: 使用海量 SFT 数据（1.4M 条），确保每个专家都有足够的训练信号

### 6.7 MoE 的系统侧考量

**并行化**: MoE 层天然适合专家并行——每个设备持有一组专家，token 通过 all-to-all 通信在设备间路由。DeepSpeed-MoE 和 MegaBlocks 等库使用高效的稀疏矩阵乘法来实现路由计算，避免了传统 scatter/gather 的通信开销。

**通信压缩**: Nemotron 3 提出了"下投影激活"（down-projecting activations）的策略——在跨设备传输前先压缩激活维度，减少 all-to-all 通信量。

**专家选择 vs Token 选择的系统影响**: Token choice（token 选专家）实现更简单但可能导致负载不均；expert choice（专家选 token）能保证负载绝对均衡但实现更复杂。主流实施全部采用 token choice。

---

## 七、MoE 前沿架构: DeepSeek MoE 三代演进

### 7.1 DeepSeek MoE v1 (16B 总参数, 2.8B 激活)

**架构**: 2 个共享专家 + 64 个细粒度专家（4 倍拆分），top-6 路由。

**路由**: 标准 top-$$k$$ token choice 路由 + sigmoid 门控。

**负载均衡**: 专家级别的辅助损失 + 设备级别的辅助损失。

这是 DeepSeek 将细粒度专家 + 共享专家范式首次大规模验证的版本。

### 7.2 DeepSeek MoE v2 (236B 总参数, 21B 激活)

**架构**: 2 个共享专家 + 160 个细粒度专家（10 倍拆分），top-6 路由（6 active）。

**新增技术**:
- **Top-M Device Routing**: 先在设备级做 top-$$M$$ 选择，再在设备内做专家级 top-$$k$$ 选择——减少跨设备通信
- **通信均衡损失**: 除了均衡专家使用，还加入对通信流入和流出量的均衡约束

### 7.3 DeepSeek MoE v3 (671B 总参数, 37B 激活)

**架构**: 1 个共享专家 + 256 个细粒度专家（14 倍拆分），top-8 路由。

**新增技术**:
- **Sigmoid + Softmax 双重路由**: 先用 sigmoid 为每个专家独立打分，然后在选中的专家子集上做 softmax 归一化——兼顾灵活性和概率解释
- **无辅助损失负载均衡 (Aux-Loss-Free)**: 使用可学习偏置的在线学习方案，减少对辅助损失的依赖（虽然未完全消除）
- **序列级辅助损失**: 辅助损失在序列级别而非 token 级别计算

### 7.4 并行创新: MLA (Multihead Latent Attention)

DeepSeek v3 使用了**多头潜在注意力 (MLA)** 作为注意力层的配套优化:

**核心思想**: 将 $$Q, K, V$$ 表示为低维"潜在"激活的函数。具体而言，引入一个压缩的潜在向量 $$c_t^{KV} \in \mathbb{R}^{d_l}$$（$$d_l \ll d_k \cdot h$$），通过上投影矩阵生成 keys 和 values。

**KV 缓存收益**: 在推理时，只需存储潜在向量 $$c_t^{KV}$$ 而非完整的 keys 和 values——当 $$d_l \ll d_k \cdot h$$ 时，KV 缓存大幅缩减。

**RoPE 冲突与解决**: RoPE（旋转位置编码）要求对 key 向量施加位置依赖的旋转——但这与 MLA 的"从潜在向量上投影生成 key"存在冲突。DeepSeek 的解决方案: 保留少量非潜在（直接可旋转的）key 维度专门用于 RoPE，其余维度通过潜在向量生成。

### 7.5 并行创新: MTP (Multi-Token Prediction)

DeepSeek v3 还引入了**多 token 预测 (MTP)**:

**思想**: 在训练时让模型同时预测接下来的多个 token（而不仅仅是一个），而非在推理时改变生成方式。

**实现**: 添加小型、轻量的"预测头"模块——每个未来的 token 位置有一个独立的预测头，结构简单、参数量少。DeepSeek v3 只做了超前一个 token 的 MTP。

**收益**: MTP 作为一种辅助训练目标，提升了模型对长程规划的能力和最终性能。这与 Meta 的 EAGLE 工作在思路上一致——EAGLE 也探索了多 token 预测对模型质量的提升。

> DeepSeek v3 的完整架构 = 细粒度 MoE + 共享专家 + 无辅助损失路由 + MLA + MTP。这是一个各组件高度协同的复杂系统。

---

## 八、上循环: 从稠密模型到 MoE

上循环（Upcycling）是指利用预训练好的稠密模型来初始化 MoE 模型，从而大幅降低 MoE 的训练成本:

1. 取一个已训练好的稠密 checkpoint
2. 复制 FFN 层为多个专家（可选的参数扰动以增加多样性）
3. 添加路由器（随机初始化）
4. 用相对较少的 token 继续训练，让路由器学会路由，专家学会专业化

**代表性案例**:
- **MiniCPM**: 基于 MiniCPM 稠密模型，8 个专家，top-$$k=2$$，约 4B 激活参数。仅用约 520B token 的训练就从基模型获得了显著增益
- **Qwen MoE**: 基于 Qwen 1.8B 稠密模型，60 个专家 + 4 个共享专家，top-$$k=4$$。架构与 DeepSeekMoE 高度相似，是最早确认上循环成功的大规模案例之一

上循环的经济意义: 它使得在稠密模型上积累的训练投入可以部分迁移到 MoE 架构，加速 MoE 的研发迭代。

---

## 九、方案对比: 如何选择注意力替代方案

| 方案 | 复杂度 | 训练兼容 | 推理优势 | 主要代价 | 适用场景 |
|------|--------|---------|---------|---------|---------|
| **全注意力（基线）** | $$O(n^2)$$ | 最成熟 | KV 缓存 $$O(n)$$ | 计算成本随 $$n$$ 平方增长 | 短-中上下文（$$n < 8\text{K}$$） |
| **滑动窗口稀疏注意力** | $$O(n w)$$ | 与稠密训练兼容 | 恒定窗口内存 | 丧失超窗口长程依赖 | 局部模式为主的任务 |
| **全局+局部混合注意力** | $$O(n w + g n)$$ | 与稠密训练兼容 | 全局 token 保持长程连接 | 仍需选择全局 token 策略 | 文档理解、摘要 |
| **DSA (稀疏适配)** | $$O(n k)$$ | 事后适配，无需重训 | 极大加速长上下文推理 | 索引器需要训练 | 已有稠密模型的长上下文迁移 |
| **线性注意力** | $$O(n d_k d_v)$$ | 需要重训（softmax 被去除） | 恒定状态$$O(d_k d_v)$$，无 KV 缓存 | 近似质量 vs $$d_k d_v$$ 维度的权衡 | 极端长上下文（$$n > 100\text{K}$$） |
| **Mamba-2 / SSM** | $$O(n d_k d_v)$$ | 需要重训 | 与线性注意力类似，门控增强表达力 | 需要专门的扫描算子实现 | 长序列 + 需要选择性记忆 |
| **Gated Delta Net** | $$O(n d_k d_v)$$ | 需要重训 | 与 Mamba-2 类似，擦除能力更强 | 额外的门控参数和计算 | 需要精细记忆管理 |
| **混合注意力 + 线性/SSM** | 混合 | 部分重训 | 兼顾质量和效率 | 架构复杂度 | 通用长上下文 LLM（当前工业共识） |
| **MoE (稠密注意力)** | $$O(n^2)$$ 注意力 + 稀疏 FFN | 需要重训（FFN 替换） | 参数大、计算少 | 显存大（全部参数）、通信复杂 | 大规模训练，参数高效推理 |
| **MoE + 线性/稀疏注意力** | $$O(n d_k d_v)$$ 注意力 + 稀疏 FFN | 需要重训 | 双维度高效 | 架构最复杂 | 极致效率的下一代 LLM（如 DeepSeek v3） |

### 选择决策树

1. **上下文长度** $$n < 8\text{K}$$: 全注意力 + MoE（如果需要大规模训练）
2. **上下文长度** $$n \in [8\text{K}, 100\text{K}]$$: 混合注意力（全局+局部）或 DSA 稀疏适配
3. **上下文长度** $$n > 100\text{K}$$: 线性注意力或 SSM 混合架构（如 Mamba-2/Attention 混合）
4. **追求训练 FLOPS 效率**: MoE ——用相同的计算量训练更大的模型
5. **已有稠密模型，想适配长上下文**: DSA 或上循环 MoE
6. **最前沿的极致设计**: DeepSeek v3 范式——MoE + MLA + MTP

---

## 十、本章总结

本章涵盖了两条应对 Transformer 注意力瓶颈的技术路线:

**路线一: 降低注意力成本**
- 稀疏注意力（滑动窗口、膨胀、全局+局部、DSA）通过限制每个 token 关注的 key 数量来降复杂度
- 线性注意力通过交换矩阵乘法顺序（$$QK^\top V \to Q(K^\top V)$$）从 $$O(n^2)$$ 降为 $$O(n)$$
- 状态空间模型（Mamba-2、Gated Delta Net）在线性注意力的基础上引入输入依赖的门控和选择性擦除，显著提升表达力
- 混合架构（少量全注意力 + 大量线性/SSM 层）是当前工业界的实用共识

**路线二: 用稀疏激活换容量**
- MoE 将 FFN 替换为多个专家 + 路由器，每个 token 只激活少数专家
- Top-$$k$$ token choice 路由 + 辅助负载均衡损失是主流训练范式
- 细粒度专家和共享专家进一步提升了参数利用效率
- DeepSeek MoE v3 代表了当前 MoE 架构的集成巅峰: 细粒度 MoE + 无辅助损失路由 + MLA + MTP

**两条路线的融合**: 前沿模型（如 DeepSeek v3）正在同时利用两条路线——MLA 降低注意力成本，MoE 提升参数容量——这代表了下一代高效 LLM 的架构方向。

> MoE 利用稀疏性——并非所有输入都需要完整的模型。离散路由很难，但 top-$$k$$ 启发式在实践中有效。大量经验证据表明: MoE 行之有效，且成本效率高。

---

<p style="text-align:right; color:#888; font-style:italic;">
译自 CS 336 Lecture 4: Attention Alternatives and Mixture of Experts.<br>
Tatsu Hashimoto, Stanford University, Spring 2025.
</p>
