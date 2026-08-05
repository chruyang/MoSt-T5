# 10 两个核心问题：3D Loss 与 motif 划分

> 审查日期：2026-07-17  
> 目标：判断当前方案在理论上是否成立，并把“已有文献支持”与“必须由本项目实验回答”分开。

## 1. 先修正整体思路的科学表述

当前方案更准确的表述是：

> 将自然语言编码为文本 token，将分子二维化学身份和连接拓扑编码为 motif token；再用 E3FP 将每个原子的化学身份、局部拓扑和构象相关三维邻域编码成多 shell 的 3D-aware token。通过 atom-to-motif 映射，把原子级 E3FP 表示局部聚合到 motif，形成带三维上下文的 motif 表示。训练时使用联合 2D+3D 遮蔽和 3D-only 遮蔽，使模型分别学习结构恢复与从二维语义/分子上下文推断缺失三维信息。

需要避免原来的两个过强表述：

1. **E3FP 不是纯粹的“3D 状态 token”**。它的初始原子不变量包含原子序数、价态、连接邻居、环信息等二维化学身份，后续 shell 又同时编码连接关系和三维相对取向。因此它是“3D-aware 局部结构 token”，不是与 2D 身份正交的纯几何通道。
2. **聚合后不是严格封闭的“motif 内部 3D 状态”**。一个原子的 E3FP shell 可以覆盖 motif 外部的邻近原子，所以聚合结果更准确地说是“以 motif 为中心的局部三维上下文”。

这两个区别会直接影响 mask 和 3D Loss 的解释。

## 2. 问题一：T5 加入 3D Loss 是否合适

### 2.1 架构层面：可以加入，不违反 T5

T5 只是 encoder-decoder 主干及 text-to-text 训练范式，并不要求总损失只能是 token cross-entropy。只要辅助目标通过 encoder hidden state 反向传播，形式上完全可以使用：

\[
L = L_{CE} + \lambda_{3D}L_{3D}.
\]

文献中也存在相邻依据：

- GraphMVP 使用 2D/3D 对齐和连续表示空间重建；
- C-FREE 使用 MSE 预测分子子图的多模态 latent target；
- GradNorm、PCGrad 等多任务学习方法专门处理多个目标的尺度和梯度冲突。

因此，真正的问题不是“能否加”，而是以下四点：

1. 3D target 是否真正代表希望学习的几何信息；
2. target 是否稳定、是否可能塌缩；
3. CE 与 3D Loss 的梯度是否同向、量级是否平衡；
4. 增加的目标是否在独立验证集和下游任务上带来增益。

### 2.2 3D-MolT5 不能直接证明当前 3D Loss

3D-MolT5 直接支持“把 E3FP 当作 T5 的三维输入 token”。但该论文明确说明其下游训练仍使用标准 cross-entropy，而且没有在 denoising 中预测 E3FP 的多分量 token。因此：

- 它能支持我们的 **E3FP 输入与 1D/3D 对齐方向**；
- 它不能直接支持当前的 **pooled E3FP latent MSE**；
- 当前 MSE 应被视为项目自己的研究假设。

### 2.3 当前代码的 3D Loss 实际监督内容

当前实现位于 `model/modeling.py:264-345`：

1. 用未遮蔽 E3FP ID 通过当前 `e3fp_embeddings` 得到原子表示；
2. 用当前 `q/k/v` attention 把原子表示聚合到 motif；
3. 对聚合结果执行 `detach`，作为 target；
4. 用 encoder hidden state 经 `geometric_head` 预测该 target；
5. 在 `mask_positions` 上计算 MSE；
6. Phase 1 使用 `CE + 500 × MSE`。

这不是坐标重建、距离重建或 E3FP identifier 分类，而是**预测同一模型当前产生的内部三维 latent**。

### 2.4 当前实现的主要风险

#### 风险 A：target 是移动的，而且没有独立 teacher

target 依赖正在更新的：

- 四层 E3FP embedding；
- `q_proj/k_proj/v_proj`；
- motif/sentinel embedding。

`detach` 只阻止本次反向传播进入 target 分支，并不使跨 step 的 target 固定。C-FREE 使用独立 target encoder，并用 EMA 更新以稳定目标、降低 representation collapse 风险；当前代码没有这一机制。

结论：**stop-gradient 是必要的保护之一，但不是目标稳定性的充分条件。**

#### 风险 B：target 不是纯几何 target

`compute_pooled_3d` 的 query 来自 `input_ids` 对应的 motif embedding：

- 若 motif 身份被遮蔽，query 是 sentinel embedding；
- 若仅 3D 被遮蔽，query 是真实 motif embedding。

所以同一个 3D head 同时拟合两类不同条件下形成的 target。target 既依赖 E3FP，也依赖 2D token/sentinel 的 query，不能严格解释成“真实几何状态”。

#### 风险 C：存在模态泄漏

E3FP 本身包含原子身份和拓扑，所以即使 motif token 被遮蔽，周围 E3FP 仍可能透露 2D 身份。反过来，即使某 motif 对应原子的 E3FP 被置空，相邻 motif 原子的较大 E3FP shell 仍可能编码被遮蔽原子的空间信息。

因此当前两个操作实质上是：

- 联合 motif+E3FP mask；
- E3FP-only mask。

它们并没有形成严格独立、无泄漏的“2D mask”和“3D mask”。

#### 风险 D：MSE 有平凡缩放/塌缩路径

如果 E3FP embedding、投影或 gate 的尺度整体缩小，target 和 prediction 都可能变小。CE 会提供一定约束，但 3D MSE 本身不保证 latent 保留有区分力的几何信息。应监测：

- target/prediction 各维标准差；
- batch covariance 的有效秩；
- 不同分子/构象间 cosine 分布；
- gate 是否普遍趋近 0；
- E3FP embedding 范数是否持续缩小。

#### 风险 E：`lambda_3d=500` 没有尺度依据

原始 MSE 数值小不代表其梯度小。固定 500 只有在比较共享参数上的梯度范数后才可解释，否则可能出现：

- 3D Loss 主导，破坏语言/分子 token 建模；
- CE 主导，3D Loss 实际不起作用；
- 两个梯度方向冲突，训练损失下降但下游性能变差。

### 2.5 E3FP 是否适合作为几何状态

#### 适合的场景

E3FP 适合作为：

- 原子中心局部三维环境的离散描述；
- 对全局平移、旋转和坐标系对齐不敏感的输入；
- 能与 motif/序列模型结合的有限词表 token；
- 大规模预训练中比直接坐标建模更便宜的三维近似。

E3FP 原始论文把它定义为快速、alignment-invariant 的 3D conformer representation；3D-MolT5进一步证明其可以转成与语言模型兼容的原子级三维 token。

#### 不适合被宣称为

E3FP 不应被宣称为：

- 纯几何、与原子身份无关的状态；
- 完整可逆的三维坐标表示；
- 能精确表达距离、键角、二面角、能量和构象概率的物理表示；
- 单一构象下对真实溶液/结合构象的无偏代表。

它存在哈希碰撞、离散化信息损失和构象依赖。E3FP 原始研究对一个分子的多个构象分别计算指纹；3D-MolT5 也专门分析了不同构象 token 与离散化损失。当前只生成一个固定 seed 的廉价构象，适合做低成本 baseline，不足以证明模型获得了可靠的“真实三维状态”。

### 2.6 三条可选路线

#### 路线 A：最稳妥基线

保留 E3FP 作为输入，只使用标准 CE，复现 3D-MolT5 类逻辑：

\[
L=L_{CE}.
\]

用途：先回答“加入 E3FP 输入本身是否有用”。这是所有 3D Loss 实验必须比较的基线。

#### 路线 B：稳定 latent prediction

保留 MSE，但把 target 改成：

- 独立且短期冻结的 target encoder；或
- EMA teacher；或
- 预先训练并冻结的 3D encoder 输出。

同时对 target/prediction 做 LayerNorm 或 L2 normalization，并比较 MSE、cosine loss。C-FREE 是比 SimSiam 更接近当前分子任务的直接参考。

#### 路线 C：让 3D 目标具有明确物理语义

可预测：

- 各 shell 的 E3FP identifier 分类；
- motif 内/邻域原子距离 bin；
- 键角或二面角 bin；
- 原子对距离矩阵；
- 由 PaiNN/Uni-Mol 等三维编码器生成的冻结 target。

如果研究目标是“精确几何状态”，优先考虑距离/角度或 E(3)-equivariant encoder；如果目标是“让 T5 感知三维局部环境”，E3FP 已足够作为第一版离散表示。

### 2.7 推荐的 3D Loss 最小实验

固定数据、模型规模和训练步数，按顺序比较：

| 实验 | 输入 | Loss | 回答的问题 |
|---|---|---|---|
| L0 | motif only | CE | 纯 2D 基线 |
| L1 | motif + E3FP | CE | E3FP 输入是否有贡献 |
| L2 | motif + E3FP | CE + 当前 latent MSE | 当前 3D Loss 是否超过 L1 |
| L3 | motif + E3FP | CE + EMA-target MSE | 稳定 target 是否更好 |
| L4 | motif + E3FP | CE + 明确几何/ID 目标 | 物理/离散监督是否优于 latent 自预测 |

每组至少报告：CE、3D loss、两者共享参数梯度范数、梯度 cosine、latent 方差/秩、gate 分布、验证集 MLM、caption/text2mol、构象敏感下游指标。

在获得这些结果前，不建议围绕 500 微调；先测试 `0/1/10/100/500`，或使用 GradNorm 做动态平衡。

## 3. 问题二：当前 motif 划分是否合理

### 3.1 当前算法是什么

`model/CAMT5/representation.py:72-207` 的规则是：

1. 读取所有环；
2. 读取所有非单键及两端原子；
3. 把存在共享原子的环/非单键集合不断合并；
4. 未进入上述集合的原子各自成为单原子 motif；
5. motif 间的键用成对 anchor 表示；
6. 对 motif 图做 DFS，线性化为 token 序列。

这与 CAMT5 论文的核心定义高度一致：环原子和非单键连接的原子被看作刚性较强、能表达共振/结构上下文的 motif，其余原子为单 token；论文还报告 motif tokenization 和 DFS 相对 atom-wise/BFS 的实验收益。

所以结论不是“不合理”，而是：

> **作为 CAMT5 风格的二维序列化 baseline，它有直接文献依据；作为最适合三维聚合、性质预测和立体化学的 motif 定义，目前证据不足。**

### 3.2 它的优点

1. 规则确定、计算便宜，适合大规模预处理。
2. 环和共轭/多键区域通常比任意原子 token 更有结构语义。
3. 将大图压缩为更短的 motif 序列。
4. anchor 能保留 motif 间连接关系，原则上可做 2D 图 round-trip。
5. atom-to-motif 是不重叠分区，便于把原子 E3FP 聚合到唯一 motif。

### 3.3 对当前研究目标的不足

#### 不足 A：这是“刚性连接 motif”，不等于完整化学功能团

例如酰胺中的 `C=O` 会成为 motif，但通过单键连接的 N 可能成为另一个 motif；磺酰胺中的 `S(=O)2` 与 N 也可能被拆开。因此它不一定保持药物化学中完整的 amide、sulfonamide 等功能团。

如果论文中称其为“功能团 token”，表述会过强；称为“ring/non-single-bond structural motif”更准确。

#### 不足 B：粒度高度不均匀

- 柔性烷基链可能产生大量单原子 motif；
- 稠合环、多环和连续共轭体系可能合并为很大的 motif；
- 大 motif 更容易成为低频或 OOV token；
- 3D attention 聚合的原子数差异很大，表示方差和训练难度不一致。

#### 不足 C：当前代码主动删除立体化学

`Chem.RemoveStereochemistry(mol)` 使对映体/非对映体可能获得相同 2D motif 序列。现有重建验证脚本也先删除立体化学再比较，所以“100% 重建”即使成立，也只证明无立体二维拓扑可重建。

后果：

- text2mol 在没有 3D 输入时无法可靠生成指定立体异构体；
- 2D identity 与 E3FP 3D channel 的职责被人为不对称分割；
- 一旦构象生成失败或 E3FP 缺失，立体身份完全丢失。

这是当前 motif 实现最需要重新决定的科学边界。

#### 不足 D：E3FP 与 motif 边界不一致

motif 按 2D 环/键划分，E3FP 按三维半径 shell 扩展。一个 motif 内原子的 E3FP 会包含 motif 外原子，而相邻 motif 的 E3FP 也可能覆盖当前 motif。因此：

- 聚合表示不是 motif 内部几何；
- 3D mask 可能通过邻居 shell 泄漏；
- 较大 shell 下，motif 划分对“局部性”的约束逐渐减弱。

这不是算法错误，但必须在方法定义和实验中量化。

#### 不足 E：不存在对所有任务都最优的唯一划分

T-SMILES 系统比较 JT-VAE、BRICS、MMPA、Scaffold 等多种 fragmentation，并指出不同表示可互补。JT-VAE 使用化学子结构构成 junction tree；BRICS 更偏向可合成连接；Bemis-Murcko 更偏向骨架。当前 CAMT5 划分针对文本到分子序列建模有效，不代表对性质预测、构象建模和 3D 聚合同样最优。

### 3.4 如何严谨判断 motif 是否合理

#### 第一层：表示正确性

必须报告：

- isomeric SMILES round-trip，而不是去立体后 round-trip；
- 原子、键、键级、芳香性、形式电荷和手性保持率；
- atom mapping 覆盖率、重复率和越界率；
- anchor 成对率及超过 100 个 anchor 的样本数。

#### 第二层：粒度与词表质量

- 每分子 motif 数、平均/最大 motif 原子数；
- 单原子 motif 比例；
- 超大 motif 比例；
- vocabulary coverage、UNK、频率长尾；
- motif 序列相对 SMILES/SELFIES 的长度压缩率；
- 不同数据集间 motif 分布漂移。

#### 第三层：化学和几何一致性

- 标准功能团被完整保留的比例；
- 可旋转键位于 motif 内还是 motif 间；
- 同一 motif 在多构象下的内部几何方差；
- E3FP shell 越过 motif 边界的比例；
- mask 后从邻居 E3FP 仍能恢复被遮蔽原子身份的泄漏率。

#### 第四层：任务消融

至少比较：

| 编号 | 划分方式 | 目的 |
|---|---|---|
| M0 | atom/SELFIES | 无 motif 基线 |
| M1 | 当前 CAMT5 划分 | 已有方法基线 |
| M2 | CAMT5 + 保留立体化学 | 判断去立体化代价 |
| M3 | BRICS 或功能团保护的 hybrid | 判断化学功能团/可合成片段是否更好 |
| M4 | JT-VAE/ring-clique 类划分 | 判断图生成型子结构是否更适合 |

保持模型、词表预算、数据和训练 token 数一致，比较 CE、有效性、重建、caption/text2mol、MoleculeNet 和构象敏感任务。

## 4. 当前建议

1. **保留当前 motif 作为基线，不立即推翻。** 它与 CAMT5 高度一致，有直接文献依据。
2. **将“3D 状态”改称“E3FP 3D-aware 局部结构状态”。** 这更符合 E3FP 的真实信息组成。
3. **将两个 mask 改称“2D+3D 联合 mask”和“3D-only mask”。** 当前没有严格独立的 2D-only mask。
4. **先做 CE-only 的 E3FP 输入基线。** 只有超过该基线，3D Loss 才有存在价值。
5. **当前 latent MSE 至少增加 EMA/frozen target 对照。** 不建议仅凭 `detach` 和 `lambda=500` 继续扩大训练。
6. **建立 stereo-preserving motif 版本。** 当前去立体化与“3D 状态建模”目标存在明显张力。
7. **量化跨 motif E3FP 泄漏。** 这是两个 mask 能否被正确解释的关键实验。

## 5. 最终判断

| 问题 | 判断 |
|---|---|
| T5 能否加入 3D Loss | 可以，架构上合理 |
| 当前 3D Loss 是否已有充分文献证明 | 没有；属于待验证的 latent self-prediction 方案 |
| 当前 `detach` 是否足够稳定 target | 证据不足，建议 EMA/frozen teacher 对照 |
| E3FP 是否适合当前方案 | 适合做可扩展、离散、局部、3D-aware 输入；不适合被当作纯几何或精确物理状态 |
| 当前 motif 划分是否合理 | 作为 CAMT5-style baseline 合理；作为 3D motif 的最优划分尚未证明 |
| 当前最明显的 motif 风险 | 去除立体化学、粒度不均、功能团可能被拆分、E3FP 跨边界泄漏 |

## 6. 主要文献

- Axen et al. (2017), *A Simple Representation of Three-Dimensional Molecular Structure*, DOI `10.1021/acs.jmedchem.7b00696`。
- Pei et al. (2025), *3D-MolT5*, ICLR 2025，本地 `paper/3dMolt5.pdf`。
- Kim et al. (2025), *CAMT5*, EMNLP Findings 2025，本地 `paper/CAMT5.pdf`。
- Ariguib et al., *C-FREE*, 本地 `paper/C-FREE.pdf`。
- Wu et al. (2024), *t-SMILES*, Nature Communications，本地 `paper/t-smiles.pdf`。
- Jin et al. (2018), *Junction Tree Variational Autoencoder for Molecular Graph Generation*, ICML 2018。
- Chen et al. (2018), *GradNorm*, ICML 2018。
- Yu et al. (2020), *Gradient Surgery for Multi-Task Learning*, NeurIPS 2020。

