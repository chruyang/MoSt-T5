# G3：先验证构象关系，再进行 motif–T5 对齐（2026-08-08）

> 执行更新：G3a 已按本协议完成并未通过 identity-disjoint 泛化门；不启动 G3b。
> 结果与后续连续几何路线见文档 64。下文保留为预注册协议与文献依据。

## 1. 为什么 G2 之后不能继续只调注入层

G2 同时给出三条证据：

1. geometry 格的生成 CE 和 accuracy 优于 topology 格；
2. 将匹配 E3FP 换成同原子数的错配 E3FP，NLL 只增加 `0.002140`；
3. 将 geometry bridge projection 整体置零，NLL 反而改善 `0.002377`。

因此问题不是“注入系数不够大”，而是普通 motif 重建 CE 没有要求隐状态保留
“这个 motif 对应这个构象状态”的关系。继续调 scale、gate 或残差形式不能直接修复
这个目标缺口。

## 2. FACET 提供的机制证据与边界

FACET（ICLR 2026）是高价值的相邻机制参考，但不是 MoSt-T5 的直接背书。

它的关键做法是：

- 用 RingPath 建立 ring/path/junction fragment、atom–fragment incidence 和 fragment graph；
- 将 fragment 表示加入 atom 表示，并由 fragment-derived 2D 表示构造 3D Transformer
  的 attention bias；
- 训练 graph Transformer 使潜空间构象距离逼近 FGW 距离：

  `L_enc = Σ_ij | ||T(H_i)-T(H_j)||²₂ - FGW(G(S_i), G(S_j)) |`；

- 单独训练上述关系模块，冻结后通过 adapter 与任务模型连接，再训练上下游编码器。

FACET 的 Figure 2 报告 learned distance 与 FGW 的 Pearson 相关约为
`0.812–0.964`；Table 5 的 fragment 消融总体支持 fragment-aware 建模，但提升并非
所有数据集都很大。它的限制也很明确：预先给定多构象、小分子性质预测，不涉及
T5、E3FP、无损 motif 语言或文本生成。

两处口径必须保持准确：

1. FACET 原目标是平方潜距离与 FGW 的绝对误差；下面的 Huber、局部距离矩阵和
   排序损失都是本项目的轻量改造，不能写成 FACET 原实现。
2. FACET 的阶段不是简单的“先训练几何模块、再训练 T5”。它先训练 2D/3D MPNN，
   再单独训练 FGW graph Transformer，最后冻结后通过 adapter 接入任务模型。

## 3. 现有 G1 证据为什么还不够

G1 已经证明：

- level 1/2 state 可预测；
- aligned E3FP 明显优于 shuffled E3FP；
- 128 个分子的 711 个构象对全部改变 G1 表示，刚体变换保持不变。

但 G1 表示距离与构象 RMSD 的 Pearson 只有：

- 全部 level：`0.20534`；
- 去掉 level 3：`0.17338`。

C0 中 raw E3FP atom-change-rate 与 RMSD 的 Spearman 也只有 `0.135`。所以当前结论
只能是“离散状态对构象敏感”，不能说“其欧氏距离表达了构象差异”。这正是 G2
可能把 geometry 当作统计偏置、而不使用对应构象的上游原因。

## 4. G3a：构象关系保持门（先执行）

### 4.1 科学问题

在固定 2D 分子身份、固定原子对应和固定 motif 划分的条件下，能否学习一个
motif-topology-aware 几何表示，使其距离随真实构象差异变化？

这一设计天然削弱元素组成、分子大小和 motif 身份捷径，因为正负关系都来自同一
分子的不同构象。

### 4.2 数据与切分

- 复用 C0 已冻结的 PF-1 train identity 候选；
- 每个分子生成 4 个 ETKDGv3 构象，保留现有氢投影、E3FP inheritance 和 frozen
  motif 分区；
- 第一轮使用约 1,000 个分子，约 6,000 个构象对；
- 以 molecule identity 做 train/dev 切分，禁止同一分子的构象跨 split；
- 该数据只来自 stage-1 train，不接触 PF-1 dev 或下游标签。

### 4.3 几何关系目标

由于同一分子构象之间有明确原子对应，不需要直接移植完整 FGW。主目标采用重原子
两两距离矩阵差：

`d_geo(c1,c2) = RMS_{i<j}( D_c1[i,j] - D_c2[i,j] )`。

它具有平移/旋转不变性，不需要 Kabsch 对齐，也不会把 4096 个 folded E3FP ID
误当作连续数值。训练目标采用：

`L_rel = SmoothL1( ||z_c1-z_c2||₂, d_geo(c1,c2) )`。

Huber/SmoothL1 是为降低极端构象对回归的影响而做的项目适配；同时报告 Pearson、
Spearman 和构象对排序准确率，避免只看训练损失。

### 4.4 表示结构

1. frozen G1b 对各 motif 产生 state embedding；
2. 以 GraphPorts 已有 motif 邻接构造一个很小的两层 message-passing adapter；
3. 对 motif 表示做置换不变池化，得到 molecule-conformer embedding `z`；
4. 不改变 tokenizer、T5 或无损 GraphPorts codec。

这一步让 motif 拓扑参与“选择和传播几何信息”，比将 motif 内 E3FP 均值直接加到
一个 T5 carrier 更接近 FACET 提供的机制证据。GraphPorts 仍是无损存储/目标语言，
但不会为了 G3a 把更多连接字段展开成新文本 token。

### 4.5 决策门

G3a 的 dev 必须同时报告：

- learned distance 与 `d_geo` 的 Pearson / Spearman；
- 相对 frozen G1b 简单池化 baseline 的改善；
- 同分子构象排序准确率；
- 刚体复制的 latent distance；
- E3FP donor 错配后的关系损失。

首轮将“dev Pearson 与 Spearman 均至少 0.45，且相对精确同数据 baseline 至少改善
0.15”作为继续 G3b 的工程门。该阈值低于 FACET 的报告值，避免把不同任务的绝对
数字机械搬用；最终论文以完整置信区间和配对结果为准，而不是只报告是否过线。

## 5. G3b：通过后才做 topology-aware T5 接入

若 G3a 通过，冻结其几何关系编码器，并以 adapter/attention bias 接入 T5 motif
carrier。保留标准 T5 CE，再加入明确的 state-match 目标；优先采用同 2D motif
身份、同 atom count、不同 geometry state 的条件对比，而不是 raw E3FP-ID MSE。

GPU 筛选只保留一对：

- G3b-0：相同 topology-aware adapter，但只用 T5 CE；
- G3b-C：T5 CE + conditional state-match CE。

主门仍是完整 dev 的几何内容使用：

- T5 CE 不得明显退化；
- same-size shuffled-minus-aligned NLL 至少 `0.01`；
- zero-geometry bridge 必须造成可观测的 NLL 恶化；
- 条件对比在同身份、同尺寸子域高于随机水平。

若 G3a 不通过，不启动 G3b。此时应裁决 E3FP/G1 是否只适合作为离散辅助任务，
或改用连续坐标/距离/扭转特征，而不是继续通过更强残差强迫 T5 接受无度量的表示。

## 6. motif 划分与创新边界

FACET 支持 fragment 层级、fragment graph 和 atom–fragment incidence 的价值，但不能
证明 CAMT5-derived motif 划分最优。当前不立即替换分区；只有在 G3a 关系门通过后，
才在固定关系模块上做当前分区与 RingPath 的一次小规模配对消融。

创新表述必须收窄为：

> 在生成式分子自然语言模型中，以无损 motif 拓扑语言作为化学身份接口，并在
> motif 粒度聚合和监督构象状态，从而统一结构恢复、文本任务和构象敏感建模。

不能再声称首次将 fragment/motif 与 3D 分子表示结合。
