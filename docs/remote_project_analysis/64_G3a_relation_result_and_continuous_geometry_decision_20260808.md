# G3a 构象关系结果与连续几何裁决（2026-08-08）

## 1. 本轮回答的问题

G1 已证明 E3FP/G1b 对构象变化敏感，G2 却没有让 T5 依赖正确配对的构象状态。
受 FACET 启发，G3a 在不含 T5 的条件下先回答：冻结 G1b motif state 加上 motif
拓扑，能否学习跨分子可迁移的连续构象距离？

该实验不是下游性能、T5 效果或完整预训练结果。

## 2. 数据与实现

- 来源：PF-1 train identity 的 C0 冻结候选；
- 构象：每分子最多 4 个 ETKDGv3 未优化构象；
- 正式重放：998 个分子、5,547 个构象对；2 个候选因当前重放不足两个构象拒绝；
- split：798 train / 200 dev，按 molecule identity 完全分离；
- target：固定原子对应下的重原子两两距离矩阵 RMS 差，单位 Å；
- input：frozen G1b motif embeddings、GraphPorts 派生 motif adjacency；
- adapter：两层 message passing、motif mean pooling、64 维 conformer embedding；
- loss：`SmoothL1(||z_i-z_j||₂, d_geo(i,j))`；
- optimization：batch 256、AdamW `1e-3`、500 updates；
- 运行：16 CPU workers 的数据重放约 19 秒；单张 RTX 4090 的训练与全量评价约 10 秒。

正式证据：

- `tmp/g3a_relation_dataset_998_v2_manifest_20260808.json`；
- `tmp/g3a_motif_relation_998_v4_manifest_20260808.json`；
- 训练长度诊断：`tmp/g3a_motif_relation_998_u50_manifest_20260808.json`、
  `tmp/g3a_motif_relation_998_u100_manifest_20260808.json`。

## 3. 正式结果

### 3.1 Frozen baseline 与 learned adapter

| dev 指标 | frozen G1b motif mean | topology adapter, 500 updates |
|---|---:|---:|
| Pearson | 0.06570 | 0.18214 |
| Spearman | 0.09928 | 0.22378 |
| pair ordering accuracy | 53.80% | 51.80% |
| MAE | 1.2019 Å | 0.2267 Å |

adapter 学会了目标的总体尺度，所以 MAE 明显下降；但相关和排序仍很低。它主要把
预测压到目标中位数附近，不能可靠恢复构象对之间的相对关系。

预注册门要求 dev Pearson/Spearman 均至少 0.45，且各自相对精确 baseline 至少
改善 0.15。四项均未通过，G3a 决策为 **FAIL**。

### 3.2 不是简单的“训练不充分”

| updates | train Pearson | dev Pearson | train Spearman | dev Spearman |
|---:|---:|---:|---:|---:|
| 50 | 0.2270 | 0.1470 | 0.2347 | 0.1772 |
| 100 | 0.3878 | 0.1749 | 0.3882 | 0.2093 |
| 500 | **0.9207** | **0.1821** | **0.9133** | **0.2238** |

随着训练进行，train 关系几乎可以被拟合，dev 只小幅改善后停滞。这不是优化器没有
学到信号，而是模型记住了分子特异的离散 state-change 到距离映射，不能迁移到新
molecule identity。继续调 epoch、seed 或 hidden width 不直接解决该问题。

### 3.3 错误构象诊断

dev 中 1,122/1,129 个构象对可选择同一分子的第三个构象作为错误 donor，覆盖
99.38%。500 updates 后：

- aligned subset MAE：0.22675 Å；
- wrong-conformer MAE：0.23874 Å；
- 增量：仅 0.01198 Å。

这与 G2 的弱 shuffle sensitivity 一致：即使关系模块在 train 上拟合得很好，跨分子
仍没有形成稳定的“对应构象状态”度量。

## 4. 对 FACET 启发的正确解释

FACET 的关系监督本身没有被否定。关键差异在输入：FACET 用接收坐标关系的 3D
MPNN/graph transformer，再以 FGW 约束潜距离；G3a 输入是 folded、离散、局部的
E3FP/G1b state。C0 已经显示 raw state-change 与 RMSD 的相关只有 0.135，G3a 又
证明小型拓扑 adapter 不能把这种离散变化转成跨分子的连续尺度。

因此不能把 FACET 简化为“对任何 3D token embedding 加距离回归都有效”。FACET
真正支持的是：连续几何关系应进入 3D encoder，并由 fragment/topology 参与交互。

## 5. 对主架构的裁决（针对数据边界的修订）

PCQM4Mv2/3DMolM 主数据每个分子只有一个提供的 3D 结构，并没有高质量同分子构象
集合。G3a 使用的 ETKDG 重放只能作为机制压力测试；其连续距离泛化失败不能用来
否定 E3FP 作为离散 3D-state descriptor，也不能据此强制主线改成 FACET/SchNet。

现有证据实际上支持保留原核心分工：

- motif/GraphPorts 承担可逆 2D identity 与连接；
- inherited E3FP 承担局部、离散、刚体不变的 3D state；
- G1b 已证明 level 1/2 categorical state 可学、aligned 明显优于 shuffled；
- G2 失败发生在“T5 identity CE + direct residual”没有要求使用 state，而不是 E3FP
  没有 state 信息。

因此主线改为 factorized 3D motif，而不是替换 E3FP：

1. `3D motif = identity carrier + E3FP state sidecar + port/topology sidecar`，不构造
   巨大联合词表；
2. T5 motif carrier 除标准 identity/text CE 外，增加 level 1 主、level 2 辅的
   categorical state reconstruction head；level 3 只保留输入与诊断；
3. identity mask 与 state-shell mask 独立采样，使模型既恢复 motif identity，也必须
   恢复被遮蔽的局部 3D state；
4. 用同一分子内部的 E3FP atom-row/motif-block corruption 建立配对匹配 CE，检验
   state 与正确 motif/atom carrier 的对应，不需要人工生成另一构象；
5. geometry 不再只做一次加法残差，而作为独立 memory 由 motif carrier 做
   cross-attention；motif adjacency/ports 进入 attention bias；
6. 不使用 raw-ID MSE，也不要求 E3FP embedding 欧氏距离等于 RMSD。

连续 SchNet/PaiNN/Uni-Mol 路线降为外部对照：若论文需要“真实构象敏感性”强声明，
应在 GEOM、QMugs 等带半经验/DFT 优化构象的数据上单独验证，而不是把 RDKit 生成
构象当作 PCQM 主训练标签。该对照不阻断当前单构象 categorical-state 主线。

## 6. 文献边界

- FACET（ICLR 2026）：fragment-aware 3D interaction 与 FGW 关系监督的直接相邻证据；
- SchNet（NeurIPS 2017）：连续距离滤波而非离散空间哈希的经典实现依据；
- PaiNN（ICML 2021）：方向信息需要 equivariant message passing 的进一步依据；
- Uni-Mol（ICLR 2023）：大规模 3D 预训练、pair representation 与坐标任务的领域先例。

这些文献支持连续几何编码器和多构象研究方向，但都不能改变当前主数据的单构象
边界，也不能替代本项目 motif–E3FP–T5 接口的独立实验。
