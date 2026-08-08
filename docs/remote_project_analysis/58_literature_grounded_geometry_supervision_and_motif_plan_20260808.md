# 文献驱动的几何监督与 motif 路线裁决

> 状态：2026-08-08 文献裁决稿。本文取代文档 57 第 4、6 节中“先做
> motif–E3FP InfoNCE、暂不做 same-2D conformer probe”的执行建议；文档 57
> 的既有实验结果仍有效。

## 1. 当前问题不是模型不能接收 3D，而是目标不要求使用 3D

已有 T3MI 和 PF-2C 已经给出一致证据：E3FP 分支能够改变优化轨迹，但最终
预测几乎不依赖 E3FP 的具体内容。T3MI 中 M1-T 的 dev NLL 从 1.795 降到
0.974，但 same-atom-count shuffled E3FP 的 ΔNLL 只有 `1.58e-5`；把 E3FP
置零、随机化或在分子内轮换也几乎不改变结果。PF-2C 冻结 T5 后只训练几何
adapter，同样没有学到配对敏感性。

这组结果排除了“直接 residual + 生成 CE 已经学会 3D”的解释，却没有否定
E3FP 或 motif 粒度 3D 建模本身。更合理的因果解释是：当前输入仍保留大量
motif 拓扑、连接关系和未遮蔽身份，标准生成 CE 可以沿 2D 捷径完成任务，
没有任何一项损失强迫 hidden state 保留 E3FP 状态。

## 2. 文献证据与适用边界

| 工作 | 已验证的做法 | 对本项目的支持 | 不能推出的结论 |
|---|---|---|---|
| 3D-MolT5（ICLR 2025） | 将每个原子的多层 E3FP ID 嵌入后平均，并与 SELFIES carrier 融合；联合做 1D+3D denoising、3D→1D、3D→text 等任务 | E3FP 可作为 T5 的离散 3D 输入；几何到分子序列的生成任务是直接而统一的桥接方式 | 不能证明 motif 内两次原始均值或仅靠 identity CE 会使用构象信息 |
| 3D-MoLM（ICLR 2024）、MolCA（EMNLP 2023） | 先训练结构投影器，再用 caption/generation 把结构表示接入冻结或近冻结语言模型 | 支持“先让几何编码器可学习，再做生成桥接”的分阶段训练 | contrastive alignment 本身不足以保证条件生成或构象语义 |
| GraphMVP（ICLR 2022） | 2D–3D contrastive 与生成式 reconstruction 联合使用 | 支持对齐和重建互补 | 不能用单一 InfoNCE 取代直接 3D 状态目标 |
| Uni-Mol2（NeurIPS 2024） | masked atom 用 CE，坐标和距离用 L1 | 损失类型应由目标语义决定 | 不能据此对离散 hash ID 使用 MSE/L1 |
| FineMolTex（KDD 2025） | molecule-level contrastive + motif/word masked classification | motif 级显式监督和多任务合理 | 它对齐的是图文语义，不是同一 motif 的不同构象状态 |
| CAMT5（Findings EMNLP 2025） | 环和非单键基团作为 motif；每个 motif 一个 token；不使用额外树语法 token | motif 粒度可提高语义密度并减少训练 token | 其表示合同弱于本项目的可逆 GraphPorts，不能直接证明我们的完整序列更短 |
| Deep Sets（NeurIPS 2017） | `rho(sum(phi(x)))` 构造可学习且置换不变的集合函数 | 为 motif 内原子/壳层状态聚合提供最薄理论结构 | 原始 embedding mean 没有元素级 `phi` 和聚合后 `rho`，表达能力不等价 |
| E3FP（J. Med. Chem. 2017） | 对构象生成旋转/平移不变的离散 3D fingerprint | 说明 E3FP 是合法的构象表征入口 | folded ID 是类别标签，不存在“ID 数值距离” |
| GEOM（Scientific Data 2022）、MARCEL（ICLR 2024）、3DCS（ICLR 2026） | 同一分子具有构象集合；评估应覆盖分子内几何变化、手性和能量景观 | same-2D/multi-conformer probe 是 3D claim 的必要证据 | 单一 PCQM 最低能构象上的跨分子指标不能证明构象敏感性 |

主要来源见第 9 节。上述证据共同指向一个原则：**既要有直接几何状态目标，
也要有几何到生成空间的桥接；只做相关性对齐不够。**

## 3. 为什么不把 InfoNCE 作为下一主线

motif 身份和文字语义通常在同一分子的不同构象间保持不变，而 E3FP 应随构象
变化。若把每个 `(motif identity, E3FP state)` 都当独立正样本：

1. 同一身份的不同构象会成为 false negatives；或
2. 为了对齐不变的身份，几何编码器会主动丢掉构象差异；
3. 最终检索成功也可能只表明 E3FP 中包含 2D/元素/连接信息，而非真正使用 3D。

因此 InfoNCE 以后可作为 molecule-level shared-information regularizer，但不能
成为几何有效性的第一证明。MolCA 同样采用 contrast/matching/captioning 的组合，
而不是把 contrastive similarity 当作生成桥接的全部。

## 4. 下一版最小而统一的架构

### 4.1 保留 motif partition 与 GraphPorts v1

当前不重新划分 motif。理由是：CAMT5-derived 的“环和非单键基团合并、其余
原子单例”具有清楚的化学语义；本项目的 graph+ports codec 已在 33,600 条上
完成可逆与立体化学边界验证。当前失败来自几何使用机制，不是 partition 已被
证伪。

GraphPorts v1 继续作为无损存储/输出基线。v2 虽将输入 token 降低 37.49%、
吞吐提高 16.99%，但 dev NLL 恶化 7.28%，说明不能为了长度盲目删减结构表达。

### 4.2 用可学习的置换不变集合编码器代替原始均值

对 motif `m` 中原子 `a`、E3FP shell `l`：

```text
u[a,l] = phi(E3FPEmbedding[id[a,l]], shell_level, atom_role)
g[m]   = rho(mean_or_sum({u[a,l]}))
```

`atom_role` 最少包含 motif core/attachment/port 角色。`phi`、`rho` 均先用小型
MLP；输出仍只注入一个 motif carrier，不增加 T5 序列长度。该形式继承 Deep
Sets 的置换不变性，并允许模型在聚合前学习哪些原子/壳层重要。Set Transformer
或更复杂 attention pooling 仅在该薄基线失败后考虑。

### 4.3 直接 E3FP 状态任务：分类，不是 raw-ID MSE

随机遮蔽 E3FP shell slot，让 motif/atom context 预测被遮蔽的 folded ID：

```text
L_state = CE(predicted_id, target_id), target_id in [0, 4095]
```

至少分别报告 level 1–3 的准确率/NLL，level 0 可作为元素/局部身份较强的简单
对照。这样目标直接要求 hidden 保存 E3FP 内容，又符合 hash ID 的类别语义。
KPGT 对 binary fingerprint 使用分类损失、对连续 descriptor 使用 RMSE；
Uni-Mol2 对类别 atom 使用 CE、对连续坐标/距离使用 L1，均支持这一语义区分。

### 4.4 几何到 motif 的生成桥接

增加 `3D motif state -> lossless motif sequence`：encoder 输入只保留每个 motif
的几何 carrier（以及最少的 segment/boundary），decoder 输出完整 motif identity
和连接序列，仍使用标准 T5 CE。这与 3D-MolT5 的 3D→1D translation 同构，
但把生成单位提升到 motif。

该任务的意义不是要求 E3FP 单独无损反演全部化学身份；而是检验几何表示在
分子上下文中是否能为 motif 身份/连接生成提供可利用信息。应同时报告与
2D-only、shuffled-E3FP 和 unigram/state-prior 的差异。

## 5. 训练顺序：先证明机制，再联合预训练

### C0：CPU 数据与可识别性审计

从 GEOM 或现有分子中选 1,000–4,000 个可旋转、同一 2D identity 有多个构象的
分子，先统计：

- 不同构象间 E3FP slot 改变率，按 shell level 和 motif 分桶；
- target ID 的频率、长尾与 unigram NLL；
- 同一构象的刚体旋转/平移应保持 E3FP 不变；
- 不同构象、对映体和随机 ID 扰动应可区分。

如果 E3FP 在目标构象对上几乎不变，就不应进入训练，而应先调整 E3FP 参数或
构象选择。该阶段只需 CPU，无需租 GPU。

### G1：单卡几何机制门

冻结现有 M0 主干，训练 Deep-Sets 几何编码器和 `L_state`，先做约 500 updates。
只比较三个必要条件：

1. 当前 raw-mean + CE（已有失败结果，可复用）；
2. Deep-Sets + state CE；
3. 2 加上 geometry→motif generation CE。

放行条件在 C0 得到基线分布后冻结，至少要求：state NLL 明显优于 unigram/
no-context；对齐 E3FP 优于 same-size shuffled；同一 2D 的不同构象导致可重复的
表示或预测变化。无需再次重跑已完成的 gate/residual/GraphPorts-v2 实验。

### G2：联合但不补丁化的预训练

G1 通过后再进入正常 motif denoising 与几何任务。优先按固定任务批次交替：

```text
motif denoising CE  <->  E3FP-state CE  <->  geometry-to-motif CE
```

这样比一开始设置任意 `CE + lambda*MSE + beta*InfoNCE` 更易解释。若必须同批
相加，只对一个小的预注册权重做一次敏感性检查，不为每个分支单独调参。

## 6. MSE/teacher 的最终位置

当前继续拒绝 `MSE(raw E3FP ID, hidden)`。ID 100 与 101 在 E3FP 中不比 100
与 300 更接近，数值回归没有化学度量意义。

teacher 方案并未被永久否定，但只能放在后续：teacher 必须产生连续、归一化、
stop-gradient 的几何 latent（例如冻结 Uni-Mol 类 3D encoder，或稳定 EMA
teacher），student 再用 Smooth L1/MSE 预测。data2vec 能支持的是这种连续 latent
回归，不能支持 hash ID 回归。只有当离散 state CE 已证明几何可识别、而生成
桥接仍不足时，才值得增加 teacher。

## 7. motif 序列长度的裁决

全 33,600 条实测：

| 表示 | mean | p95 | max | 长于 AtomSELFIES |
|---|---:|---:|---:|---:|
| AtomSELFIES | 23.321 | 30 | 40 | — |
| GraphPorts v1 motif | 49.804 | 79 | 137 | 96.59% |
| GraphPorts v2 motif | 31.215 | 55 | 103 | 71.26% |
| identity-only、无 graph（诊断下界） | 14.823 | — | — | 18.03% |
| one-token/motif、无 graph（诊断下界） | 9.196 | — | — | 0% |

所以“当前 motif 序列更长”是真的，但原因主要是无损连接语法和 fallback，而不是
motif 数量本身；平均 motif 数仅约 7.20，明显少于平均原子数 14.13。CAMT5 的
短序列来自“一 motif 一 token且不带 grammar token”，与本项目更强的可逆合同
并不等价。

后续优化方向应是**分离存储/输出表示与 encoder 计算表示**：GraphPorts v1
继续提供无损 target；encoder 可用 one-carrier-per-motif，并把 ports/topology
作为 sidecar embedding 或 attention bias，而不是全部展开为文本 token。这一
结构有望接近 9–15 token 的下界，但应在几何机制 G1 通过后单独做配对实验，
不能现在同时改变几何监督和序列语法。

## 8. 当前最终决策

1. 暂停最终全量预训练；已有结果不足以声称使用了 3D。
2. 保留当前 motif partition 与 GraphPorts v1，不再继续 v2/BPE 压缩。
3. 不做 raw-ID MSE；teacher/MSE 后置。
4. 不把 motif–E3FP InfoNCE 作为下一主线；仅保留为后续可选正则。
5. 下一步先做 C0 same-2D multi-conformer E3FP 可识别性审计。
6. C0 通过后，用一张 RTX 4090 运行 Deep-Sets + state CE 的 G1 短门；通过后
   才加入 geometry→motif CE。
7. 只保留三项必要机制比较，复用既有 M0/raw-mean 结果，不重复无关训练。

该路线把论文主张收束为一个可证伪的问题：**motif 粒度的可学习集合编码与直接
几何监督，能否在保持自然语言生成接口的同时，稳定保留同一分子的构象差异？**
它比“加一个 3D residual 后看总体 CE”更接近可发表的机制证据。

## 9. 主要参考资料

- 3D-MolT5, ICLR 2025: https://proceedings.iclr.cc/paper_files/paper/2025/file/3d4dc72d715bd6415d356293079adf3d-Paper-Conference.pdf
- 3D-MoLM, ICLR 2024: https://arxiv.org/pdf/2401.13923
- MolCA, EMNLP 2023: https://aclanthology.org/2023.emnlp-main.966/
- FineMolTex, KDD 2025: https://arxiv.org/abs/2409.14106
- CAMT5, Findings EMNLP 2025: https://aclanthology.org/2025.findings-emnlp.1221/
- GraphMVP, ICLR 2022: https://arxiv.org/abs/2110.07728
- Uni-Mol2, NeurIPS 2024: https://papers.nips.cc/paper_files/paper/2024/file/53923bb44655a7defb31c7744c01b62b-Paper-Conference.pdf
- KPGT, Nature Communications 2023: https://www.nature.com/articles/s41467-023-43214-1
- Deep Sets, NeurIPS 2017: https://papers.nips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html
- E3FP, Journal of Medicinal Chemistry 2017: https://doi.org/10.1021/acs.jmedchem.7b00696
- GEOM, Scientific Data 2022: https://www.nature.com/articles/s41597-022-01288-4
- MARCEL, ICLR 2024: https://openreview.net/forum?id=NSDszJ2uIV
- 3DCS, ICLR 2026: https://iclr.cc/virtual/2026/poster/10010228
- data2vec: https://arxiv.org/abs/2202.03555

## 10. Deep Sets 的现行地位与更适合本项目的变体

### 10.1 Deep Sets 并未过时，但通常是基础组件而非最终复杂架构

MARCEL（ICLR 2024）在 6 个单构象 3D backbone、9 个任务上比较 mean、
DeepSets 和 self-attention 三类 conformer-set encoder。DeepSets 在 54 组
实验中的 42 组取得显著改善；简单 mean 丢失辨别信息，而理论表达力更强的
self-attention 结果反而不稳定。作者将其归因于 DeepSets 的两个非线性变换
能够以较小开销建模集合，而 pairwise attention 更难优化。这是与本项目最接近
的现成证据：**Deep Sets 仍是可信且强的低成本基线，复杂 attention 不保证更好。**

AllSet（ICLR 2022）把 Deep Sets 和 Set Transformer 都作为可替换 multiset
function；Graph Multiset Transformer（ICLR 2021）则把结构依赖加入 attention
pooling。这说明近年的趋势不是抛弃 Deep Sets，而是在需要元素交互或图结构时，
在其置换不变框架上加入 attention/structural bias。

### 10.2 本项目不应把 atom×shell 直接视作一个扁平集合

每个 motif 的输入实际有两层结构：

1. 同一原子的 shell level 0–3 有固定语义和顺序；
2. motif 内原子的排列应当不影响输出。

因此“把所有 `(atom, shell)` embedding 直接 mean”以及“对所有 pair 直接做
Deep Sets”都会丢掉层级。更合适的最薄结构是 **level-aware gated set pooling**：

```text
h_a = phi(concat_l(E[id_a,l] + LevelEmbedding[l]), valid_shell_mask)
score_a = w^T tanh(W h_a)
alpha_a = masked_softmax(score_a over atoms in motif)
g_m = rho(sum_a alpha_a h_a)
```

- shell 先在 atom 内按固定 level 编码，不把 level 当作无序元素；
- atom 再以 attention-weighted set pooling 聚合，保持 atom permutation invariance；
- `h_a` 可直接接 per-shell categorical state head；
- `g_m` 作为唯一 motif geometry carrier 接入 T5，不增加序列长度；
- 参数量只是一组小 MLP 和一个标量 attention scorer。

这一形式可视为 gated attention MIL pooling 与 Deep Sets 的结合。它比纯
Deep Sets 多了可解释的 atom 权重，但不引入 atom–atom 的二次 attention。

### 10.3 候选方法的适配度

| 方法 | 优点 | 对当前任务的风险 | 裁决 |
|---|---|---|---|
| 原始 mean | 无参数、快 | 已被实验证明容易被忽略；无聚合前非线性 | 淘汰为历史对照 |
| 标准 Deep Sets | 置换不变、轻量；MARCEL直接支持 | 若扁平化会丢失atom/shell层级 | 保留为必要基线 |
| level-aware gated set pooling | 保留shell层级；学习atom重要性；开销小 | 不显式建模atom–atom pair | **G1主候选** |
| Set Transformer/PMA | 显式建模集合元素交互 | MARCEL中attention结果混合；参数与优化更复杂 | 主候选失败后再启用 |
| atom–motif cross-attention / hierarchical GNN | HimGNN等证明适合多尺度分子性质学习 | 容易把2D拓扑捷径重新引入几何机制门 | 后续完整模型候选，不用于首个3D因果门 |
| SE(3)/E(3) GNN（PaiNN/GemNet等） | 直接使用坐标，几何归纳偏置最强 | 更换E3FP输入定义、数据流和计算预算 | 若C0证明E3FP不可辨时才转向 |

HimGNN 的 atom-MPNN、motif-MPNN 与 cross-attention，以及 Substructure-Atom
Cross Attention 都证明 atom–motif 交互在性质预测中有价值；但它们的目标是
融合完整化学拓扑。当前 G1 的目的则是隔离“E3FP内容是否被使用”，过早引入
motif graph message passing 会增加 2D 混杂。因此“领域内更复杂”并不等于
“本机制实验更合适”。

### 10.4 更新后的最小比较

G1 只需同一数据、同一 state CE 下比较：

1. standard Deep Sets：atom-level `phi -> mean/sum -> rho`；
2. level-aware gated set pooling：atom-level shell encoder + learned atom weights。

若 gated pooling 未优于 Deep Sets 或仍没有 shuffled/conformer sensitivity，再试
单层 Set Transformer/PMA；不同时展开 GNN、cross-attention 和多层 set attention。
这样既有领域内强基线，也能把改进归因于“层级与自适应聚合”，而不是模型规模。

补充来源：

- MARCEL set encoder comparison: https://sxkdz.github.io/files/publications/ICLR/MARCEL/MARCEL.pdf
- Set Transformer, ICML 2019: https://proceedings.mlr.press/v97/lee19d.html
- Graph Multiset Transformer, ICLR 2021: https://iclr.cc/virtual/2021/poster/3311
- AllSet, ICLR 2022: https://iclr.cc/virtual/2022/poster/6302
- Attention-based Deep MIL: https://arxiv.org/abs/1802.04712
- HimGNN: https://academic.oup.com/bib/article/24/5/bbad305/7245716
- Substructure-Atom Cross Attention: https://arxiv.org/abs/2210.08243
