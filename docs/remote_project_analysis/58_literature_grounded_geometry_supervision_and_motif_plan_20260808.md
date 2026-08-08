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
