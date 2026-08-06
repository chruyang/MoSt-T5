# MoSt-T5 科学路线、必要对照、数据准备与执行计划（2026-08-06）

状态：当前科学执行总计划。本文在不改变既有数据 release 事实和三域 codec 合同的前提下，收束文档 35–37 中过宽的候选矩阵；若候选优先级、下游范围或三天调度与旧文档冲突，以本文为准。

2026-08-07 执行修订：核心假设与 A0/A1/M0/M1 四格不变，但数据证据和调度方式作如下收紧。

1. QM9 保留 3D-MolT5 发布的处理后 E3FP/指令 artifact 为主来源，但不沿用其有缺陷的行级 split：固定 HF revision `bfe55090be9ebf1c9cbbe6687a5796711ac0edd8`，只合并 train 347,774 与 validation 1,928，并在构建器中明确禁止读取与 validation 字节相同的 test；删除 42 对连 SELFIES 和完整 `molecule_fp` 都相同的模型可见精确重复，保留 349,660 instruction rows / 128,836 canonical-isomeric-SMILES groups。主协议按 group ID 排序后用 NumPy PCG64 seed 42 重排，采用 QM9 几何模型常见的 `110,000 train / 10,000 validation / remainder test` 规模，即 110,000/10,000/8,836 分子组和 298,529/27,211/23,920 instruction rows；所有 HOMO/LUMO/gap prompts 及同一 strict canonical-isomeric 身份的不同 E3FP 观察必须留在同一 split。该协议只称 `qm9-3dmolt5-idgroup-110k10k-rest-s42-v1`，而 canonical non-isomeric connectivity 只用于更保守的 P1/P2 protected union；不声称复现 DimeNet/SchNet 的 exact membership。
2. FineMolTex 的正式实现使用 prompt IDs `101–106`、`205–206`、`501–504`，并默认读取 MoleculeSTM 发布的 `Editing_data/single_multi_property_SMILES.txt`。因此不再自行随机抽一套“协议兼容 200 分子”：发布的 200 分子必须原样封存为 `compatibility test`，不能用于选择 checkpoint、beam、编辑预算或阈值；开发集从其余 ZINC250K 中另行冻结并保持 connectivity-disjoint。
3. 下游来源遵循“3D-MolT5 发布物优先；缺失时使用官方/领域权威发布物”的顺序。3D-MolT5 论文只报告 MoleculeNet 四项结果与样本数，官方仓库没有相应 runner、数据或 split，因此 BACE、BBBP、ClinTox 的首选备选协议改为 KPGT 发布的 scaffold 8:1:1 预切分。KPGT 作者论文源码列出的 11-task benchmark 不含 HIV；进一步核对 3D-MolT5 官方仓库完整数据表及作者公开 HF dataset inventory 后，也未发现可追溯 HIV members/split。HIV 因此固定使用 SHA-256 绑定的 DeepChem `HIV.csv` 权威成员与本项目透明复刻 DeepChem 2.8.0 语义的 deterministic Murcko 8:1:1 derived split，并单独命名为 `HIV-MoleculeNet/DeepChem-Murcko-8:1:1-derived-v1`。MoleculeSTM 四项清洗成员只作为计数匹配 fallback 证据。3D-MolT5 Table 7 只进入 `published results` 表；严格比较必须让 3D-MolT5 与 MoSt-T5 在每一任务完全相同的冻结成员、split 和微调配置上重新评估。
4. R1 不再串行阻塞生产训练代码：数据协议/保护集合与 production bridge/四格接口并行推进；二者只在 full P1 admission 前汇合。PF-CANARY 可以先于最终 protected union 运行，但不得据此启动 full P1。

对应的机器可执行合同为 [`downstream_scientific_registry_contract_v2.json`](../../most_t5_next/r1/contracts/downstream_scientific_registry_contract_v2.json)，当前任务登记为 [`downstream_scientific_registry_20260807_v2.json`](../../most_t5_next/r1/overlap/configs/downstream_scientific_registry_20260807_v2.json)。v1 继续保留为来源证据清单，v2 作为科研协议来源；v2 通过语法与语义测试不等于 full-P1 admission。

## 0. 总体裁决

整体方向没有发现必须推翻的理论矛盾，但原计划存在四类需要现在修正的遗漏：

1. 缺少同一数据、同一 T5 和同一预算下的“原子/ motif × 无 3D/有 3D”内部对照，因而还不能排除“CAMT5 tokenizer 与 3D-MolT5 E3FP 的简单拼接”这一解释；
2. `molecule-global E3FP broadcast`、interface residual、legacy online MSE 等候选过多，且并非回答核心命题所必需；
3. 数据资产数量充足，但 QM9、Motif Editing 和 MoleculeNet 四任务尚未形成可发表协议，P2 也不能直接投影到尚未冻结的 P1 codec；
4. 仅有性质或文本指标不足以证明模型真正读取了构象信息，也不足以证明 motif/interface 对局部编辑有独有价值。

因此当前路线改为：

> 先用一个最小 2×2 回答“motif 粒度是否使 3D 信息更有用”，再用同一个胜出 P1 检验 P2 的细粒度文本对齐；teacher、partition、anchor/interface 和 retrieval 都按明确条件后置。全量只训练最终方案及最近因果对照，不把所有候选完整预训练。

现有 288/288 topology canary、标准 T5 CE 前后向和 production release 已足以结束通用完整性排查。除非数据源、codec 或 release 发生变化，后续不再扩大文件哈希、映射抽查或泛化工程审计；精力转向真实训练链路和科学问题。

## 1. 论文应回答的三个问题

### H1：motif-local 3D 是否构成有效的中观归纳偏置

给定相同分子、T5 主干和 E3FP 输入，atom-centered state 在化学 motif 内归约后，是否比原子对齐表示更适合结构 denoising、构象敏感性质和跨模态任务？

核心不是单独证明 `M1 > M0`，而是同时观察：

\[
M1>A1,\qquad M1>M0,
\]

并在资源允许时估计交互项：

\[
(M1-M0)-(A1-A0).
\]

### H2：身份—连接因子化是否有利于 motif 定位的文本编辑

将 motif identity 与 attachment/connection 显式分开不会增加新的化学信息；它改变的是信息可访问性和归纳偏置。因此可以主张“factorized interface representation improves localized editing”，不能主张“anchor 本身发现了新信息”。

证据必须同时包含目标编辑成功与非目标区域保持，不能只报 FineMolTex 式 hit ratio。

### H3：P2 是否学到细粒度 motif—text 对齐且不遗忘 P1 的 3D 能力

分阶段训练本身已有 3D-MoLM 等先例，不作为创新点。需要证明的是：冻结同一 codec 后，P2 的跨模态 CE 目标能够改善 captioning/editing/generation，同时 P1 的构象敏感 probe 不明显退化。

### 当前不预注册为主张的内容

- EMA teacher/MSE 必然优于 CE；
- 当前 CAMT5-derived partition 是最优划分；
- E3FP motif mean 可以无损重建坐标或具备 SE(3) 等变性；
- 本模型生成 3D 构象；
- staged training 优于 joint training；
- anchor/interface residual 是必要组件。

E3FP 的准确表述应为 `alignment-invariant, conformer-conditioned 3D fingerprint state`。它保留拓扑、原子类型、立体与空间邻域信息，是有损指纹，不等同于坐标解码器。

## 2. 文献与本项目的真实对应关系

| 工作 | 直接支撑 | 不能替本项目证明的部分 |
|---|---|---|
| [E3FP](https://pmc.ncbi.nlm.nih.gov/articles/PMC6075869/) | ECFP 思路可扩展为 alignment-invariant、conformer-specific 3D fingerprint，并可区分立体/空间邻域 | motif pooling 后的信息保真度、T5 融合和坐标重建 |
| [3D-MolT5, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/3d4dc72d715bd6415d356293079adf3d-Paper-Conference.pdf) / [仓库](https://github.com/QizhiPei/3D-MolT5) | T5 中使用 E3FP、固定 embedding fusion、PCQM4Mv2 结构预训练、PubChem 跨模态训练及多类下游可行 | motif 粒度优于 atom 粒度、identity/interface 因子化、teacher/MSE |
| [CAMT5, Findings of EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.1221/) / [仓库](https://github.com/Songhyeontae/CAMT5) | context-aware motif tokenization 可用于 T5 text-to-molecule，并显著压缩 token 预算 | motif 与 3D 状态的结合、当前修改版 partition 的最优性 |
| [FineMolTex, KDD 2025](https://smufang.github.io/paper/KDD25_FineMolTex.pdf) | motif/word 细粒度对齐与 200 个 ZINC 分子 × 12 prompts 的编辑评测有直接先例 | T5 生成式实现、3D motif、非目标结构保持；正式代码入口是论文给出的 [Zenodo DOI](https://doi.org/10.5281/zenodo.15501037) |
| [MoleculeSTM, Nature Machine Intelligence 2023](https://www.nature.com/articles/s42256-023-00759-6) / [仓库](https://github.com/chao1224/MoleculeSTM) | PubChemSTM、zero-shot retrieval/editing；仓库提供 DrugBank、ZINC250K、Editing 和 MoleculeNet 数据入口 | motif 粒度优势和生成式 T5 目标 |
| [3D-MoLM, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/4af24e6ce753c181e703f3f0be3b5e20-Paper-Conference.pdf) / [仓库](https://github.com/lsh0520/3D-MoLM) | representation learning → generative alignment → instruction tuning 的分阶段路线 | 本项目的 motif codec、E3FP 聚合和 CE/teacher 选择 |
| [MoleculeNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC5868307/) | BACE、BBBP、HIV、ClinTox 的任务与推荐评测语义 | 3D-MolT5 Table 7 未写出的具体 split/seed 不能由表格反推 |
| [KPGT, Nature Communications 2023](https://www.nature.com/articles/s41467-023-43214-1) / [仓库](https://github.com/lihan97/KPGT) | BACE/BBBP/ClinTox 等 11 项任务的 released scaffold 8:1:1 预切分与三次运行先例 | 不包含 HIV，也不能证明 3D-MolT5 Table 7 使用了相同 membership |
| [DimeNet, ICLR 2020](https://openreview.net/forum?id=B1eWbxStPH) | QM9 上 110k train/10k validation/remainder test 与 seed 42 的常用规模先例 | 3D-MolT5 artifact 缺原始 molecule-ID 映射，故本项目只能对齐规模，不能声称 exact DimeNet split |
| [NeurIPS 2022 分子预训练复核](https://proceedings.neurips.cc/paper_files/paper/2022/hash/4ec360efb3f52643ac43fda570ec0118-Abstract-Conference.html) | 小分子预训练收益可能小于 split、特征和超参数影响，需配对与多 seed | 不否定本项目；它要求我们减少无关候选并冻结公平协议 |

这些工作对组件级可行性提供了充分依据，但没有一篇工作已经验证“motif identity/connection codec + motif-local E3FP + T5 + 细粒度文本编辑”的精确组合。这正是可研究空间，而不是可以省略内部对照的理由。

## 3. 架构比较：只保留一个必要 2×2

### 3.1 四个内部匹配条件

| ID | 2D 表示 | 3D 表示 | 回答的问题 |
|---|---|---|---|
| `A0` | atom/SELFIES-aligned identity | 无 | 原子级、无 3D 基线 |
| `A1` | atom/SELFIES-aligned identity | atom-aligned E3FP | 内部匹配的 3D-MolT5-style 对照 |
| `M0` | hybrid motif identity + connection codec | 无 | motif 本身和序列压缩带来的增益 |
| `M1` | 与 M0 相同 | atom → logical motif 的 E3FP invariant mean | 当前核心方案 |

四者共享：

- 同一个 T5-1.1-base 初始化、数据成员和 molecule order；
- 同一个一次性构建并冻结的 union tokenizer snapshot：同时容纳基础文本/SELFIES token、motif macro token、可逆 fallback 与 connection token；A0/A1 和 M0/M1 只使用各自输入子集，但四格保留完全相同的 embedding/LM-head 参数形状，训练中不得再扩词表；
- 相同 corruption-rate 超参数、优化器、更新规则和最大输入/输出边界；同时报告实际 masked atoms/motifs、masked identity tokens 与 supervised target tokens，不把 atom 与 motif 粒度下相同的 `p` 误写为相同 mask realization；
- 相同的有效监督 token 归一化；
- 同时记录 molecule exposures、non-padding tokens、GPU hours、显存和参数量。
- A1/M1 共享唯一几何链 `level-aware E3FP shell embedding → fixed level aggregation → atom state → group scatter-mean → carrier token → embedding addition`；不同 shell radius/level 不共享同一个无序 bit lookup，A1 的 group size 恒为 1，M1 的 group 是 logical motif，A0/M0 只关闭同一侧路输入；
- identity carrier 的隐藏维度、几何投影层和输出头定义固定；同时报告 nominal total parameters 与 gradient-active parameters。共享 union vocabulary 消除不同 embedding 大小这一容量混杂，但 A0/M0 不激活 geometry pathway，因此不能仅凭相同 state schema 宣称完全排除函数容量影响。

tokenization 长度本来就是 motif 方法的一部分，无法同时严格匹配 molecule 数和 token 数。主表按相同 molecule subset/epoch 比较效果，并利用训练中间 checkpoint 画 performance-vs-token/GPU-hour 曲线，避免再开一套昂贵实验。

四格的可识别 estimand 必须限定：`A1-A0` 与 `M1-M0` 分别是 atom-representation package 和 motif-representation package 内部的 E3FP 条件效应；差分中的差分衡量 E3FP 在两种表示/序列化/腐蚀 package 下的相对效应。由于 A/M 同时改变 tokenization、connection 表示和 corruption unit，`M1-A1` 不能被写成“仅由 atom→motif pooling 造成”的纯因果效应。当前不为此增加第五个大架构，而以实际 mask/token 暴露量、同身份构象干预和后续最近机制对照收紧解释。

### 3.2 多保真运行矩阵

| 层级 | 必跑 | seed | 作用 |
|---|---|---:|---|
| `PF-CANARY` | A0/A1/M0/M1 单 batch、32 条 overfit、256 条真实边界样本 | 1 | 只验证链路、显存、收敛方向 |
| `PF-1` | 完整 A0/A1/M0/M1 | 1 个 paired seed | 淘汰明显失败方案，不能作论文结论 |
| `PF-10` | A0/A1/M0/M1 各 2 个 paired seeds | 2 | 用于方向筛选、候选排序和估计交互方向；正好是 8 个独立 job，但不作为顶刊正式交互证据 |
| `PF-FULL` | 最终 winner + 最近机制对照 | 当前至少 1，最终目标 2 | 正式主比较；不把四格全部全量训练，未全量覆盖的交互不作正式强主张 |

最近机制对照由 PF-10 决定：若关键问题是 motif representation package，则选 A1；若 3D 增益不清楚，则选 M0。后续完整消融可以补齐其余 full 条件，但在补齐前，论文只对 winner 与实际 full 对照之间的差异作正式主张。

PF-1 前另加入一个不扩展架构网格的小型同身份几何干预探针：保持分子 2D 身份与拓扑不变，只改变构象/扭转角/已声明的立体状态并重新计算 E3FP，检验 A1/M1 的表示或预测是否随几何发生一致变化。跨分子随机打乱 E3FP 只能作为破坏性诊断，不能单独证明 3D 因果，因为 E3FP 同时携带原子与拓扑信息。

### 3.3 从当前竞争中移出的候选

| 候选 | 裁决 | 理由 |
|---|---|---|
| `C1-G` molecule-global E3FP broadcast | 移出主比较；至多作一个 PF-1 机制诊断 | 它不是强外部基线，且不能替代 A1 |
| `C1-R/C1-Rpseudo` interface residual | 当前不实现 | 先用显式 connection codec 和 editing 证明接口价值；额外残差会把表征与容量问题混在一起 |
| legacy online-detach raw MSE | 仅保留历史结果，不再训练 | target 语义和 loss 量级均不适合作为最终方法 |
| sum/concat/gate/cross-attention 全搜索 | 不做 | 3D-MolT5 已显示 sum 与 concat 接近，而 concat 训练更慢；本项目不以 fusion search 为贡献 |
| 16k/32k vocab 完整预训练对比 | 不做 | 用覆盖率、fallback 率、P95/P99 长度和显存 canary 决策即可 |
| BRICS/RECAP/Murcko/全部 partition 全量比较 | 不做 | 只需一个最接近的文献 partition 对照 |
| staged vs joint full pretraining | 不做 | staged 是工程与课程安排，不作为论文创新主张 |
| 多种 MSE、EMA、loss weighting 网格 | 不做 | teacher 只允许一个预注册实现做条件增量测试 |

### 3.4 外部模型不需要全部重训

- 3D-MolT5：QM9、Caption、MoleculeNet 四项和 3D 输入的主要外部基线；
- CAMT5：ChEBI-20、motif tokenization 与原始 partition 的主要基线；
- FineMolTex/MoleculeSTM：Editing 与 Retrieval 的主要外部基线；
- 3D-MoLM/Uni-Mol：3D property/caption 的补充基线。

优先使用论文结果和官方 checkpoint；只有任务 split/输入不一致且该比较会改变核心结论时，才在我们的冻结协议上复评 checkpoint。论文中的 published-result table 与 same-protocol reproduced table 分开，避免为了“公平”重新训练所有大模型。

## 4. Loss 与两阶段目标的简洁实现

### 4.1 P1 主线：CE-first

P1 的准确定位不是“重新学习通用自然语言语法”，而是学习 motif 序列化的化学语法和构象条件的 motif 表征。

主线只使用：

```text
mask complete motif identity span
  -> keep its valid motif-local E3FP condition visible
  -> T5 span CE reconstructs motif identity
```

因此 M1 的首要问题是“3D condition 是否帮助 motif identity/context reconstruction”，而不是“能否重建 E3FP bit”。3D-MolT5 也明确没有把多分量 E3FP token 作为 denoising target。

### 4.2 3D state mask 与 EMA teacher：条件组件，不是默认第二头

只有 M1 在 PF-10 的 3D-sensitive endpoint 上优于 A1/M0 后，才增加：

```text
mask motif 3D state, keep identity handling fixed
  -> student carrier predicts normalized full-input EMA target
  -> one bounded latent MSE + standard CE
```

只比较 `M1 CE-only` 与 `M1 + one preregistered teacher`；固定一个 EMA schedule、target normalization 和 loss weight，不做网格。若它不改善构象 probe，或损害 CE/captioning，就删除 teacher，并从最终框架图中删除“3D state mask”。这使两个 mask 的逻辑完整，又避免为了保留原设想而强行叠加补丁。

### 4.3 P2：仍然使用同一 T5 CE 头

P2 等 P1 codec/tokenizer 冻结后重新线性化，不扩词表。只比较两种训练目标，不新建一套架构：

| 条件 | 目标 |
|---|---|
| `P2-G` | molecule↔text 双向生成/denoising CE |
| `P2-F` | P2-G + 两个对称 CE corruption：文本保持可见时遮蔽 complete-motif identity 及该 motif 的 3D state；分子保持可见时遮蔽普通 text span |

P2-F 是 FineMolTex 细粒度对齐思想在生成式 T5 中的简化实现：不预先依赖人工 motif-word 配对标签，也不新增 cross-attention/contrastive head；通过成对输入中的另一模态恢复完整 motif identity 或普通 text span，保持一个 decoder 和一种 CE。motif 方向必须同时隐藏原 motif 的 E3FP state，否则 3D state 会成为身份侧信道，也无法训练后续 replacement editing。先在 P2 的 10% 数据和 editing/caption dev 上比较 P2-G/P2-F；胜者再跑完整 P2。

P2 前后必须用同一个冻结 QM9/conformer probe，报告 3D 能力保留或遗忘。

## 5. 当前 motif 划分、anchor 与词表裁决

### 5.1 当前 CAMT5 修改具有化学动机，但尚未被证明更好

本地 `representation_new.py` 相对原逻辑的关键变化是：

- 原逻辑把所有相交的环和非单键传递合并；
- 修改版先保持稠环/螺环系统完整，再只把环外非单键合并成独立 motif；
- 与环相连的 exocyclic 多重键不再吞并整个环，其外侧原子可成为独立单原子 motif/连接端。

该修改能避免芳环与外接羰基等结构被过度合并，但也可能把共轭官能团切得过细。因此现在只能称为 `ring-preserving CAMT5-derived partition`，不能称为最优 motif 定义。

最小验证不是完整训练所有 fragmentation，而是：

1. 原 CAMT5 vs 当前修改版：比较 motif 数、size 分布、ring/exocyclic bond 完整率、序列长度、fallback 率；
2. 在同一个 T5-small 或 PF-1 M1 上做一次 ChEBI/editing dev 短测；
3. 若二者接近，选择更短、更稳定、连接恢复更好的版本；若当前版没有优势，回退原 CAMT5，不继续发明第三种 partition；
4. real motif vs size-matched pseudo partition 仅在论文机制分析阶段补做，不进入架构选择期。

### 5.2 anchor/interface 的优雅实现

当前保留 identity 与 connection 的显式因子化，不增加 interface residual。anchor/slot 的价值由以下证据回答：

- P2-F 中保留 connection span vs 隐去/折叠 connection span；
- motif editing 的 attachment atom、bond type、stereo 和非编辑区保持；
- 按 0/1/多 attachment、motif size 和 edit type 分层。

这比在 P1 中额外加入一个 interface-specific neural block 更直接，也更符合“锚点从词表身份中单独提取”的原始创新定位。

Text-Based Editing 不能只写成一个下游名称，必须在 P2 前冻结可执行算子。当前首选是 **instruction-conditioned motif-span infilling**：对源分子的一个完整 logical-motif identity span 置 sentinel，并同时把该 motif 的原 E3FP state 设为 missing；保留 connection span、其他 motif 与其他 motif 的 3D condition；把编辑指令作为同一 T5 输入上下文，生成与原 slot/连接约束兼容的 replacement identity，再由 codec 重建分子。

编辑预算按每个 `source × prompt` 固定总候选数，而不是每个 span 固定 beam；跨 span 用预注册的长度归一化 log-likelihood 排序，禁止根据失败结果自适应补生成。候选选择只允许模型分数、codec 约束与化学有效性，最终报告的 RDKit property evaluator 不得参与挑答案；可过滤 exact-original candidate，因为它只施加“必须变化”约束。报告 raw validity、pass@1、pass@K、target success 与非目标保持。第一版只允许一次 motif substitution；motif 插入/删除和两步编辑只有在单步覆盖不足的预注册诊断成立后再加入。

该任务准确名称应是 `codec-enabled constrained editing`：connection 保持使一部分局部性由硬约束保证，不能据此声称模型已经学会 attachment。H2 至少需要同一算子的 connection-visible vs target-connection-hidden/collapsed 诊断；若要主张 attachment/bond 语义学习，再补一个带已知 attachment/bond target 的 paired edit 集。这样 P2-F 的跨模态 mask 与下游操作闭环，而不是把 FineMolTex 的 latent optimization 名称直接贴到普通 molecule↔text CE 上。

### 5.3 词表

采用“高频 motif macro token + 可逆结构 fallback + 独立 connection span”，不采用全 motif 闭集词表。K 值只由全量 P1 train census 的覆盖率、fallback 率、P95/P99 序列长度、截断率与 4090 canary 显存共同决定。

当前 32k 可保留为首选候选，但在上述数字和 chemistry-aware fallback 真正实现前不得冻结。下游 valid/test 不用于选择 K；其长尾 motif 只用于检验 fallback 与组合泛化。

## 6. 数据集准备度与取舍

来源选择不是“统一改回官方原始数据”，而是按任务逐级裁决：优先采用 3D-MolT5 实际发布且可追溯的处理后数据；若其文件缺失、切分不可恢复或存在已证实的 split 缺陷，则保留其分子/特征资产并重建透明 split，或退回该 benchmark/任务论文的官方权威发布物。任何 fallback 都记录 DOI/repository revision、文件 hash、成员 manifest、grouping unit 和 split 算法，且双方模型在同一协议上重训/微调。FineMolTex/CAMT5 等 motif-native 任务以各自论文 release 为第一来源，因为它们并非 3D-MolT5 已发布任务。

| 数据/任务 | 当前事实 | 当前裁决 | 下一动作 |
|---|---|---|---|
| P1 PCQM4Mv2 | production admitted 3,365,577；provisional clean-v0 为 3,362,842；288/288 topology 与标准 CE smoke 已通过 | 数据底座基本就绪，训练主线未就绪 | production record → chemistry-aware BoundRecord/collator/model canary；最终下游集合冻结后重派生 membership |
| P2 PubChem | 301,655 条 source 完整；provisional clean-v0 为 289,634；`direct_projection_domain_compatible=false` | 数据够用，禁止直接沿用旧 motif 投影 | 等 P1 codec/tokenizer 冻结后全量重新线性化 |
| QM9 HOMO/LUMO/gap | 3D-MolT5 HF revision 已固定；旧 validation/test 字节相同且 train/eval identity 高度重合；合并 train+validation、禁止 test 并去除 42 个模型可见精确重复后为 349,660 rows / 128,836 groups | 使用 `qm9-3dmolt5-idgroup-110k10k-rest-s42-v1`，不再把旧 split 当正式结果 | 在高 CPU 实例固化 110,000/10,000/8,836 group membership、298,529/27,211/23,920 row membership 与两级 canonical identity manifest；明确 released E3FP 的坐标逐行来源不可独立回溯 |
| PubChem caption | 当前 LMDB 约 12,000/1,000/2,000；3D-MolT5 论文表为 11,955/996/1,988 | 基本可用但来源口径需统一 | 冻结 `reported-protocol` 与 connectivity-clean 两个视图；补 BLEU-2/4、ROUGE、METEOR 与化学事实性 |
| ChEBI-20 | 26,407/3,301/3,300；exact SMILES/CID 初检无 split 交叉 | 最接近正式可用 | 完成 connectivity view、最终 codec adapter 和完整 evaluator |
| Controlled Motif Editing | MoleculeSTM 发布的 200 分子已远端固化为 200/200 unique；FineMolTex 12 个 prompt ID 与默认读取路径已由官方 Zenodo 代码确认 | 200 分子是 sealed compatibility test，不是 dev；实际 motif-span editing runner 尚未实现 | 从其余 ZINC250K 冻结 disjoint dev；冻结 prompt/evaluator 语义和单步 motif substitution 算子；先只把 test identities 纳入 P1 保护集 |
| BACE/BBBP/HIV/ClinTox | 3D-MolT5 官方 runner/split 不存在；KPGT 官方论文/release 覆盖 BACE/BBBP/ClinTox，采用 scaffold 8:1:1，并发布 `scaffold-0/1/2` membership replicas；论文报告三次独立运行，但源码未公开可据以重造相同 membership 的数值 seed；KPGT 不含 HIV；远端已有三项的三组 candidate split。HIV 已固定 DeepChem 2.8.0 `HIV.csv`：2,193,844 bytes、41,127 rows、SHA-256 `9ffa7fe5...1d22` | 三项冻结从官方 Figshare archive 直接绑定并与读取目录逐字节一致的 KPGT membership manifests；HIV 物化 `HIV-MoleculeNet/DeepChem-Murcko-8:1:1-derived-v1`；Table 7 单列 | full P1 前冻结每个 split 的成员与 val/test canonical identity；标签、3D conformer recipe、adapter 与微调 runner 在 P1 期间并行完成 |
| PubChemQC | 数据大、JSON 需 LMDB join，split 口径未闭合 | 首轮移出核心 | QM9 结果稳定后再补，不阻塞 P1/P2 |
| DrugBank retrieval | 当前未注册；无 global contrastive objective | secondary/optional | 只有 P2-F 已证明有效且时间允许时，用 generative likelihood 或后续 alignment head 评测 |

关键判断：当前是“数据资产领先于可发表协议”。不能因为 P1/P2 数量完整就直接开全量，也不需要继续做与科学结论无关的全盘审查。

## 7. 下游任务组合与指标

### 7.1 主结果

| 角色 | 任务 | 最低指标 | 主要回答 |
|---|---|---|---|
| 3D-sensitive | QM9 HOMO/LUMO/gap + same-2D conformer/torsion/stereo probe | MAE、valid numeric rate、direct regression probe、geometry shuffle/perturbation delta | 是否真正使用构象状态 |
| motif-native | FineMolTex-compatible Text-Based Editing | validity、target success、原始 hit ratio、非目标 scaffold/motif preservation、attachment/bond/stereo correctness | motif/interface 是否支持局部编辑 |
| cross-modal | PubChem captioning | BLEU-2/4、ROUGE-1/2/L、METEOR、motif/property factuality | P2 molecule-to-text 能力 |

QM9 的 `110k/10k/rest` 仅借用 SchNet/DimeNet 系列广泛采用的训练/验证规模作为权威先例；由于 3D-MolT5 artifact 不提供原始 QM9 molecule ID/row map，不能把本项目 membership 冒充它们的 exact split。另保留 90/5/5（115,952/6,442/6,442 groups；314,715/17,464/17,481 rows）为低优先级 split-size sensitivity，不进入三天主流程。3D-MolT5 论文将几何称为 DeepChem/QM9 SDF 来源，官方预处理代码直接读取 SDF conformer 坐标生成 E3FP，但公开 artifact 缺 source-SDF revision、坐标和逐行映射；因此论文表述限定为 `paper-claimed DeepChem-SDF-derived 3D-MolT5 released E3FP artifact`。

### 7.2 支撑结果

- ChEBI-20：validity、canonical exact、SMILES BLEU、Levenshtein、MACCS/RDK/Morgan、FCD、Text2Mol；主要证明 codec/fallback 与 text-to-molecule 生成。
- MoleculeNet 四项：BACE、BBBP、HIV、ClinTox，报告 ROC-AUC；KPGT 的 `scaffold-0/1/2` membership-replica variation 与同一 membership 内的 training-seed variation 分开报告，不混写为一个“seed 方差”。这些任务用于与 3D-MolT5 Table 7 同任务对照，但不作为 3D 因果证据。

3D-MolT5 论文只给出四任务表格，没有在正文中充分固定 split/seed。执行顺序是：

1. 优先从 3D-MolT5 仓库/数据 release 恢复 exact protocol；当前尚未恢复成功，因此不把论文分数与我们的数字写成严格同协议优劣；
2. BACE、BBBP、ClinTox 使用经 KPGT repository revision、Figshare 文件 hash 和成员清单核验的发布 split；首表命名为 `KPGT-scaffold 8:1:1 same-protocol reproduced evaluation`，不命名为 `3D-MolT5 Table 7 reproduction`；
3. 当前三项现有文件尚待与 Figshare 发布包逐字节/成员级同源核验；在此之前只称 `KPGT-layout candidate copy`，不把它们升级成正式协议。构建器必须直接读取已 hash 的 ZIP/TAR 归档成员，并证明实际 `dataset_root` 12 个文件与归档逐字节相同后才输出 official manifest。若当前地区持续无法访问 Figshare，则换网络地区远端补齐；
4. KPGT 不含 HIV，3D-MolT5 官方发布清单也无可追溯 HIV 数据/split。HIV 使用 SHA-256 绑定的 DeepChem `HIV.csv`，按 DeepChem 2.8.0 的非手性 Bemis–Murcko group ordering 与 greedy 8:1:1 语义生成并发布 exact project membership manifest，命名为 `HIV-MoleculeNet/DeepChem-Murcko-8:1:1-derived-v1`；不能伪称 KPGT/3D-MolT5/DeepChem released exact split；
5. MoSt-T5 final/nearest control 与 3D-MolT5 PRETRAIN checkpoint 采用相同 split、分类头、训练预算、早停与评价代码重新微调；3D-MolT5 published Table 7 独立列为背景参考；
6. 不为四个小任务单独预训练新架构；三项先用 KPGT split-0、HIV 用 derived split-0 完成方向性结果，资源允许再补 split-1/2 与微调 seeds。

### 7.3 次要结果

Zero-Shot Retrieval 保留，但不进入首轮架构选择，也不为它预先增加 contrastive head。若后续加入，只能作为 P2 对齐能力的补充结果。

## 8. 最小数据政策

只保留下列与论文结论直接相关的处理：

1. 先固定每项任务的来源优先级：`3D-MolT5 released artifact → task/benchmark official authoritative release → declared fallback`；数据源和 split 是两个字段，不能因沿用 3D-MolT5 分子/E3FP 就沿用已证实有缺陷的 row-level split；
   - downstream split grouping 与 pretraining decontamination identity 分开记录：QM9 主 split 用 strict canonical-isomeric molecular identity 近似官方 molecule ID；P1/P2 保护集合仍用更保守的 canonical non-isomeric connectivity，不能把两者混成一个字段；
2. 在 full P1 前冻结确定会进入论文的 validation/test molecule IDs，并由此生成最终 protected union；这项门禁不要求所有下游 evaluator、conformer adapter 或微调 runner 已经完成；
   - 对正式计划使用的 KPGT `scaffold-0/1/2`，保护集合取三份 validation/test membership 的并集，而不是只保护开发阶段先跑的 split-0；
   - HIV 的多次训练 seed 共用同一冻结 split，只计一次 validation/test membership；
3. 从 P1/P2 clean view 中排除这些 exact canonical-connectivity 分子，保留官方 benchmark 原始 split 作为 `reported-protocol` 视图；
4. 同一分子的多性质 prompt/多构象必须按 molecule group 分组，不能按 instruction row 随机切分；
5. downstream train 与预训练重合可以保留并披露；P1 与 P2 的有意 replay 不是泄漏；
6. 不做全局 scaffold 排除，不因预训练内部重复重做全量数据，也不把所有未来可能任务都变成当前 P1 blocker；
7. optional retrieval、PubChemQC 和未来扩展不阻塞 canary；若在 full P1 前确认进入论文，再把其 valid/test 加入最终保护集合。

## 9. 执行顺序

### R1 收口：不需要 GPU

1. 将当前 downstream registry 升级为科研版 v2：核心为 QM9、Caption、ChEBI、Editing；MoleculeNet 四项为 supporting；PubChemQC/ retrieval 后置；
2. 修复/重建 QM9 molecule-level split；
3. 远端获取并冻结 Editing 数据与 12 prompts；
4. 冻结 MoleculeNet 四任务的数据版本与 split 语义；
5. 完成 Caption/ChEBI clean view，重建最终 P1/P2 membership；
6. 冻结 hybrid codec、K 值和 P1 tokenizer；随后重新线性化 P2。

当前 `0.5 vCPU / 2 GiB` 只够来源下载与轻量 manifest 核验。进入 QM9 分组重切、KPGT/HIV canonical membership、protected union、motif census 和重新线性化时，需切回至少 `16 vCPU / 120 GiB`；E3FP 主 release 已完成，不需要再租 96 vCPU，也暂时不需要 GPU。

### P1 实现与选择

1. production BoundRecord、chemistry-aware fallback、collator 和 trainer；
2. 实现 A0/A1/M0/M1，先不实现 C1-G、C1-R、C3；
3. 单张 4090 跑真实 PF-CANARY，并据实测 tokens/s 估算每个预算；
4. 通过后切 8×4090：PF-1 四个独立单卡任务并行；
5. PF-10 四格各两个 paired seeds，共 8 个独立单卡 job；
6. 选定 winner 与 nearest control。

### P2 与下游

1. 从同一个 winner P1 初始化 P2-G/P2-F 10% 对照；
2. 用 Editing dev、Caption dev 与 P2 前后 3D probe 选目标；
3. 完整 P2 只跑胜出目标；
4. 首轮最终结果优先 QM9/conformer probe、Editing、Caption；ChEBI 与 MoleculeNet 四项并行补充；Retrieval 后置。

## 10. 三天资源计划的真实边界

三天目标应定义为：

> 完成架构选择，得到一个使用全量成员/预注册 token budget 的 final P1 checkpoint，尽可能完成 P2 和 1–2 个主下游方向性结果；不是在三天内完成顶刊所需全部 full seeds、消融和统计。

需要区分：

- `dataset-full`：训练访问了全部冻结成员或等价的全量 token budget；
- `convergence-full`：在预注册更新数上达到稳定收敛。

三天内能否达到后者只能由真实 canary 的 tokens/s 决定，不能由“10%/100% records”直接推算。

当前 `most_t5_next/p1` 已有 production motif/atom record bridge、A1/M1 共用 E3FP carrier fusion、A0/A1/M0/M1 统一 T5 wrapper，以及 A1/M1 post-hydrogen-projection atom-domain parity gate；不再只是 synthetic smoke。尚缺的是从冻结 PCQM topology/geometry 到两个 production record family 的批量 producer、最终 union tokenizer snapshot、训练 launcher/checkpoint 日志和 P2 adapter。预计仍需约 6–12 小时高 CPU 数据物化与开发收口；这段时间不应租 GPU。下面的 72 小时是 **production canary 就绪后的计算窗口**，不是从当前代码状态立即起算。

建议调度：

| 时间 | 任务 | 资源 |
|---|---|---|
| GPU 前 6–12h | 物化上述 R1 科研协议、冻结 union tokenizer，并完成 production producer/training launcher | 至少 16 vCPU/120 GiB + 本地；不租卡 |
| 0–6h GPU | 真实 A0/A1/M0/M1 canary、吞吐/显存/断点 | 1×4090 |
| 6–18h | PF-1 四格；同时准备 downstream adapters | 4–8×4090 独立 job |
| 18–42h | PF-10 四格各 2 个 paired seeds | 8×4090 独立 job |
| 42–72h | 8 卡训练 final P1；若提前完成则继续 P2-F/P2-G 胜出方案与主下游 | 8×4090 DDP 或按吞吐最优切分 |

若 canary 预测 final P1 在剩余时段内无法达到预注册预算，则完整保存可恢复 checkpoint，并将“三天结果”明确标记为 scale-confirmed proxy；不能事后把未收敛 checkpoint 称为最终模型。

若“三天”必须包含当前尚未完成的 6–12 小时准备期，则现实目标仍应优先保证 PF-10 架构确定 + final P1 尽量完成；不能同时承诺收敛的 full P1、full P2 和多个下游最终数字。

## 11. 晋级与止损标准

M1 只有同时满足以下条件才作为 final：

1. PF-10 两个 paired seeds 中，M1 相对 A1 在预注册 3D-sensitive endpoint 方向一致；
2. 相对 M0 有明确几何增量，geometry shuffle/perturbation 后该增量显著衰减；
3. motif identity recovery/held-out CE 不劣于预注册容忍区间；
4. 同时报告 nominal/gradient-active 参数、效率曲线与资源记录；同身份几何干预和 geometry shuffle 支持“模型使用了 E3FP 条件”，但在没有容量匹配消融时不宣称完全排除 active-path 容量贡献。

若失败：

- `M1 ≈ A1`：保留 motif codec 作为生成/编辑表示，但删除“motif 3D 优于 atom 3D”的主张；
- `M1 ≈ M0`：说明 E3FP 未被有效使用，停止 teacher 和 full P1，先修正 3D 输入/任务；
- `M1 < A1`：原子级 3D 作为主模型，motif 仅留作 P2/编辑分支；
- P2-F 不优于 P2-G：删除细粒度 mask 目标，保留更简单的生成式 P2；
- teacher 不优于 CE-only：完全删除 MSE，不再调权重补救。

## 12. 当前下一步

立即执行改为两条并行支线，而不是等待所有下游数据完成后才写训练代码：

```text
数据协议线（远端下载/CPU）             训练主干线（本地开发/测试）
registry v2                            production record → CE bridge
  → QM9/Editing/MoleculeNet manifest     → A0/A1/M0/M1 共用接口
  → Caption/ChEBI clean view             → chemistry-aware tokenizer/codec producer
  → final protected union                → CPU contract tests
                    \                    /
                     \→ full-P1 gate ←/
                              ↓
                       1×4090 PF-CANARY
                              ↓
                  PF-1/PF-10 → final P1 → P2
                              ↓
                        headline downstream
```

当前无卡远端实例的真实 cgroup 上限为 `0.5 vCPU / 2 GiB`，只承担官方资料下载、文件固化和轻量元数据检查；本机承担代码实现与单元测试。它不适合 QM9 全量解析、motif census、最终保护集合重建或任何训练。完成下载与 CPU 合同测试后，如果下一步只剩这些重处理，就停止并申请切换到至少 `16 vCPU / 120 GiB`；production 四格 batch 可以端到端前向/反向时，再申请 `1×4090` 做 PF-CANARY。

当前不继续实现 C1-G、C1-R 或 teacher。最先要补的是可发表数据协议、可执行生产 record producer 和四格训练主线，而不是更多审查项。
