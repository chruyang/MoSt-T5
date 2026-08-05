# 下游任务组合、锚点因子化与 motif 词表政策（2026-08-05）

状态：主干文献复核与远端频率实验已完成；本文件给出研究设计裁定，不代表 tokenizer 已冻结，也不放行 P1。

## 0. 裁定摘要

1. 原定四项下游任务需要分层，不能继续作为四个等权主任务。**Property Prediction** 与 **Molecule Captioning** 保留；用 **Text-to-Molecule Generation** 取代当前核心中的 Text-Based Editing；将 Zero-Shot Retrieval 降为可选诊断。锚点表示若要成为论文贡献，应配置一个专属机制评测，而不是依赖编辑或检索间接证明。
2. [3D-MolT5](https://arxiv.org/html/2406.05797) 没有提供“预训练前系统排除全部下游 valid/test”的可复核证据。论文只明确提到 ChEBI-20 test 与 PubChem 预训练分子对之间的一次排除操作；删除发生在哪一侧、精确删除清单和预训练清洗代码均未充分公开。因此不能把它写成全局预训练去污染先例。
3. “从完整 motif token 中单独抽出锚点/连接信息”在宽口径上**不是首创**。fragSMILES、SAFE、t-SMILES、HELM，以及 HierVAE、MoLeR、PS-VAE 等都已有片段身份与连接端口/边分开表达或分阶段预测的先例。
4. 当前仍可能成立的贡献边界是：**面向 CAMT5/T5 和 motif-3D 对齐的、端口位置保持的上下文归一化 edge-label 因子化**。必须保留 attachment slot 的原子位置，只把分子内任意的 edge-pair ID 从 motif lexeme 中分离。
5. 不应把 441,769 个 exact motif 全部加入 T5 词表。远端实测支持采用：**高频 deterministic slot-template 原子 token + 显式连接/立体控制 token + 稀有 motif 的确定性结构词法回退**；回退以可逆为设计目标，但必须经化学 round-trip 验证，合法结构不得映射为普通 `<unk>`。
6. 当前预注册上档候选为 **top-32,768 slot-template**，轻量对照为 **top-16,384**。32k 在当前 census 上直接命中 98.9192% 的 motif occurrence，只增加约 25.2M 个共享 embedding 参数；完整 229,600 个 slot-template 会增加约 176.3M 参数。最终选择仍须等 molecule-level P95/P99 长度、截断和 round-trip 门禁后决定。

证据标签：`[文献]` 原论文/官方资料；`[代码]` 官方或本项目源码直接证据；`[远端实测]` 本轮远端产物；`[裁定]` 基于证据的项目决策；`[待验证]` 不能由现有证据推出。

## 1. 原四项下游任务是否修改

### 1.1 推荐的任务分层

| 原任务 | 文献与模型适配 | 对本项目核心主张的证明力 | 裁定 |
|---|---|---|---|
| Property Prediction | 3D-MolT5、3D-MoLM 都有直接任务；QM9 的 HOMO、LUMO、gap 是与平衡构型关联的量子性质基准，但与 2D 拓扑也高度相关 | 中强；单独结果不能识别几何贡献，必须有 matched no-3D、无 geometry-loss，并至少增加一种 geometry corruption/conformer/stereoisomer probe | **核心保留**；先 QM9，PubChemQC 全量后置 |
| Molecule Captioning | 3D-MolT5 与 3D-MoLM 均在 PubChem 上报告 | 中强；验证 molecule-to-text，但 BLEU/ROUGE 本身不足以证明 3D | **核心保留**；增加 3D 事实正确率/幻觉审计和 no-3D 对照 |
| Text-Based Editing | 3D-MolT5、3D-MoLM 都未把它作为下游；MoleculeSTM 的编辑依赖对比空间、额外生成模型及逐样本优化 | 现有协议主要以 2D 结构规则、可计算描述符或代理分类器评分；TOMG MolEdit 具体是官能团 add/delete/substitute，不能自然隔离 3D motif/anchor 贡献 | **移出核心**；以 ChEBI-20 text-to-molecule generation 取代；编辑作为后续应用 |
| Zero-Shot Retrieval | 3D-MolT5 未报告 retrieval；3D-MoLM retrieval 的 Stage-1 checkpoint 又在 PubChem downstream 上微调 10 epochs，并非 zero-shot | 可测跨模态对齐，但当前 T5 CE 主干没有天然的双塔相似度头 | **降为可选诊断**；没有严格冻结协议时改称 molecule-text retrieval |

[3D-MolT5 主文与附录](https://arxiv.org/html/2406.05797)直接覆盖 computed/descriptive property prediction、PubChem 3D captioning 和 ChEBI-20 text-based molecule generation；其消融主要集中在 PubChemQC，而非让全部变体跑遍全部任务。[3D-MoLM](https://arxiv.org/html/2401.13923)覆盖 retrieval、captioning 和 open-text QA，但论文 Appendix C 明确其 retrieval checkpoint 还微调了 10 epochs。[MoleculeSTM](https://www.nature.com/articles/s42256-023-00759-6)确实定义了 zero-shot retrieval 和 editing，但基础是超过 28 万 molecule-text pair 上的对比学习，不是标准 T5 CE 自动获得的能力。

### 1.2 当前推荐的论文证据组合

建议把任务分成三层：

| 层级 | 任务 | 论文中承担的作用 |
|---|---|---|
| 核心效能 | QM9 HOMO/LUMO/gap property prediction | 提供 3D-associated 量子性质结果；只有与 matched no-3D/no-loss 及几何扰动 probe 结合，才可归因 3D |
| 核心跨模态 | PubChem 3D molecule captioning | 检验 3D/molecule-to-text 理解；必须增加事实性与 no-3D 对照 |
| 生成广度 | ChEBI-20 text-to-SELFIES generation | 覆盖 text-to-molecule 方向；只有和同参数预算、同微调协议的无 3D 基线配对时，才可解释为能力保持 |
| 连接机制证据 | anchor-aware motif recovery / masked edge-pair assignment | 检验 slot、edge pairing 和连接表征；本身不证明局部几何学习 |
| 几何机制证据 | geometry-conditioned discrimination 或 3D-state reconstruction | 单独检验 motif-level 3D 状态是否被模型使用 |
| 可选应用 | frozen retrieval 或公开 MolEdit 二选一 | 补充展示，不进入第一轮核心资源预算 |

框架图不应把 anchor recovery 冒充公共下游任务。建议把 `(d) Downstream tasks` 改为三个核心公共基准：Property Prediction、3D Molecule Captioning、Text-to-Molecule Generation；另设独立的 `Mechanism Evaluation / Internal Probes` inset，放置 connection recovery 和 geometry probe。Text-Based Editing 与 Molecule-Text Retrieval 放在 `Optional Applications` 侧栏。若版式必须保留第四个公共下游框，可放 Molecule-Text Retrieval，但应标为 secondary，并在未满足严格协议前去掉 `Zero-Shot`。

### 1.3 为什么不直接保留“Zero-Shot”名称

本文采用如下 **strict zero-shot operational definition**；只有满足这些条件时才使用 zero-shot：

- 下游 train 不参与任何参数、prompt、task-specific calibration 或词表选择；
- 候选池、双向检索协议和评分函数在看 test 前冻结；
- 预训练语料与 test identities 已审计；
- 报告 molecule-to-text 与 text-to-molecule 的 R@1/5/10、median rank 及候选池规模；
- 存在明确的检索打分机制，例如冻结的对比空间。

当前 T5 可选的“生成似然排序”与双塔对比检索不是同一协议，若采用必须单独命名和比较，不能直接引用 3D-MoLM/MoleculeSTM 的结果。

### 1.4 最低评测合同

- QM9：按 molecule identity 分组，HOMO、LUMO、gap 分别报告 MAE（Hartree）、valid numeric rate 与 macro/micro 聚合方式；instruction row 数不能冒充独立 molecule 数，并审计跨 split identity。
- Captioning：名称预测与描述事实性分开；除 BLEU/ROUGE/METEOR 外，预注册可复核的分子事实检查器或人工盲审协议。
- ChEBI-20：至少报告 validity、canonical exact match、Morgan similarity、FCD 与 Text2Mol；它是生成广度证据，不是独立 3D 证据。

## 2. 3D-MolT5 是否提前过滤下游数据

### 2.1 可以确认的事实

[3D-MolT5 论文 Appendix F](https://arxiv.org/html/2406.05797)写道：为避免泄漏，排除 ChEBI-20 test 中同时出现在预训练 PubChem 3D molecule-text pairs 的分子，并删除文本中的分子名称。论文还给出处理后的 ChEBI-20 规模 26,407 / 3,301 / 3,300。

论文和[官方仓库](https://github.com/QizhiPei/3D-MolT5)没有同时公开：

- 完整预训练 corpus construction/cleaning 实现；
- exclusion manifest 或被排除的 molecule IDs；
- PCQM4Mv2 与 PubChemQC/QM9 的全局 identity join；
- PubChem SELFIES/3D pairs 与 caption/ChEBI 的全局 valid/test 去污染；
- connectivity、stereo、scaffold 多层重叠统计。

### 2.2 严谨结论

原句对“从 ChEBI test 删除，还是从 PubChem pretraining pairs 删除”缺乏可复核的操作对象。表中 test 仍为标准的约 3,300 条，未显示测试集缩减，因此与“预训练侧排除”解释相容；但也可能是实际重合为零、表格沿用名义规模或存在未披露处理，**不足以判断删除方向，更不能升级为代码事实**。

因此，对用户问题的直接回答是：

> 3D-MolT5 报告过一次针对 ChEBI-20 test 与 PubChem 预训练 pairs 的重合处理，但没有证据证明它在预训练前系统过滤了所有下游 valid/test；也无法从公开材料完整复核具体删除方向和清单。

这不是否定作者做过未报告处理，而是限定当前可引用证据的强度。

### 2.3 我们应采用更严格、可复核的协议

[3D-MoLM](https://arxiv.org/html/2401.13923)报告了 PubChem 的 301,658 pretrain 与 12k/1k/2k downstream pair/record-level partition，但这不自动证明 canonical identity、stereo 或 scaffold 层面无重合。论文还因 PCDes test 出现在其预训练集而不进行该评测，并因作者判断 ChEBI-20 的 PubChem 来源存在潜在泄漏而避免用其做 captioning。这证明同源泄漏不是理论风险，也说明 split 表仍需 identity-level 复核。

[OGB 官方说明](https://ogb.stanford.edu/docs/lsc/pcqm4mv2/)明确 PCQM4Mv2 源于 PubChemQC；本项目若用 PCQM4Mv2 预训练、再用 PubChemQC 下游，必须先做 identity audit。

[裁定] 本项目采用：

1. 先冻结官方 downstream train/valid/test 和身份 manifest；
2. 保护所有选定 downstream valid+test 的 connectivity identity 全局并集；
3. 从 P1/P2 的训练 membership、tokenizer discovery、motif 频率统计以及所有会更新参数的 generalist/multi-task SFT 中排除命中记录；若任务 A train 命中任务 B valid/test，仍以保护集合为准；
4. 不删除原始文件，只生成不可变 clean view 与 exclusion manifest；
5. 预训练内部重复控制与下游去污染分开报告，并额外审计 caption/ChEBI 文本的 exact/near-duplicate、分子名称和 target leakage；scaffold overlap 作为分层报告，不默认全删；
6. 主结果使用 clean-pretrain 协议，未清洗结果只能标为 `overlap-permitted / non-decontaminated reproduction`，不能默认等同 3D-MolT5 未公开的处理协议，也不能自动称为 transductive。

## 3. 锚点从 motif 词表中独立提取是否可行、是否创新

### 3.1 可行，但不是宽口径首创

应按先例强度区分：SMILES ring closure、SAFE、t-SMILES 与 fragSMILES 是直接字符串表示先例；HELM 是“构件—端口—连接”的结构类比；HierVAE、PS-VAE、MoLeR、MiCaM 是模型决策因子化先例；Lingo3DMol/FSMILES 是最接近 3D 分子语言模型的相邻工作；CAMT5 则是本项目需要直接复现和比较的表示基线。

| 一手工作 | 已有做法 | 对本项目创新边界的影响 |
|---|---|---|
| [fragSMILES](https://www.nature.com/articles/s42004-025-01423-3) | 将 fragment、connector index 和 branch bracket 分开 token 化，并明确以独立报告 fragment 与 breaking bond、避免 fragment redundancy 为动机 | 对“首次拆分锚点”主张威胁最大；必须进入 related work |
| [SAFE](https://doi.org/10.1039/D4DD00019F) | 用成对 ring digits 连接 fragment block，两端共享同一连接标签 | 已有“全局成对连接编号 + 保留端点位置”的直接先例 |
| [t-SMILES](https://www.nature.com/articles/s41467-024-49388-6) | TSID 使用两次出现的 `[n*]`，TSDY 在相同位置保留 generic `*` | 当前候选很容易被理解为 TSID 到 “TSDY-like slot-template + 独立 ID 流” 的因子化 |
| [SMILES ring closure](https://pubs.acs.org/doi/10.1021/ci00057a005) | 同一数字在两个原子位置表示非局部闭合边 | 证明成对 edge label 的基本语法思想早已存在 |
| [HELM](https://pubs.acs.org/doi/10.1021/ci3001925) | 分开表达 monomer、局部 R ports 与显式 connection | 提供构件—端口—连接三层分解的强结构类比 |
| [HierVAE](https://proceedings.mlr.press/v119/jin20a.html) | 分层预测 motif、attachment configuration 和 atom contact point | 支持 motif identity 与 attachment configuration 分开建模，但不是同一字符串表示 |
| [PS-VAE](https://proceedings.neurips.cc/paper_files/paper/2022/file/1160792eab11de2bbaf9e71fce191e8c-Paper-Conference.pdf) | 先生成 subgraph type，再全局预测 inter-subgraph bonds | 支持“生成什么 motif”与“如何连接”因子化 |
| [MoLeR](https://openreview.net/forum?id=ZTsoE8G3GG) | 依次执行 motif/atom 选择、attachment 选择和 bond 选择 | 支持连接变量独立建模，但不是 T5 词表实现 |
| [MiCaM](https://arxiv.org/abs/2302.01129) | connection-aware motif 保留连接位点；其早期变体把 motif 与连接决策分开 | 说明连接信息不能简单删除，也支持因子化对照 |
| [Group SELFIES](https://pubs.rsc.org/en/content/articlehtml/2023/dd/d3dd00012e) | group token 保存多个 attachment point，后续通过 attachment index 连接 | 证明“构件 + 端口”是成熟表示思路 |
| [Lingo3DMol/FSMILES](https://www.nature.com/articles/s42256-023-00775-6) | 在 3D 分子语言模型中以 `*` 保留 fragment connection point，并以遍历规则连接 | 是 3D 模态最接近的相邻先例，但没有当前的外部成对 edge-ID 流 |

结论不是“不能成为创新”。在本次检索覆盖的分子字符串、片段生成及三维分子语言模型工作中，尚未发现同时采用以下全部设计的实现：CAMT5-style motif token、端口原位置保持、分子局部 edge-ID 归一化分离、可逆重建，以及 motif-level 3D state 的一一对齐与联合掩码学习。该结论是有范围的检索结果，不是对全部文献的绝对缺失证明。

### 3.2 精确的信息分解

把第 \(i\) 个 motif 表示为：

\[
(c_i,\;p_i,\;e_i,\;g_i)
\]

- \(c_i\)：motif 的局部化学身份；
- \(p_i\)：attachment slot 数量及其在 motif 原子/字符串中的位置；
- \(e_i\)：每个 slot 对应的分子级 edge-pair label；
- \(g_i\)：与该 motif 对齐的 3D 状态。

当前正确候选是把 \((c_i,p_i)\) 保留为一个 deterministic slot-template token，只分离 \(e_i\) 和 \(g_i\)。这属于 **CAMT5-compatible context-normalized, port-preserving edge-label factorization**，不是把 attachment point 从 motif chemistry 中完全删除。

直接删除 `<n*>` 的旧投影已经在文档 32 中证明非单射：不同 attachment 位置可能得到相同 pure core 和 ID 序列。正确操作是将每个 `<n*>` 在原位置替换为 `<*>`，外部 ID 按 deterministic slot 顺序保存。只有该顺序进一步通过 atom renumbering、motif traversal/root choice 和等价 SMILES 序列化稳定性审计后，才称为 canonical。

### 3.3 可用于论文的有边界表述

中文：

> 提出一种 CAMT5 兼容、上下文归一化且端口保持的 edge-label 因子化：将分子局部、任意编号的跨 motif 边配对标签从 motif 词元中分离，同时在 motif 模板中保留无标签 attachment slot 的位置，使相同 motif-port 构型能够跨连接上下文共享词元，并与 motif 级三维状态稳定对齐。

英文：

> We introduce a CAMT5-compatible, context-normalized, port-preserving edge-label factorization. Molecule-local edge-pair labels are removed from motif lexemes, while deterministic unlabeled attachment slots remain embedded in an atomic motif template, enabling stable alignment with motif-level 3D states.

只有在明确定义的 supported chemistry domain 内，对全语料和扰动样本完成分子图 round-trip，并公开全部拒绝/失败样本后，才能加入 `lossless` 或 `invertible`。比较对象应是标准化且保留立体信息的图同构或 canonical isomeric representation，而不是要求原始 SMILES 字符串逐字相同。

不得写成：首次使用独立 attachment token、首次使用成对连接 ID、首次解耦 motif 与 connectivity，或首次从 motif 中提取 anchor。

### 3.4 成立所需的硬门槛

- 对闭合分子内的每条已切断边，每个 edge-ID 严格出现两次；多组分、盐、开放端口另有显式处理规则，ID 不得跨 disconnected component 配对；
- 每个 motif 的 slot 数等于有序 edge-ID 数；
- slot 顺序对 atom renumbering、motif traversal/root choice 和等价序列化稳定，并覆盖环、同一 motif 对间多重边及同一原子的多个 slot；
- 每条切断边的两端 atom/slot、bond order/type、芳香状态、必要的 bond direction、原子手性、双键 E/Z、formal charge、isotope，以及重连时显式/隐式氢处理均可恢复；若限定跨 motif 边恒为普通单键，必须由全量统计证明；
- edge-ID 是无化学语义哑变量；表示应对 ID 置换等变。实现必须二选一并冻结：规范化编号，或训练期随机重命名并验证任务输出不变性；
- motif/component 顺序、slot 顺序、edge-ID 配对、切断边属性、3D state 与 atom-to-motif 映射共同形成完整逆映射合同，不能以词元无碰撞代替分子级可逆；
- 超过 ID 上限或最大序列长度的比例已全量统计，不能静默截断；
- 报告表示引入的序列长度、吞吐和显存代价。

## 4. motif 词表不能使用全部 unique motif

### 4.1 文献共识不是某个固定大小，而是构建原则

| 工作 | 词表事实 | 对本项目的支持 |
|---|---|---|
| CAMT5 | 远端官方 commit `5875a0a...` 的 `frag_camt5*.txt` 为 11,803 / 18,867 / 22,728，合并 PCDes 后为 24,735；生成与 merge 脚本以 `set` 收集并合并 exact fragments，未见频率阈值 | 证明约 12k–25k 全量词表在其数据规模可运行；不能把这一策略直接外推到 230k–442k |
| [MGSSL](https://proceedings.neurips.cc/paper_files/paper/2021/file/85267d349a5e647ff0a9edcb5ffd1e02-Paper.pdf) | BRICS 产生超过 100k motifs，超过 90% 频次低于 5；作者约 12k 的中等粒度效果最佳 | 直接反驳“词表越全越好”；极低频大 motif 难以学到可迁移语义 |
| [MoLeR](https://openreview.net/forum?id=ZTsoE8G3GG) | 从训练集候选中选择 top-n 高频 motifs，同时始终保留 atom-level generation | 支持固定 top-K 和无结构性 `<unk>` 的基础单元回退 |
| [PS-VAE](https://proceedings.neurips.cc/paper_files/paper/2022/file/1160792eab11de2bbaf9e71fce191e8c-Paper-Conference.pdf) | 从原子开始反复合并最频繁相邻片段，到预定词表规模；100/300/500/700 的任务效果不单调 | 支持用词表—序列—任务 Pareto 选择，而非追求最大词表 |
| [MiCaM](https://arxiv.org/abs/2302.01129) | 频率驱动 merge，并显式保留 connection sites | 支持高频组合与可组合基础单元并存，且不能丢失连接位点 |
| [fragSMILES](https://www.nature.com/articles/s42004-025-01423-3) | ZINC250K 中 exocyclic-single-bond 规则得到 5,869 个 token types，rotatable-bond 规则为 13,035；论文显式优化词表大小与序列长度折中 | 支持把 vocabulary size 与 sequence length 作为同一设计问题 |

### 4.2 当前远端 census 的实测结果

输入：

`/root/autodl-fs/most-t5-r1/runs/cpu-p0-20260805/incoming/pcqm-geometry-production-v2-20260805T0430Z/motif_census.jsonl`

- 输入 SHA-256：`165933bf8b148b4163cf3b2ba5d78a173f4522dd98a5bc4967757d183fe8750b`
- exact unique motif lexeme：441,769
- deletion-core unique：214,554
- position-preserving slot-template unique：229,600
- motif occurrences：24,180,228
- anchor occurrences：41,628,180
- constrained lexical fallback 解析失败：0

因子化将 exact type 数减少约 48%，但 229,600 仍远大于 CAMT5 的 24,735。deletion-core 比 slot-template 少 15,046 种，相对 slot-template 仅减少 6.55%；为这点规模收益删除端口位置信息不合理。

#### 频率阈值

| minimum frequency | slot-template vocab | occurrence coverage | fallback occurrence |
|---:|---:|---:|---:|
| 2 | 78,519 | 99.3752% | 0.6248% |
| 3 | 48,437 | 99.1264% | 0.8736% |
| 5 | 28,658 | 98.8513% | 1.1487% |
| 10 | 15,030 | 98.4892% | 1.5108% |
| 20 | 8,096 | 98.1031% | 1.8969% |

#### top-K + 确定性词法回退（化学可逆性待验证）

| top-K | single-token template direct-hit occurrence coverage | fallback occurrence | motif-segment token multiplier 下界 | 额外 embedding 参数（d=768） |
|---:|---:|---:|---:|---:|
| 8,192 | 98.1106% | 1.8894% | 3.1020× | 6.29M |
| 16,384 | 98.5396% | 1.4604% | 3.0248× | 12.58M |
| 32,768 | 98.9192% | 1.0808% | 2.9515× | 25.17M |
| 65,536 | 99.2678% | 0.7322% | 2.8794× | 50.33M |
| 131,072 | 99.5925% | 0.4075% | 2.8064× | 100.66M |
| 229,600 | 100.0000% | 0 | 2.7216× | 176.33M |

参数估算假设 T5-base `d_model=768` 且输入/输出 embedding 共享。完整 229,600 slot-template 仅共享 embedding 就约 336 MiB BF16；按 12–16 B/可训练参数粗估，仅这些 motif rows 的权重、梯度与优化器状态约为 1.97–2.63 GiB，尚未计基础词表/grammar rows、激活、logits 和 allocator。输出 softmax 每步也要处理更大词表。若直接加入 441,769 exact motif，新增共享 embedding 约 339.3M 参数，已超过 T5-base 主体量级，长尾 token 又只能得到极少更新。

表中的 token multiplier 只相对于“每个 exact motif 一个 token”的 motif segment 基线，不等同于整条 text+molecule 输入长度；脚本还假设每个 anchor 恰为一个 edge-ID token，并未计入拟议 fallback begin/end 或额外 motif delimiter，因此是**下界估算**。若每个 fallback 发出 begin/end 两个边界 token，16k/32k 的相应下界约变为 3.0540×/2.9731×，仍须由最终 tokenizer 重算。它已经揭示：**拆分锚点虽减小词表，却增加序列长度**。最终冻结前必须统计每个分子的 mean/P95/P99、超过 512 的比例、tokens/s 和峰值显存。

这些数值来自当前 production census，尚未应用最终 downstream valid/test protection manifest，所以只能用于确定候选区间，不能直接生成最终 vocab。保护集合冻结后必须在 clean P1 train membership 上重算，若 top-K 成员或覆盖率变化则以重算结果为准。

### 4.3 本轮可复现实验资产

- 本地只保存轻量脚本：[motif_vocab_tradeoff_v1.py](scripts/motif_vocab_tradeoff_v1.py)
- 脚本 SHA-256：`d17a6bcf972d0a45c5c63b466e28db10d50942852391f2741f3ed4a9c1452cb0`
- 远端脚本：`/root/autodl-fs/most-t5-r1/analysis-tools/motif_vocab_tradeoff_v1_20260805.py`
- 远端报告：`/root/autodl-fs/most-t5-r1/reports/cpu-p0-20260805/motif-vocab-tradeoff-v1-20260805-main-review/`
- `report.json` SHA-256：`d575acb07c643ce15a3480002051b4d27c0afd82c5b271e3a4f0c6e1931f580f`
- `summary.md` SHA-256：`8203a7cc250cb70a6d6c9dfe263c57ea85d8d6bfe9c0f4709eb83a4ea0de218f`

实验仅读取远端 census；没有使用 GPU，也没有把大数据下载到本地。

## 5. 推荐的 tokenizer/vocabulary policy v1

### 5.1 三层词表

1. **不可删的基础语法层**
   - 原子、键型、芳香性、形式电荷、同位素、手性、E/Z；
   - motif begin/end、fallback begin/end；
   - generic slot、edge-ID/连接顺序和必要的连接键型 token；
   - 文本与分子模态边界 token。
2. **高频 motif 原子层**
   - 只收录 clean P1 train membership 上的 deterministic slot-template；
   - 以频次降序、lexicographic tie-break 确定 token ID；
   - 当前候选 top-32,768，轻量对照 top-16,384。
3. **确定性结构词法回退层（目标可逆，尚待门禁）**
   - 未进入高频词表的 slot-template 在 motif 边界内展开为基础原子/键/slot token；
   - 外部 edge-ID 和 motif-level 3D state 仍与该 motif 对齐；
   - 合法但稀有结构不映射为普通 `<unk>`；解析失败样本隔离并计数。

回退不能破坏当前“一项 motif 3D state 对应一个逻辑 motif”的合同。dataset/collator 至少为每个逻辑 motif 返回 `logical_motif_id / identity_span / connection_span / carrier_idx / state_idx / atom_indices`；每个 motif 恰有一个 carrier 和一个 geometry target，fallback 展开与 padding 后映射仍保持。可令高频 motif token 自身为 carrier、稀有 motif 的 `<motif_fallback_begin>` 为 carrier，并只向 carrier 注入 3D state，避免按 span 长度重复放大几何信号。

但 carrier 规则本身尚不足以冻结：必须分别定义 2D identity mask 是否遮蔽 edge-ID/connection span、3D-state mask 只遮 geometry 的语义，以及两个 mask 同时命中同一 motif 时是禁止、互斥采样还是定义联合目标。fallback 还会产生更多 CE target；标准 token-mean CE 会使稀有 motif 获得更大权重。需要预注册“保留标准 CE 并分层报告”或“按 logical motif 对分子 span 做 `1/span_len` 归一化”之一，再做梯度与效果短测。

### 5.2 为什么先选 top-32k，而不是直接 min-frequency=5

- min-frequency=5 得到 28,658 types、98.8513% occurrence coverage，与 top-32k 很接近，是很好的 CPU 稳健性参照；
- top-K 对显存、softmax 和 checkpoint 大小提供确定上界，并能在数据增量或重算时保持预算合同；
- 32k 相比 16k 只增加约 12.6M embedding 参数，却把 fallback occurrence 从 1.4604% 降至 1.0808%；
- 64k 再增加约 25.2M 参数，只换来 0.3486 个百分点覆盖提升，边际收益明显下降。

这只是当前 occurrence-level Pareto 的预注册候选，不是最终结论。如果 32k 在 molecule-level P95/P99 或 held-out coverage 上没有实际优势，就选择 16k。

clean-train 重算时还应报告 top-K cutoff frequency 及 cutoff 上同频 template 数。lexicographic tie-break 保证可复现，但化学上是任意的；若边界同频项很多，应同时保留 minfreq policy 的 CPU 对照或调整 K，不能把字典序选择解释成化学优先级。

### 5.3 构建与冻结顺序

1. 冻结下游 valid/test 保护集合；
2. 生成 clean P1 train membership；
3. 以标准化、保留完整立体化学的分子身份作为词表学习文档，避免同一分子的多文本、多任务行和重复构象重复加权；分子内部 motif 重复 occurrence 可保留；connectivity identity 只用于保守去污染，不用于折叠不同手性或 E/Z 异构体；
4. 同时统计 occurrence frequency 与独立 molecule document frequency；
5. 生成 16k、32k、minfreq-5 三个 CPU 候选报告；
6. 通过 round-trip、held-out、长度与截断门禁后，只把 16k/32k 两档带入短程 GPU；
7. 选择后一次性冻结 tokenizer JSON、vocab、special-token map、构建脚本 commit、membership SHA、排序规则和所有依赖版本；
8. P1、P2 和所有下游只能加载同一个冻结 tokenizer，不再训练中途扩词表。

冻结前增加 tokenizer 硬门禁：每个高频 template 必须编码为恰好一个 ID 并 decode 回同一 deterministic template；每个稀有 span 必须无 `<unk>`、无 SentencePiece normalization collision，并通过 `decode → standardized graph → deterministic re-encode`。图比较要求键型、电荷、同位素和 stereo 一致，不要求原始字符串逐字相同。

## 6. 最小实验，不扩大消融网格

### 6.1 CPU 必做

下列 A/B/C 只做 tokenizer、碰撞与 round-trip CPU audit，不训练模型；已知有损的 B 不进入 GPU 训练。

| 组别 | 表示 | 目的 |
|---|---|---|
| A | CAMT5-style exact anchored motif token | 原始高词表/短序列基线 |
| B | deletion core + 独立 ID | 有意的有损负对照，验证端口位置不能删除 |
| C | slot-template + 独立 ID | 当前候选 |

三组报告：type 数、frequency tail、OOV/fallback、mean/P95/P99 长度、>512 比例、token/molecule round-trip、键型/电荷/同位素/stereo 恢复、随机 edge-ID 重命名鲁棒性、atom renumbering 稳定性，以及 motif—3D 映射完整率。round-trip 以 standardized stereochemistry-aware molecular graph equivalence 为准，而非原始 SMILES 字符串相等。

另对 16k/32k/minfreq-5 输出：

- train、P1 held-out、P2 和每个下游集的 occurrence 与 molecule-level coverage；
- 至少触发一次 fallback 的分子比例；
- fallback 后长度、截断率和每个保留 token 的频次分位数；
- exact→template 的频次加权多重度和条件熵；
- template + 外部连接流的 exact/molecule round-trip rate。

### 6.2 GPU 只保留两档短测

- top-16k slot-template + fallback；
- top-32k slot-template + fallback。

固定数据顺序、非 padding token 数、更新步数、mask、loss 和 seed，比较 tokens/s、峰值显存、CE、geometry objective、anchor recovery 以及一项 QM9 几何任务。不再训练 8k/64k/128k/全量词表。

若两档效果接近，优先 16k；只有 32k 在 held-out fallback、截断或下游上显示可重复收益时才保留 32k。

## 7. 尚未解决、不能在论文中提前宣称的事项

1. 当前 census 是 occurrence-level 总表，尚未完成 molecule-level P95/P99、fallback molecule rate 和 >512 截断统计。
2. constrained lexical fallback 的词法覆盖为 100%，不等于化学图、键型和 stereo round-trip 已为 100%。
3. edge-ID 上限、跨 motif 非单键/芳香键/立体键的真实分布尚未形成 release gate。
4. 当前只有 deterministic slot order；是否可称 canonical，必须在完整分子上通过 atom renumbering、traversal/root choice 和等价序列化的 permutation audit。
5. “端口保持因子化提升下游效果”尚无训练证据；当前只有可行性、词表压缩和文献合理性证据。
6. 3D-MolT5 的 ChEBI overlap 删除方向与清单不能由公开仓库完全复核。
7. CAMT5 官方脚本显示 set-union 构词表，但其论文未把 frequency/OOV policy 写成完整数据合同；本项目不能据此照搬全量 unique 策略。

## 8. 对后续流程的直接影响

在进入 P1 前，新的最短关键路径为：

1. 固定三项核心下游与一项机制任务的 registry、official split 和 protection manifest；
2. 完成 P1/P2 对这些 valid/test identities 的全局 overlap proof；
3. 将旧 deletion projection 替换为 port-preserving slot-template，并做分子级 round-trip；
4. 补齐 16k/32k/minfreq-5 的 molecule-level 长度与 held-out coverage 报告；
5. 只在 16k 与 32k 中选择一个冻结 tokenizer；
6. 再执行 Dataset/Collator/CE+geometry 梯度 smoke gate，随后才放行 P1。

因此，本轮结果不是增加训练工作，而是取消两项当前不合适的主下游、避免 230k–442k 词表方案，并把 GPU 决策压缩为两档短测。
