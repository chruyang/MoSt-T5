# MoSt-T5 整体成熟度评估与后续执行路线

日期：2026-08-06  
证据边界：基于当前本地代码、R0/R1 远端执行记录和文档 22–36；本轮未重新连接远端实例，不能把机器当前状态写成实时确认。本文保留为 2026-08-06 的成熟度快照；当前科学比较、数据取舍与执行总计划见[文档 41](41_scientific_design_comparison_dataset_and_execution_plan_20260806.md)。

## 0. 总体结论

项目**科学上可行、工程上可实现，但尚未形成已被实验闭环的顶刊方法**。当前最准确的阶段描述是：

> 数据 release 与可复现性工程已进入后期；科学架构已从宽泛组合收束为一个可证伪的 motif-native 候选；训练数据准入、统一 tokenizer/codec、端到端训练实现和效果证据仍未完成。

因此：

- 现在不能判断“模型有效”，只能判断“研究命题合理且已经具备可检验形式”；
- 当前第一阻塞不是 GPU，也不是 MSE 权重，而是训练 membership、P1/P2 统一 motif 域和 hybrid codec 尚未闭合；
- 不应把全部想法同时塞入主模型。先证明 motif-local E3FP 比无 3D、全局 3D 和原子级 3D 基线更有价值，再决定 interface residual 和 EMA teacher；
- 顶刊路线的关键不是增加任务数量，而是形成一条审稿人可以逐项排除替代解释的因果证据链。

整体状态评为 **黄色（promising but unvalidated）**，不是绿色放行，也不是方向性否定。

## 1. 最终应研究的核心对象

以一个逻辑 motif 为单位：

\[
M_i=(I_i,A_i,S_i)
\]

- `I_i`：motif 内部标准化化学身份及原位置 attachment slots；
- `A_i`：跨 motif 连接端点、边类型和 pairing；
- `S_i`：给定构象上由 atom-centered E3FP 在该 motif 原子集合内归约出的状态。

表面序列可以是一个高频 macro token，也可以是多个 fallback token；二者必须指向同一个 `logical_motif_id`、同一个 atom group、同一个 carrier 和同一个 3D state。由此统一：

```text
identity codec (I): 它是什么
connection codec (A): 它如何与其他 motif 连接
motif state encoder (S): 当前构象下它处于什么 E3FP-derived 状态
```

这才是项目可能成立的核心贡献。T5、E3FP、motif、anchor、MSE 任一单项都已有先例；不能以单项首创定位论文。[3D-MolT5](https://arxiv.org/abs/2406.05797) 已使用 T5、E3FP 3D tokenization 和 1D/3D 联合预训练；[CAMT5](https://aclanthology.org/2025.findings-emnlp.1221/) 已证明 motif/context-aware tokenization 对 text-to-molecule 有价值；[FineMolTex](https://arxiv.org/abs/2409.14106) 已直接研究 motif-level graph-text 对齐和编辑；[E3FP 原论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC6075869/)支持它作为 alignment-invariant、conformer-specific 3D fingerprint，但不支持把它称为可逆几何或等变坐标表示。

推荐主张只有“一主一辅”：

1. **主张**：logical-motif 级身份—连接—构象状态对齐，以及 motif-local E3FP 相对无 3D、全局 3D、原子级 3D 基线的可归因增益；
2. **辅助**：端口位置保持的 hybrid codec 在声明支持域内可验证 round-trip，并改善长尾组合、连接恢复或受控 motif 编辑。

EMA teacher/MSE、anchor auxiliary task、retrieval 和更多下游广度都属于条件扩展，不预先写进标题和核心摘要。

## 2. 当前成熟度矩阵

| 模块 | 当前证据 | 状态 | 裁决 |
|---|---|---|---|
| 数据来源与几何 release | PCQM4Mv2 train-3D 3,378,606 条；production-v2 admitted 3,365,577 条；136 shards、双端审计和语义复算已通过 | 绿 | 可以作为候选数据底座 |
| P1/P2 身份事实 | P1、P2 identity extraction 与候选 overlap 已完成；P1 connectivity unique 3,145,515，P1/P2 overlap 已量化 | 绿 | 事实层完成，不等于策略层完成 |
| 下游去污染 | protected downstream registry/exclusion 尚未全量冻结和证明 | 红 | `all_downstream_exclusion_proven=false`，禁止 P1 |
| P1/P2 关系 | overlap policy 未裁决；P2 与 P1 direct motif projection incompatible | 红 | 必须统一域或明确隔离，不能直接冻结 tokenizer |
| motif identity/connection 设计 | 三域合同、slot-factorized codec、16k/32k+fallback 方案已设计；远端 frequency census 支持 32k 候选 | 黄 | 尚未实现全量 round-trip 与长度门禁 |
| tokenizer | 确定性合同和 builder/gate 已有，但最终 membership/vocab/codec 未冻结 | 红 | `TOKENIZER_FREEZE_PERMITTED=false` |
| admission/record contracts | 当前 v3/v2 仍以 legacy merged mask、必选 geometry-MSE 字段为中心，与 doc35 的 logical-motif、CE-first、条件 C3 不一致 | 红 | 必须发布 vNext contract/validator，不能用旧合同“填满后放行”新方法 |
| 训练数据接口 | geometry sidecar、schema、auditor 较完整；token/logical motif/atom BoundRecord 尚未成为训练实现 | 黄偏红 | 当前 sidecar 明确不可直接训练 |
| 模型架构 | C0/C1-G/C1-L/C1-R/C3 的因果顺序和接口已设计 | 黄 | `most_t5_next` 尚无完整 Dataset/Collator/Model/Trainer 主线 |
| GPU 证据 | 尚无新架构 PF-CANARY、PF-1 或正式训练结果 | 红 | 不能评价效果、稳定性和显存 |
| P2 文本对齐 | 数据规模与旧数据已盘点；统一 motif 投影和新任务接口未实现 | 红 | 必须等待 P1 tokenizer/codec 与 CE winner |
| 下游评测 | 任务角色已基本形成，但 registry、split、保护集和统一 protocol 未全部冻结 | 黄偏红 | 现在准备合同，不启动完整微调 |
| 论文证据 | 文献链和风险边界较完整；无核心实验 | 红 | 仍处研究设计阶段 |

## 3. 已完成工作为什么有价值

当前已完成的大量工作不是“没有训练所以没有进展”。它解决了大模型论文中经常被忽略、但会直接使结论失效的问题：

1. **来源与成员可追溯**：主线 PCQM4Mv2 与 legacy 3D-MoLM/3D-MolT5 口径分开，避免继续把 3,119,717 与 3,378,606 混为同一集合；
2. **几何—分子身份同源**：保留 source atom index，禁止以重建 SMILES 或 compacted offset 猜测 atom mapping；
3. **全量语义可复算**：release 不只通过文件 hash，还通过独立 adapter/semantic recompute；
4. **P1/P2 冲突被提前发现**：direct projection compatibility=false 在大训练前被识别；
5. **tokenizer 风险被显式化**：旧代码 `list(set)`、checkpoint 未绑定 tokenizer 等风险已有替代合同；
6. **原代码被保留**：`most_t5_next/` 作为候选实现边界，legacy 路线仍可作为历史对照。

这些属于可信研究基础和工程贡献，但单独不能证明新算法优于基线。

## 4. 当前五个主要科学风险

### 4.1 E3FP 增益可能主要来自 2D 身份/拓扑

E3FP 从 ECFP 思路扩展而来，atom identity、邻域和拓扑仍在状态中；PCQM4Mv2 又通常只有一个计算构象。仅有预训练 CE、MSE 或 QM9 gap 改善，不能证明模型真正使用构象几何。

裁决：最小必做是 C0/C1-G/C1-L、E3FP shuffle/locality 和同一 2D identity 的多构象/扭转/立体 probe。若论文要写“3D-specific gain”，再条件启用 matched 2D ECFP/coordinate-blind control；若不做，只写 `E3FP-derived conformer-conditioned state`。

### 4.2 当前 CAMT5-derived motif 划分未被证明最适合 3D

C1-L 胜过 C1-G 只证明局部归属可能有用，不证明当前 motif 划分优于原子级、BRICS 或其他 partition。

裁决：C1-L 通过 PF-1 后，在小规模加入一个且仅一个强替代：优先相同预算的 3D-MolT5-style atom-level baseline；若要进一步声称当前划分合理，再做 BRICS 或尺寸匹配 pseudo-partition 短测。替代划分不进入全量消融矩阵。

### 4.3 codec 可能正确但成本过高

高频 macro + 结构 fallback 能解决开放词表问题，但可能造成长序列、attention 成本和显存上升。正确性不自动等于学习价值。

裁决：先 CPU 比较 exact anchored token、top-16k、top-32k+fallback 的 round-trip、P95/P99 长度、截断率、参数量和预计 attention 成本；只有同时满足正确性与成本门禁，才冻结 32k。

### 4.4 interface/anchor 任务可能学习序列化捷径

任意 edge-ID、DFS 顺序、attachment count 和 group size 都可能泄漏答案。

裁决：anchor ID 重编号和 atom 重排不变量必须通过；C1-R 必须优于保持真实 group-size/empty-pattern 的 C1-Rpseudo；编辑任务必须测 attachment atom、slot、bond type 和非编辑区域保持。

### 4.5 CE+teacher 可能把未证明的复杂度绑进主线

EMA teacher 有通用自蒸馏先例，但本项目 target 是 E3FP latent，不是物理坐标距离。它不能因为“常见”而自动合理。

裁决：C0/C1-G/C1-L/C1-R 全部先保持标准 T5 CE。只有 CE-only winner 产生后才实现 C3；C3 不改善 3D-sensitive endpoint，或使 CE/生成超过非劣界，立即删除 teacher/MSE。

## 5. 后续总路线

### 阶段 A：R1 收口与 P1 准入（当前，无 GPU）

目标：把“候选 release”变成唯一、冻结、可训练的 P1 membership。

1. 冻结 Property、Captioning、Text-to-Molecule、Motif Editing 和次级 Retrieval 的数据 registry；
2. 生成 valid/test protected identity 并从 P1/P2 membership、词表发现、频率统计中排除；原始文件不删除；
3. 裁决 P1/P2 overlap：明确 train overlap 是否允许，以及内部 validation 是否必须 group-disjoint；
4. 最短路线冻结 **P1-only vocabulary**，同时在政策中写明 P2 后续必须通过同一冻结 projector 重线性化到 P1 logical-motif 域；P2 重线性化本身不阻塞 P1；
5. 发布 logical-motif/CE-first 的 admission、record、collator contract vNext：`identity_recovery_mask` 为主线字段，`state_prediction_mask/teacher target` 只在 C3 profile 中出现；
6. 冻结 `I/A/S` 字段、membership manifest、reject ledger 和 `N_train_permitted`；
7. 只有 `all_downstream_exclusion_proven=true`、`p1_p2_policy_compliance_proven=true` 后，才进入 codec freeze。

出口：`P1 membership admission = PASS`，但仍不等于可训练。

### 阶段 B：hybrid codec、tokenizer 与三域 binding（无 GPU）

目标：证明同一 logical motif 在 macro/fallback 两种表面表示下身份、连接和 3D mapping 不变。

1. 实现 slot-preserving identity codec、separate connection codec 和 deterministic fallback；
2. 完成 top-16k/top-32k 的一次性长度/覆盖/参数裁决；
3. 支持域内 chemical graph round-trip=100%；stereo/charge/isotope/edge pairing 按声明域验证；
4. macro/fallback 双编码必须得到相同 identity digest、atom group 与 E3FP motif state；
5. 冻结 ordered vocab、tokenizer snapshot 和所有 hash；
6. 构建 128/256 条 tokenizer-bound golden records。

出口：`TOKENIZER_FREEZE_PERMITTED=true` 和 `codec/binding gate=PASS`。

### 阶段 C：候选训练主干实现（CPU + 单 batch）

只在 `most_t5_next/` 新增粗粒度模块，保留原代码：

- `MotifCodec / ConnectionCodec`；
- `BoundRecord`：显式 token、logical motif、atom 三域映射；
- `MotifStateEncoder`：共享 E3FP table、level mean、global/local reduction；
- `MotifDataCollator`：完整 motif span mask、stateless corruption；
- `MoStT5`：一个 T5 backbone、一个 carrier、固定融合；
- trainer/checkpoint：有效 token 归一化、strict resume 和 manifest binding。

先实现 C0、C1-G、C1-L。C1-R、C3 代码都不得成为首版依赖。

出口：unit/contract tests、one-batch forward/backward、strict save/load 全部通过，并形成 tokenizer-bound candidate release。此时只准许 GPU canary，不直接签发正式 P1 admission。

### 阶段 D：PF-CANARY 与 PF-1 科学筛选（RTX 4090）

1. PF-CANARY：32 条重复过拟合 + 256 条边界样本；
2. PF-1：`floor(0.01*N_train_permitted)`，共享 membership/order/masks/token budget；
3. 首轮只比较 C0、C1-G、C1-L，再加一个参数/预算匹配的 atom-level E3FP baseline；
4. 同时运行一个冻结的 downstream dev probe 和同一 2D identity 的构象敏感 probe；
5. C1-L 未稳定优于 C1-G 和 atom-level baseline，则停止 motif-local 3D 主张，不实现 C1-R/C3；
6. C1-L 通过后才短测 partition sensitivity 和 C1-R/C1-Rpseudo。

PF-CANARY PASS 后必须生成独立、可机读的 admission-decision artifact，将 candidate release 明确升级为 `p1_admitted=true`；只有随后才运行 PF-1。出口是一个 CE-only candidate 和一个 nearest causal control。

### 阶段 E：PF-10 尺度确认与条件模块（RTX 4090）

1. PF-10 只运行晋级方案和最近对照，不扩散全部消融；
2. 第一组配对 seed 为正且 CE 非劣才补第二 seed；
3. PF-1/PF-10 排序反转则加入条件 PF-30，不直接外推全量；
4. C1-R 只有优于 C1-Rpseudo 才保留；
5. 选出 CE-only winner 后，才比较 winner vs C3 EMA teacher；
6. 若声称 3D-specific，再加入一个 matched 2D control；否则收缩文字主张。

出口：一个 scale-confirmed winner；明确删除未通过的模块。

### 阶段 F：P1 全量确认

只跑最终方案和最近因果对照，使用同一 clean manifest、token/update、scheduler 和 paired seeds。顶刊目标至少 2 个独立配对 pretraining seeds；若方差或效应接近门槛则补到 3 个。完整报告 tokens、updates、GPU hours、显存和重启。

出口：P1 checkpoint、tokenizer、release manifest 和训练配置严格绑定；若核心差异不稳定，返回收缩论文主张，而不是追加新模块补救。

### 阶段 G：P2 文本/连接对齐

P2 必须使用与 P1 完全相同的 tokenizer、logical-motif schema 和 projector：

- identity/text 双向 masked modeling；
- connection/slot recovery 或 masked edge-pair assignment，尽量仍使用 T5 CE；
- 只有 P1 保留 C3 时 P2 才继续 teacher；P1 删除则 P2 不复活；
- anchor/interface 只有在受控编辑中改善连接正确性和非编辑区保持，才进入论文贡献。

### 阶段 H：下游与论文证据

把任务角色分开，解决此前“Editing 与 Generation 谁是核心”的表述冲突：

| 角色 | 任务 | 证明什么 |
|---|---|---|
| 3D-associated public benchmark | QM9 HOMO/LUMO/gap + conformer/torsion/stereo slice | 性质能力；必须配 no-3D/扰动对照 |
| molecule-to-text public benchmark | PubChem 3D captioning | 跨模态描述；增加事实性/幻觉审计 |
| text-to-molecule public benchmark | ChEBI-20 | decoder、fallback 和组合泛化 |
| motif-native mechanism task | controlled Motif Text-Based Editing | motif/slot/interface 的独有价值 |
| optional diagnostic | molecule-text retrieval | 只有严格冻结才称 zero-shot |

三项公共 benchmark 提供可比性；Motif Editing 是核心机制任务，不再与公共 benchmark 混为同一角色；Retrieval 不占首轮资源。

## 6. 最小且可归因的实验矩阵

实验不是一次性全开，而是按问题晋级：

| 因果问题 | 必要比较 | 运行层级 | 不通过时 |
|---|---|---|---|
| 3D 输入是否有用 | C0 vs C1-G | PF-1 | 删除所有 3D 扩展主张 |
| motif-local 是否有用 | C1-G vs C1-L | PF-1 | 保留全局基线或收缩为 codec 论文 |
| motif 是否优于原子粒度 | C1-L vs atom-level E3FP | PF-1/PF-10 | 不声称 motif 粒度优势 |
| 当前 partition 是否合理 | CAMT5-derived vs 一个强替代 partition | C1-L 通过后的短测 | 只称兼容划分，不称最优 |
| interface role 是否有用 | C1-L vs C1-R vs C1-Rpseudo | 条件 PF-1/PF-10 | 删除 residual/anchor 主张 |
| teacher 是否有增量 | CE winner vs C3 | 条件 PF-10 | 删除 MSE/teacher |
| 是否真正依赖构象 | shuffle + same-2D conformer/torsion/stereo | PF-1 起 | 只写 E3FP-associated，不写几何学习 |
| 是否可扩展到全量 | winner vs nearest control | PF-FULL | 不提交强算法主张 |

这比对所有 loss、mask、vocab、partition 和任务做笛卡尔积更省资源，也更容易形成可解释证据。

## 7. 资源安排与止损

- 当前无卡阶段只做 A–C；不需要复制 10% 数据，只生成 manifest；
- 4090 首次启用只跑 PF-CANARY，先测真实 sequence length、tokens/s、显存和 checkpoint 体积；
- 在 canary 数据出来前不承诺 PF-1/PF-10/全量墙钟，因为 fallback 后有效 token 数比 record 数更决定成本；
- 令 `H1` 为一个 PF-1/C0 在冻结 token 预算下的实际 GPU 小时；若每成员预算近似一致，则单模型 PF-10 约为 `10*H1`，单模型 PF-FULL 约为 `100*H1`，最终两个模型、两个 paired seeds 约为 `400*H1`。这只是 canary 后的容量推算，不代替实测；
- 单张 4090 足以完成 canary、PF-1 和 PF-10 筛选，但 PF-FULL 的最终 pair × paired seeds 很可能成为数周至数月级瓶颈；是否临时升级多卡在 PF-10 后决定；
- 同一层级模型比较固定有效 non-padding token/update、masked target tokens、optimizer updates 和 data order；
- 单 seed 只能淘汰，不能作为论文显著性结论；
- 任一阶段出现 protected overlap、mapping 错误、NaN/Inf、manifest mismatch、CE 越过非劣界或候选排序反转，立即停止晋级；
- 资源不足时优先删除 C3、retrieval、额外 property 数据和全量消融，不删除核心相邻对照。

## 8. 当前下一步

当前只执行阶段 A，并按以下依赖顺序：

```text
下游 registry / protected identities
        + contract vNext（可并行开发）
        ↓
P1/P2 overlap policy + P1-only vocab/P2 later-relinearization decision
        ↓
final permitted membership
        ↓
hybrid codec 16k/32k + fallback round-trip/length gate
        ↓
tokenizer freeze
        ↓
BoundRecord / Collator / C0-G-L implementation
        ↓
PF-CANARY → PF-1 → PF-10 → FULL
        ↓
P2 → downstream/mechanism evaluation
```

在前三个节点完成前，不启动 P1，不生成论文结果，不讨论 MSE 权重调优。
