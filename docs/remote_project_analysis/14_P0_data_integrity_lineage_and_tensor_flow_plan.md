# 14 P0 计划：数据完整性、血缘与端到端张量流验收

> 制定日期：2026-07-30  
> 目标：在任何昂贵 Phase 1/2 训练前，证明实际使用的数据完整、可追溯、语义一致，并证明从原始记录到模型 loss 的每一段输入输出符合研究设计。  
> 操作边界：P0 的扫描、trace 和 smoke test 必须默认只读原始 LMDB/检查点；修复或重建只能产出新版本，不覆盖既有数据。

## 1. P0 的重新定义

P0 不只是修复 tokenizer 或 NaN。它是一次“训练发布前验收”，同时回答四个问题：

1. **数据完整吗？** 每条进入训练的数据是否有完整字段、合法类型、可解析化学结构和明确拒绝原因。
2. **数据正确吗？** E3FP、原子索引、motif 映射、文本、构象来源是否在同一原子编号体系下对应同一分子。
3. **数据怎样流动？** 原始记录经过 Dataset、Collator、Model、Trainer 后，张量形状和语义是否仍是预期的。
4. **运行的到底是哪一版？** train 文件、词表、tokenizer、构象/E3FP 参数、代码版本、启动参数能否被完整复现。

P0 通过后，才能把后续 CE-only、current MSE 和 EMA target 的差异解释为“方法差异”，而不是数据版本、映射或路由错误。

## 2. 本轮代码审计已经发现的 P0 风险

下列项是代码事实，不代表远端实际训练数据一定已经出错；P0 必须以实际训练文件和启动命令核验它们。

| 编号 | 观察到的路径/行为 | 风险 | P0 处置 |
|---|---|---|---|
| P0-R1 | process/process_qc_step1_e3fp.py 使用 source coordinates、重原子与一套 E3FP 手工写入逻辑；process/build_phase2_ready_lmdb.py 使用 RDKit 构象、AddHs 和 E3FPTokenizer；process/3Dtoken2db.py 又使用另一套坐标/加氢路径。 | 同名 e3fp 字段可能来自不同原子集合、构象来源、立体化学参数或 shell 编码。 | 对每个实际 LMDB 建立 provenance manifest；禁止未标记混用。 |
| P0-R2 | Dataset 在 e3fp 缺失时会从 SMILES 即时重算，维度不符时会静默 pad/截断。 | 同一个训练集可因缺字段而在运行时变成另一种构象/E3FP 版本。 | 发布训练中 fallback 和自动修复计数必须为零；否则拒绝该数据版本。 |
| P0-R3 | MotifTokenizer 以 set → list 注册 motif，token ID 可能随进程变化。 | 相同 token ID 未必代表相同 motif，checkpoint 与数据语义可错位。 | 词表排序、哈希、保存/重载一致性为硬阻断项。 |
| P0-R4 | process 脚本和 LMDB 记录存在多种 schema：smi/coordinates_list、smiles/coordinates、smiles_kekule、motif_seq、description/text 等。 | Dataset 的字段 fallback 会掩盖缺字段或错误数据来源。 | 制定 canonical schema；实际数据必须映射到它，不能只依赖 fallback。 |
| P0-R5 | Phase 2 的 GSMATDataset 内部随机选择四个任务，未使用传入的 task_probs；run_train2.sh 又引用本地未见的 vocab_phase2_25k.txt。 | 配置文件声明的任务比例或词表不一定是运行时事实。 | 做启动参数解析、路径存在性、resolved config 和运行前 hash 验收。 |
| P0-R6 | Dataset 在异常时改读下一个索引，model 对非有限值调用 nan_to_num。 | 数据丢失和数值异常可能被吞掉，训练样本分布与报告不一致。 | 将跳样、fallback、NaN/Inf 变为带 key 的计数日志和阻断阈值。 |
| P0-R7 | process/build_phase2_ready_lmdb.py 将 linearize(sub_smi) 的第三返回值写成 atom_mapping；当前 representation.py 的第二返回值才是 atom_mapping，第三返回值固定为空列表。 | 若实际 Phase 2 LMDB 由此版本构建，atom_to_motif_map 可能全为 -1，局部 3D 融合与 3D mask 会退化。 | 冻结远端构建脚本/representation.py 哈希；全量比较 stored atom_mapping 与 linearize(smiles)[1]，再决定是否重建新 release。 |
| P0-R8 | Phase 1 launcher 传入 --dataloader_num_workers 16，但 train1.py 会以 DataArguments.num_workers 覆盖该值，默认实际为 4。 | shell 中写的资源配置不一定等于实际生效配置，也会影响吞吐测量和租用规格。 | P0.6 保存 HfArgumentParser 解析后的 resolved config；容量 profile 只能以该配置为准。 |

## 3. P0 必须冻结的“真相来源”

先建立一份 release manifest；没有 manifest 的数据不进入训练。当前启动脚本声明的实际对象如下，均须在远端以只读方式确认其存在、大小、LMDB 统计和哈希。

| 训练阶段 | 启动脚本声明的主数据 | 辅助数据 | 词表 | P0 要确认的事实 |
|---|---|---|---|---|
| Phase 1 | /root/autodl-tmp/3D-MoIT/3d-mol-dataset/pubchemqc/pubchemqc_final.lmdb | 无 | asset/mol_vocabs/vocab_20k.txt | LMDB 真实 schema、E3FP 是否来自 PubChemQC 坐标、tokenizer hash、样本总数。 |
| Phase 2 | /root/autodl-tmp/3D-MoIT/3d-mol-dataset/pubchem/pretrain/phase2_pubchem_final.lmdb | c4_pretrain.lmdb、phase2_text_weights.json | asset/mol_vocabs/vocab_phase2_25k.txt | 词表是否存在且与 Phase 1 checkpoint 对齐；四任务数据来源、每类可用样本数、C4 key/len 约定。 |

每个 manifest 至少包含：

- run_id、生成时间、代码 commit/文件哈希；
- LMDB 路径、文件大小、subdir 模式、entry 数、实际 key 数、保留 meta key；
- 数据 source、split、上游文件哈希和构象/E3FP 生成器版本；
- RDKit、E3FP、PyTorch、Transformers、Python 版本；
- tokenizer JSON/词表文件哈希、special token 到 ID 的完整映射；
- 训练启动参数的解析结果，而不是仅保存 shell 脚本；
- schema version 与所有允许字段的类型定义。

## 4. 目标数据流：必须被 trace 证明

原始分子/文本/坐标记录 → 标准化 SMILES 与原子编号 → 构象或 source coordinates → E3FP（atom × shell）  
同一标准化 SMILES → CAMT5 linearize（motif + atom_mapping）  
两条支路 → canonical LMDB + manifest → GSMATDataset task item → Phase 1/2 Collator → batch tensors + semantic masks → MoSt-T5 encoder fusion → decoder CE + geometry branch → Trainer logs + checkpoint + tokenizer + manifest

任何箭头都需要同时通过“值是否正确”和“输入输出是否满足契约”两类检查。P0 不接受只打印少量样本的人工目测。

## 5. Canonical record contract

### 5.1 发布版本中必须显式保存或以 manifest 可逆恢复的字段

| 字段组 | 必需内容 | 硬性不变量 |
|---|---|---|
| 身份与血缘 | record_id、source_dataset、source_split、source_key、schema_version | record_id 唯一；source 可追溯；没有无来源样本。 |
| 化学结构 | raw_smiles、canonical_isomeric_smiles、smiles_used_for_mapping、atom_universe | 明确是否去立体化学、是否 kekulize、是否显式 H；不能在不同步骤隐式改变。 |
| 原子编号 | n_atoms、atom_index_convention、可选 original_to_canonical_atom_map | E3FP 行、坐标行和 atom_mapping 都指向同一编号体系。 |
| 3D 来源 | conformer_source、source_coordinate_id 或 RDKit seed/参数、MMFF 状态、failure flag | source coordinates 与 RDKit 生成构象不得无标记混合。 |
| E3FP | 数组、shape、dtype、padding policy、bits、levels、stereo、invariants、E3FP version | 每个 ID 为 -1 或 [0, bits-1]；shape 和 atom universe 一致。 |
| Motif | linearize code hash、motif sequence、atom_mapping、anchor convention | 每个真实原子恰好归属一个化学 motif；anchor/文本/pad 不绑定原子。 |
| 文本 | description/enriched_description/text、语言/来源/清洗标记 | Phase 2 所需字段缺失时必须明确拒绝或降级，不得静默替空。 |
| 质量状态 | accepted/rejected、reject_reason、warning flags | 所有未通过样本有可统计原因；训练只读取 accepted 版本。 |

### 5.2 推荐的原子宇宙决策

本项目当前最容易产生语义错位的点是“E3FP 行代表什么原子”。P0 必须在训练前二选一：

1. **推荐：canonical LMDB 只保存重原子 E3FP。** E3FP 第 i 行与 RDKit 重原子 i、atom_mapping 的 i 严格一致；Collator 负责 batch padding。
2. **若保存显式 H：** 必须额外保存 heavy_to_e3fp_row 映射，并定义 H 行是否进入 attention、mask 和 target。只靠未映射行填 -1 不足以说明语义。

不能让 PubChemQC 的重原子 E3FP、RDKit AddHs E3FP 和 padded 256 行数组在同一训练 release 中混合而没有 manifest。

## 6. 分段输入输出契约与验收项

### 6.1 原始记录 → canonical LMDB

| 输入 | 输出 | 必须自动检查 | 失败处理 |
|---|---|---|---|
| SMILES、可选文本、可选坐标 | canonical SMILES、原子编号、文本、provenance | RDKit 可解析；原子数/电荷/键类型记录；跨 split 的 canonical SMILES 重复；文本为空率 | 记录拒绝原因，不补空字段后继续训练。 |
| source coordinates 或 RDKit 参数 | 构象元数据 | 坐标行数等于声明 atom universe；无 NaN/Inf；构象来源唯一可复现 | 坐标不匹配直接拒绝；不在 Dataset 临时替换。 |
| E3FP 生成器 | e3fp[N,4] | bits=4096、level=0..3、stereo/invariants 参数、dtype、范围、padding | 全量检查；抽样重新计算并逐元素比较。 |
| linearize | motif_seq 与 atom_mapping | 映射原子集合等于 0..N-1 且各出现一次；所有 map index 有效；round-trip 单列报告 | 结构/映射失败变成 rejected，而非 Dataset 重试下一个样本。 |

### 6.2 Canonical LMDB → GSMATDataset item

Dataset 当前读取 smiles_kekule 或 smiles、e3fp、atom_mapping 和多个文本 fallback。P0 需验证每个 accepted record 生成的 item 具备如下语义：

| item 字段 | 预期 | 必须验证 |
|---|---|---|
| task | Phase 1 仅 mmm；Phase 2 为 mmm/caption/text2mol/denoise 的显式路由 | 实际计数与 resolved config 一致，不能被内部随机或缺 C4 改写。 |
| text_input_ids | 任务前缀保持为一个单独 token，截断后仍语义完整 | 特殊 token ID、首 token、长度、EOS/pad 约定。 |
| motif_input_ids | <bom>、motif/anchor、<eom> 的稳定 ID 序列 | 两独立进程编码完全一致；UNK 率、截断率可报告。 |
| e3fp_input_ids | 与 motif/atom map 同一 atom universe 的 [N,4] | 训练 release 中不得触发即时 from_smiles fallback 或维度自动修复。 |
| atom_to_motif_map | 对每个真实 E3FP atom 给出合法 motif token 位置，其他行是 -1 | map 值范围、覆盖率、文本/anchor/pad 不绑定原子、截断后不会指向已删 token。 |

### 6.3 Dataset item → Collator batch

P0 必须对 Phase 1 与 Phase 2 分开记录 batch contract。

| 情形 | 预期输入 | 预期输出与语义 |
|---|---|---|
| Phase 1 mmm | motif token、atom E3FP、atom map | input_ids 只含 motif；e3fp_ids 为 atom×4；mask_positions 与 motif 长度一致；joint mask 同时改 motif/E3FP，geo-only 只改 E3FP。 |
| Phase 2 mmm | 文本前缀+正文和 motif | input_ids 为 text+motif；E3FP 先放 text dummy 行再放 atom 行；map 加 text_len 后仅指向 motif 段；mask 只能落在 motif 段。 |
| Phase 2 caption | caption prompt+motif | labels 是原描述 token；3D 可作为编码输入，但 geometry mask 必须为全 false。 |
| Phase 2 text2mol | text prompt | labels 是完整 motif 序列；无 E3FP 时模型必须安全处理空 3D view，且不计算 geometry loss。 |
| Phase 2 denoise | C4 text | labels 是 T5 sentinel target；无 E3FP/motif map；C4 的 __len__ 和 key 规则必须正确。 |

目前 Collator 只返回合并后的 mask_positions。P0 的 trace 版应额外输出 joint_mask_positions、geo_only_mask_positions 和每个样本的有效 3D atom 数，用于证明“2D+3D 联合遮蔽”与“仅 3D 遮蔽”真的执行了预期操作。

### 6.4 Collator batch → Model → Trainer

| 模块 | 输入 | 输出 | P0 断言 |
|---|---|---|---|
| GSMATEmbeddings | input_ids、e3fp_ids | motif embedding、atom embedding | token ID 与 shell ID 都在范围内；padding embedding 恒为零。 |
| GeoSemanticFusion | motif/atom embedding、atom map、atom mask | 与 input_ids 等长的 fused embedding | 仅映射 atom 可进入对应 motif；文本/anchor 空行 gate 为零；无 3D view 不产生 NaN。 |
| T5 encoder/decoder | fused input、labels | logits、CE loss | logits 的 vocab 与 tokenizer manifest 一致；labels 均合法或 -100。 |
| Geometry branch | encoder state、full/masked E3FP、mask | raw geometry loss、有效位置数 | 仅 mmm 的有效 geometry mask 贡献 loss；target/prediction 均 finite；不允许 nan_to_num 掩盖异常。 |
| Trainer/checkpoint | model output、resolved config | 日志、checkpoint | 保存 tokenizer、vocab、manifest、构象/E3FP spec、数据哈希和 loss 子项。 |

## 7. P0 核验方法：全量扫描 + 固定 golden traces + smoke test

### 7.1 全量扫描：判断“完整无误”

全量扫描不是随机抽样。对每个训练/验证/C4 LMDB 逐 key 读取并产生以下统计：

- schema 字段缺失/类型不符/反序列化失败数；
- actual key 数、meta key 数、manifest 声明数、重复 key；
- accepted/rejected 数和每种 reject_reason；
- SMILES 解析失败、canonical 重复、跨 split 泄漏；
- E3FP shape、dtype、range、padding、shell 缺失、bits/level 不符；
- atom_mapping 覆盖、重复、越界、空 motif、截断后 dangling map；
- motif UNK、序列截断、anchor 异常、round-trip 差异；
- 文本为空、过长、异常字符、任务可用性；
- Dataset fallback、维度修复、重试跳样的逐 key 计数。

**硬判定：** 对一个标记为 accepted 的训练 release，必需字段缺失、E3FP/atom map 越界、原子覆盖错误、token ID 不稳定、未记录 fallback 必须为零。允许排除的边缘样本必须在 rejected 清单中，不得被静默略过。

### 7.2 Golden traces：判断“流动是否符合想法”

从每个实际数据 source 固定选择并保存不可变样本：

1. 简单链状分子；
2. 芳香环；
3. 稠合环；
4. 带电荷；
5. 含立体化学；
6. 多组分；
7. 长序列/会截断的分子；
8. description 缺失或极长文本；
9. C4 样本；
10. E3FP/映射边缘样本。

每个 golden trace 保存原始 key、record hash、canonical SMILES、motif tokens/IDs、atom mapping、E3FP 前若干行及 hash、mask 前后张量、decoder labels、model 输出 shape、loss 有效位置数。相同环境中重复两次、两个独立进程各一次，结果应字节级相同；mask 的随机性通过固定 seed 固定。

### 7.3 端到端 smoke test：判断“模型真的消费了正确数据”

至少构造以下五个固定 batch：

1. Phase 1 mmm；
2. Phase 2 mmm；
3. Phase 2 caption；
4. Phase 2 text2mol；
5. Phase 2 denoise。

对每个 batch 执行一次无梯度 forward 和一次可反向的最小 backward，断言：

- 所有输入/输出 shape 与 contract 一致；
- logits、CE、几何 loss、梯度均有限；
- CE label 方向与任务定义一致；
- text2mol/denoise 不产生伪 3D supervision；
- caption 的 E3FP 是输入而非被错误遮蔽的 target；
- mmm 的 mask 与 atom map 同步；
- 修改一个被遮蔽 motif 对应的 E3FP 后，只有预期 mask/target 行变化；
- 同一 batch 使用空 3D view 时不崩溃，但 gate/geometry loss 符合“无 3D”的定义。

## 8. P0 执行工作包

| 工作包 | 产物 | 通过条件 |
|---|---|---|
| P0.0 Freeze | release manifest、实际启动命令、路径存在性报告 | 训练文件、C4、词表、checkpoint、权重文件全部可定位并哈希。 |
| P0.0a Baseline harness | p0_validation 目录、主干哈希清单、baseline parity 报告 | 通过验证层调用与直接调用当前主干得到相同 item、batch、forward 结果；不复制或改写训练主干。 |
| P0.1 Schema | schema inventory、字段覆盖报告、canonical schema 定稿 | 每个 source 都能映射到 canonical contract；无未说明字段 fallback。 |
| P0.2 Chemical integrity | SMILES/atom/coordinate/E3FP 全量报告 | accepted 记录的原子宇宙、坐标与 E3FP 行严格一致。 |
| P0.3 Motif integrity | mapping/round-trip/UNK/截断报告 | 每个原子恰好映射一次；token ID 在独立进程稳定。 |
| P0.4 Dataflow trace | Phase 1/2 golden traces、tensor contract 表 | 所有路径输入输出符合第 6 节语义。 |
| P0.5 Model smoke | 五类 batch 的 forward/backward、finite/gradient 报告 | 无 silent NaN、无空 3D 崩溃、任务/loss 路由正确。 |
| P0.6 Launch integrity | shell 语法、HfArgumentParser dry-run、resolved config | 声明的 task_probs/词表/数据路径确实被运行时消费。 |
| P0.6a Capacity profile | p50/p95/p99/max shape、显存峰值、tokens/s、OOM 边界 | 以真实 Phase 1/2 batch 决定 GPU 数、显存和全局 batch，不从 launcher 中的卡数反推资源。 |
| P0.7 Sign-off | P0 summary、例外清单、release tag | 所有硬项通过；例外只存在于明确 rejected 数据，不进入训练。 |

## 9. 必须先修复或显式决策的阻断项

1. motif vocabulary 的确定顺序和 checkpoint 同步保存。
2. Phase 1、Phase 2 实际 LMDB 的 E3FP provenance；尤其要裁定重原子/显式 H 与 source-coordinate/RDKit-conformer 的统一规范。
3. Dataset 的即时 E3FP fallback、自动维度修复与“换下一个索引”机制：发布版必须改为可追踪的拒绝，不得静默。
4. nan_to_num 的语义：改为显式错误统计和阈值中止。
5. Phase 2 的真实任务采样来源、C4 key/length、词表路径和 launcher dry-run；参数写在 shell 脚本中不等于真正生效。
6. joint mask 与 geo-only mask 的显式输出；否则不能证明两种监督操作按设计存在。

## 10. P0 的完成定义

只有同时满足以下条件，P0 才能标记为通过：

1. 实际训练 release 的数据、词表、tokenizer、checkpoint 和启动配置都有唯一 manifest；
2. 每条 accepted 记录通过全量 schema、原子、E3FP、motif map、文本完整性检查；
3. 任何被跳过、重算、截断、降级或拒绝的样本都可按 key 和原因追溯；
4. Phase 1 与 Phase 2 所有任务的 golden trace 和 forward/backward batch 都通过；
5. 两独立进程的 tokenizer、mapping、固定 seed trace 一致；
6. 训练日志和 checkpoint 能恢复相同的数据/词表/3D 规范；
7. 主干签字确认：数据流与研究命题一致，而不是“能跑但无法解释”。

在 P0 通过前，不应比较 lambda、EMA teacher、motif 划分或下游指标；这些实验会把数据/接口不确定性误判为算法差异。

## 11. P0 的实现约束：旁路验证，不复制主干

P0 的首要对象是**已经实现的训练主干**，即当前 `train1.py`、`train2.py`、`model/`、`dataset/`、`tokenization/` 和实际 LMDB；不能先复制整套实现后验证一个“看起来相同”的版本。

1. 先冻结主干文件与实际运行资产的 SHA-256。当前工作区存在未提交内容，故不能以一次 `git checkout` 或假定某个 commit 代表训练现场。
2. 新增 `p0_validation/` 只放 release manifest、旁路扫描、trace、hook、smoke 和报告；它直接调用主干的 Dataset、Collator、Tokenizer 和 Model。
3. 验证应只在四个语义边界封装：`record -> item`、`item -> batch`、`batch -> model forward`、`model output -> report`。不为每个 E3FP shell、Q/K/V 操作或 motif 分割细节创建独立替代实现。
4. 首个验收是 **baseline parity**：固定 record、tokenizer、checkpoint、seed 后，直接调用主干与经 `p0_validation` 调用主干的 token IDs、mapping、batch tensor、logits、CE/3D loss 必须相同（浮点使用预先规定容差）。
5. 只有 P0 表明主干接口确实无法表达所需语义时，才在 `most_t5_next/` 放入单一候选模块。例如显式输出 joint/geo-only masks、稳定词表或严格 Phase 2 mapping。每个候选模块先做 parity；有意修复的差异必须以化学真值和新 release manifest 说明。

这一分层参照 3D-MolT5 将 3D tokenization、模型、微调脚本与评测脚本分开的方式，但不照搬其原子级对齐逻辑。本项目的关键验证对象是 `atom-level E3FP -> atom_to_motif_map -> motif-level aggregation -> T5`，而不是强行要求 E3FP 行数等于 motif token 数。详见 [15_noninvasive_validation_architecture_and_compute_plan.md](15_noninvasive_validation_architecture_and_compute_plan.md)。

## 12. GPU 资源的 P0 决策门

P0 的全量 LMDB/schema/血缘扫描主要受 CPU、内存和 NVMe 影响，不应为此先租用 8 卡。GPU 资源按真实 profile 逐级升级：

| 阶段 | 推荐资源 | 进入下一档的条件 |
|---|---|---|
| 数据审计、manifest、golden case 选择 | 无 GPU；8–16 vCPU、64 GB RAM、NVMe | 通过完整性与血缘检查。 |
| 五类任务 forward/backward 与显存 profile | 优先 1×48 GB L40S/RTX 6000 Ada；24 GB 卡仅作 batch=1 冒烟 | 得到 p99/max 序列、原子数、显存和吞吐。 |
| 单卡原型 | 1×48 GB，24–32 vCPU、64–128 GB RAM | p99 完整路径峰值低于约 38 GB，且保留 20% 显存余量。 |
| 首轮完整预训练和系统消融 | 同机 4×A100 80 GB 或 H100 80 GB，128 GB RAM、1 TB NVMe | DDP/NCCL smoke 通过；以梯度累积保持可审计的全局 batch。 |
| 最终多 seed 主结果与复现 | 同机 8×H100 80 GB SXM/NVSwitch 优先；8×A100 80 GB 为稳妥替代 | 仅用于最终模型、关键消融和重复种子，而非未验证的代码调试。 |

当前 `t5-v1_1-base` 加入四层 E3FP embedding、motif-atom attention，以及额外一次无梯度 3D target 路径；Phase 1 还在长度 512、每卡 batch 64 且未显式使用 gradient checkpointing。因此脚本里的批大小不能直接视为可装入某种显存的事实。正式租卡前，必须分别 profile CE-only、fusion without 3D loss 和完整 MSE 路径；远端目前拒绝连接，实际数据 p99/max 与已有 GPU 尚待重新只读核验。
