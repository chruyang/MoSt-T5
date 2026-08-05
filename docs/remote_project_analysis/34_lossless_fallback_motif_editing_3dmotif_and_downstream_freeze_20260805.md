# 无损回退、motif 文本编辑、3D motif 假设与下游任务冻结裁决（2026-08-05）

状态：主干完成文献复核、官方实现对照与本地代码审查；本文是研究设计裁决，不表示 tokenizer 已冻结或 P1 已放行。

> 2026-08-05 后续限定：用户确认“无损”只针对 motif 离散身份，3D 状态采用 EMA teacher latent prediction。二者的统一实现与更新后的总体路线以 [35_unified_motif_identity_codec_ema_teacher_and_integrated_roadmap_20260805.md](35_unified_motif_identity_codec_ema_teacher_and_integrated_roadmap_20260805.md) 为准。

与前文关系：本文根据“Text-Based Editing 原本就是用于验证 motif—语言优势”的补充信息，修正文档 33 中把 Editing 降为可选应用的排序。文档 33 关于 anchor 因子化、词表规模和去污染的事实结论仍然有效。

## 0. 四个问题的直接结论

| 问题 | 结论 | 成立边界 |
|---|---|---|
| 1. 无损回退是否可行 | **有条件可行，但当前代码尚未实现。** | 只能承诺声明支持域内的“标准化二维化学图 round-trip”；不能承诺原始 SMILES 字符串逐字一致，更不能称 E3FP 或 motif 3D 状态无损。 |
| 2. Text-Based Editing 是否适合突出 motif 优势 | **适合，而且应恢复为核心机制任务。** | FineMolTex 是直接动机，但其协议是潜空间优化而非端到端 T5 编辑；需增加受控 motif 编辑协议，且 3D 主张需要几何相关指令和指标。 |
| 3. 原子 3D 与 motif LM 结合的总体想法是否成立 | **研究假设合理，已有多条直接或相邻文献链；效果并非逻辑必然。** | “首次 motif-level 3D”不成立。可研究的贡献是 CAMT5 对齐、E3FP 驱动、局部 atom-to-motif 聚合及身份/几何互补掩码。 |
| 4. 是否现在选择下游任务 | **是。现在应冻结任务 registry、官方 split 和 valid/test 保护集合；不等于现在执行全部训练。** | valid/test 绝不能参与词表、频率阈值或任何参数选择。推荐“预训练 train 构词表 + 确定性结构回退”；下游 train 入词表只能作为任务特定政策。 |

本文采用以下证据等级：

- `A`：同任务、同表示或非常接近的直接一手证据；
- `B`：机制相邻的一手证据，能支撑合理性但不能替代本项目实验；
- `C`：本项目假设，必须通过可证伪实验成立；
- `代码事实`：当前本地实现直接可见；
- `裁决`：综合证据后对本项目采取的设计决定。

## 1. 无损回退对当前算法是否可行

### 1.1 必须先限定“无损”的对象

“无损”至少有三个不同含义，不能混写：

1. **字符串无损**：解码后与输入 SMILES 的字符完全相同。该目标没有必要，因为同一分子可有多个等价 SMILES。
2. **化学图无损**：在声明支持的标准化域内，编码—解码后保持原子、键型、形式电荷、同位素和所声明的立体化学，并得到图同构或相同 canonical isomeric identity。这是本项目应采用的目标。
3. **三维无损**：从 token/E3FP/motif state 精确恢复原坐标或完整构象。当前方案做不到。E3FP 的哈希与 folding、跨 level 聚合、atom-to-motif pooling 均是多对一映射。

因此严谨表述应为：

> 在预先声明的标准化化学图支持域内，经全量及扰动 round-trip 验证的确定性结构回退。

在门禁通过前只称 `round-trip-oriented fallback`，不称 `lossless` 或 `invertible`。

### 1.2 文献为什么支持“高频 motif 宏 token + 稀有结构回退”

`A/B` 级直接先例是 [Group SELFIES](https://pubs.rsc.org/en/content/articlehtml/2023/dd/d3dd00012e)：表示可把常用子结构编码为 group token，同时仍保留基本原子、分支、环和 attachment-index 语法；作者在 2500 万个 eMolecules 分子上验证了编码—解码。该工作也明确说明立体化学不是自动保留的：若缺少必要的手性 group，round-trip 会丢失 chirality。这一点正好说明“无损”必须依赖明确支持域和测试，而不能依赖表示名称。

[ByT5](https://aclanthology.org/2022.tacl-1.17/) 证明 T5 架构可以在极低层级、开放词汇的 byte 序列上工作，说明“宏 token + 低层回退”在 Transformer/T5 中没有结构性冲突；其代价是序列变长和计算增加。它是语言模型层面的相邻证据，不是化学正确性证据。

[CAMT5](https://aclanthology.org/2025.findings-emnlp.1221/) 则证明 motif token 能改善 text-to-molecule 的结构上下文建模，但其公开实现没有独立的 motif OOV codec：未注册为 added token 的完整 motif 可能继续被 T5 SentencePiece 切分。因此 CAMT5 不能直接证明我们的 fallback 可逆。

综合起来，文献支持以下结构，而不是支持普通 `<unk>`：

```text
高频 motif:     <MOTIF_1234>
稀有 motif:     <FALLBACK_BEGIN> atom / bond / branch / slot / stereo ... <FALLBACK_END>
连接关系:       保留 motif 内 attachment slot 位置，分离分子局部 edge-pair label
```

### 1.3 当前代码为什么不能直接称为可行实现

| 当前事实 | 代码位置 | 对无损回退的影响 |
|---|---|---|
| 词表通过 `set` 收集后直接 `list(set)` 注册 | `tokenization/motif_tokenizer.py:79-94` | token ID 顺序不确定，破坏版本化 codec。 |
| 词表外 motif 被替换为单一 `<unk>` | `tokenization/motif_tokenizer.py:124-145` | 多个不同结构发生不可逆碰撞。 |
| 达到最大长度时在任意 token 位置截断 | `tokenization/motif_tokenizer.py:147-150` | 将来可能截断 fallback span 或连接对。 |
| decoder 假设一个 motif 恰好对应一个实体 token | `tokenization/motif_tokenizer.py:174-212` | 无法解析一个逻辑 motif 对应多个低层 token 的情况。 |
| 解码后主动移除立体化学 | `tokenization/motif_tokenizer.py:217-221` | 当前实现不满足含 stereo 的化学图 round-trip。 |
| 词表构建删除全部 anchor | `process/build_vocab_pipeline.py:105-120` | 删除了 attachment slot 的原子位置，不能仅靠外部 ID 唯一恢复。 |
| Collator 按单 token 采样、掩码和生成 label | `dataset/dataset2.py:308-375` | fallback span 会被部分掩码；长 span 在 CE 中得到不成比例的权重。 |
| 3D attention 的 query 来自 motif token embedding | `model/modeling.py:109-147` | 若所有稀有 motif 都使用同一 `<FALLBACK_BEGIN>` carrier，query 看不到其后 identity span，与高频 motif 不公平。 |

这意味着无损回退在**算法架构层面可行**，但在**当前代码实现层面不是一个配置开关**。

### 1.4 与当前 3D motif 融合兼容的最小数据合同

每个逻辑 motif 应至少保存：

```text
logical_motif_id
identity_span                 # 高频时长度为 1，fallback 时长度可大于 1
connection_span               # slot 与 edge-pair 信息
carrier_idx                   # 该 motif 承载 3D 融合的位置
state_idx                     # 3D loss 对齐位置
atom_indices                  # 属于该 motif 的原子
exact_graph_digest
```

必须同步修改：

1. **版本化 hybrid codec**：固定 token 排序、规范化规则、支持化学域和 checksum；未知合法结构不得落入 `<unk>`。
2. **逻辑 span**：Dataset/Collator 以 motif span 而非单 token 为基本单位。whole-span masking，禁止 span 中间截断。
3. **carrier 表征**：稀有 motif 的 carrier query 应由整个 identity span 池化得到，或显式比较“统一 begin token”与“span-aware carrier”；否则 attention 对高频/稀有 motif 的条件信息不对称。
4. **CE 口径**：标准 token-mean CE 会让长 fallback motif 贡献更多 loss。至少报告高频/回退分层 token CE；候选主方案是按逻辑 motif 对 span 内 CE 做归一化，再与标准 CE 进行短跑比较。
5. **严格解码器**：按语法解析 atom/bond/slot/stereo，slot 必须成对且 attachment 位置可复原；语法错误 fail closed，不能静默跳过。
6. **3D 对齐**：无论 motif 用 1 个还是多个 identity token，每个逻辑 motif 只产生一个 geometry target；atom-to-motif 映射指向 carrier/logical ID，而不是任意子 token。

R1 sidecar 已保存 exact motif digest、atom groups 和 E3FP，通常无需重算 PCQM4Mv2 几何；需要新增的是 tokenizer-bound record 构建层。

### 1.5 “无损”成立前的硬门禁

| 门禁 | 接受条件 |
|---|---|
| 化学图 round-trip | 支持域内 100% 保持声明的原子/键/电荷/同位素/stereo；所有拒绝样本有 manifest。 |
| 无碰撞 | 不出现 `<unk>`、normalization collision 或不同图映射到同一 codec 序列。 |
| 连接一致性 | 每个跨 motif edge 恰有两个端点；slot 原子位置、键型和配对均恢复。 |
| 扰动不变性 | atom renumbering、等价 SMILES、motif traversal/root choice、edge-ID 重命名后得到相同规范化语义。 |
| 长度安全 | 报告 molecule-level P50/P95/P99、超过 max length 比例；不允许中途截断合法结构。 |
| 模型接口 | identity span、carrier、atom mapping、mask、CE 和 geometry target 的张量断言全部通过。 |
| 解码生成 | 生成序列严格解析；分别报告 syntax validity、chemical validity、round-trip exactness 和拒绝率。 |

**裁决**：采用该方向，但把它列为 tokenizer/P1 前置工程与科学门禁。当前 top-32k 的高 occurrence coverage 只说明压缩性好；正则表达式能切开字符串也只说明词法覆盖，二者都不能证明化学图无损。

## 2. Text-Based Editing 是否应围绕 motif 重新定位

### 2.1 用户原始动机与 FineMolTex 高度一致

[FineMolTex](https://smufang.github.io/paper/KDD25_FineMolTex.pdf) 把 text-based molecule editing 明确归为 “motif-centered task”：文本可能要求替换与性质或名称相关的 motif，因此细粒度 motif—word 对齐对任务重要。其预训练同时学习分子级对比对齐与 motif/word 级 masked multimodal modeling；论文的可解释性分析也显示 motif 名称和相关文本词在联合空间中相邻。

其公开评测事实是：

- 从 ZINC 随机取 200 个分子；
- 12 条固定提示，其中 8 条是 LogP/QED/tPSA 相关性质方向，4 条是 motif 名称；
- motif 指令用 RDKit 检查目标子结构是否出现；性质指令以输出相对输入的描述符变化是否超过阈值判定；
- 对 4 条 motif-name 指令的提升尤其明显；
- 生成不是 FineMolTex 自身端到端解码：先把独立生成模型的潜空间与 FineMolTex 联合空间对齐，再逐样本优化潜变量，使其接近文本、同时通过 L2 保持接近原分子。

因此 FineMolTex 对我们提供的是 `A` 级**任务动机证据**和 `B` 级**实现先例**：它直接支持“编辑能显化 motif—语言知识”，但不直接支持“当前 T5 decoder 能做局部、锚点可控、3D 保持的编辑”。

### 2.2 为什么只复现 FineMolTex 仍不足以支撑顶刊主结论

1. 200 × 12 是小规模固定提示评测，容易受候选采样与超参数搜索影响。
2. 8 个性质提示主要验证描述符方向，不隔离 motif 表示的贡献；真正直接针对 motif 的只有 4 条。
3. 仅检查“目标 motif 出现”不能保证原 scaffold、非编辑区域、连接位置或立体化学被保持。
4. 该协议依赖独立生成器和逐样本优化，不能与“统一 T5 直接执行编辑”混为一项能力。
5. 目标子结构出现也不能证明使用了 3D motif；纯 2D motif 模型可能完成同一任务。

### 2.3 推荐的双协议设计

#### E1：FineMolTex-compatible 协议

目的：与已有 motif—text 模型横向比较，确认项目至少复制其“motif 名称编辑优势”。

- 固定相同的 12 类提示和分子筛选规则；
- 预先冻结候选数、采样温度和超参数，不允许“任一候选成功即整样本成功”而不同时报告 pass@1；
- 报告 pass@1、pass@k、有效率和每条提示的结果；
- 单独汇总 4 个 motif-name 指令，不能用 8 个性质任务的平均值代替 motif 结论。

#### E2：本项目的 anchor- and 3D-aware 受控编辑协议

输入应显式包含：

```text
source molecule + natural-language instruction
    -> edited molecule
    -> target motif / operation / attachment slot / bond type
    -> optional geometric-state constraint
```

任务至少分为：

1. motif add；
2. motif delete；
3. motif substitute；
4. 保持 scaffold 的性质方向编辑；
5. 几何条件编辑，例如指定立体构型、局部扭转区间或“保持其余 motif 构象”。

公开的 [TOMG-Bench](https://arxiv.org/abs/2412.14642) 和 [MolLangBench](https://openreview.net/forum?id=KbXl2jfFRn) 可作为更大规模自然语言分子编辑协议的参考；若其许可、数据版本或任务定义不适合当前 tokenizer，则可用 RDKit/反应模板构造确定性 paired edits，但必须公开规则、失败样本和 split。 [MoleculeSTM](https://www.nature.com/articles/s42256-023-00759-6) 是 FineMolTex 所沿用的潜空间编辑直接先例，可作为 E1 基线而非 E2 的同构实现。

### 2.4 E2 必须报告的指标

| 维度 | 指标 | 防止的错误结论 |
|---|---|---|
| 生成正确性 | strict parse rate、RDKit validity、canonical exact match（有唯一目标时） | 目标属性提高但结构语法无效。 |
| 指令遵循 | add/delete/substitute success；性质阈值成功率 | 仅生成相似分子而未执行指令。 |
| 局部性 | 未编辑原子/键保持率、scaffold retention、最小 graph-edit distance | 通过大幅改写分子“取巧”。 |
| anchor | attachment atom、slot、edge-pair、bond type accuracy | 目标 motif 存在但接错位置。 |
| 化学细节 | charge/isotope/stereo preservation | 当前 decoder 丢立体信息却被有效率掩盖。 |
| 3D | 构象生成成功率、局部 RMSD/TFD、关键扭转和碰撞/应变审计 | 用纯 2D 改写冒充 3D motif 能力。 |
| 多样性 | success-conditioned uniqueness/diversity | pass@k 通过大量近重复候选堆出。 |

核心因果比较应控制相同数据、生成器、参数量和候选预算：

```text
atom/SELFIES baseline
CAMT5-style 2D motif
本项目 2D motif + anchor factorization
本项目 motif + E3FP 3D state
```

**裁决**：Text-Based Editing 恢复为论文的核心机制任务，但不应成为唯一公共下游证据。E1 负责与 FineMolTex 对齐；E2 才负责检验本项目的 motif、anchor 和 3D 机制。

## 3. “把原子 3D 与 motif 语言模型连接起来”是否是合理主线

### 3.1 文献证据链是完整的，但不是唯一性证明

| 层级 | 一手工作 | 对本项目的支持 | 不能推出的结论 |
|---|---|---|---|
| 原子中心局部 3D | [E3FP](https://pmc.ncbi.nlm.nih.gov/articles/PMC6075869/) | 给定构象下，以原子为中心逐 shell 产生 alignment-invariant、构象敏感的局部 3D 标识，并可包含非键合空间邻居与相对取向。 | 不支持 motif pooling 最优，也不支持单构象代表分子全部三维状态。 |
| E3FP + T5 | [3D-MolT5](https://proceedings.iclr.cc/paper_files/paper/2025/file/3d4dc72d715bd6415d356293079adf3d-Paper-Conference.pdf) | 将 atom-centered E3FP token 与原子级 SELFIES 对齐并融入 T5，直接证明接口方向可行。 | 仍是原子粒度；其 hashing/folding 也不是无损表示。 |
| motif + 文本 | [CAMT5](https://aclanthology.org/2025.findings-emnlp.1221/)、[FineMolTex](https://smufang.github.io/paper/KDD25_FineMolTex.pdf) | 支持 motif token 的结构上下文与 motif—word 细粒度对齐价值。FineMolTex 使用经过 2D/3D 多视图预训练的 GraphMVP 原子编码器，再把原子表示平均池化成 motif 表示。 | FineMolTex 推理时不是显式输入指定构象；两者也不能证明当前 CAMT5 划分适合几何聚合或 E3FP attention 优于均值。 |
| 显式 motif + 3D | [Molformer](https://ojs.aaai.org/index.php/AAAI/article/view/25662) | 在 3D 异质分子图中同时使用 atom 与 motif 节点，证明 motif/3D 多尺度表示有直接先例。 | 不是 T5、不是 E3FP，也不是本文的 masking 机制。 |
| fragment 级 2D/3D | [HoliMol](https://openreview.net/pdf?id=ufDh55J1ML) | 将 3D-GNN 原子状态聚合到 fragment，进行 fragment-level 2D/3D 对比，并预测相邻 fragment 扭转角；是最接近的机制先例。 | 使用 BRICS 和连续 3D-GNN，不能证明 CAMT5 motif 边界。 |
| 对称性与长程消息 | [LSR-MP](https://proceedings.iclr.cc/paper_files/paper/2024/hash/a3eadeebbc9eecd621086f6978865a85-Abstract-Conference.html) | 说明在合适的不变权重下，原子标量/向量可以聚合成保持不变性/等变性的 fragment 表示，并加入 atom-fragment 长程交互。 | 本项目没有向量等变通道，不能援引其结论把当前模型称为 E(3)-equivariant。 |

因此原始推理：

```text
原子级 3D 表示已有效
motif 级分子—语言表示已有效
motif + 3D 多尺度建模也有先例
------------------------------------------------
在统一 T5 中研究 atom-centered 3D -> CAMT5 motif 的局部聚合，是合理且可证伪的研究问题
```

最后一行是**研究假设**，不是由前三行必然推出的效果保证。

### 3.2 当前实现准确表达了什么

当前机制可概括为：E3FP 把每个原子的多个 shell ID 映射为 embedding；motif token 作为 query，只对映射到自己的原子 key/value 做 attention pooling，再通过 gate 融回 T5。

在 E3FP 输入本身对平移和 proper rotation 不变的前提下，后续只操作标量 embedding、softmax 权重与加权和，因此当前 pooled state 应称：

> SE(3)-invariant, motif-conditioned aggregation of atom-centered E3FP identifiers.

它不是 E(3)-equivariant encoder，也不应称 reflection-invariant；启用 stereochemistry 的 E3FP 本来就可能区分镜像立体异构体。

更重要的是，E3FP shell 会覆盖 motif 外的成键邻居与空间邻近非键合原子。故“3Dmotif”更准确的操作定义是：

> 以该 motif 所属原子为中心、受分子环境条件化的局部三维状态。

这比“纯 motif 内部几何”严谨，也与代码真实操作一致。

### 3.3 当前方案仍需证明的四个独立假设

| 假设 | 必需对照 | 若失败应如何解释 |
|---|---|---|
| H1：motif 粒度比原子粒度更有利于语言对齐/编辑 | atom-E3FP T5 vs 2D motif vs motif-E3FP；相同参数/数据预算 | motif 粒度未带来可测优势，不能以理论动机代替结果。 |
| H2：收益来自 3D，而非 E3FP 中附带的原子身份和拓扑 | 用 ECFP/E2FP 替代 E3FP并保持同一聚合器 | 若相当，则只能称额外局部结构特征，不能称几何收益。 |
| H3：local attention 比简单聚合必要 | mean-pooled E3FP vs 当前 attention | 若不优，采用更简单的均值，避免把复杂度写成贡献。 |
| H4：CAMT5 motif 划分适合几何聚合 | 当前划分 vs BRICS/功能团或 size-matched 对照 | 若不优，只能把 CAMT5 分区视为序列 baseline，不能称 3D-optimal motif。 |

### 3.4 最值得先查的风险

1. **跨 motif shell 泄漏**：遮掉 motif A 的中心原子 E3FP 后，邻近 motif B 的可见 shell 仍可能含 A 的原子信息。应统计 shell-overlap，并比较 center-only mask 与 shell-aware halo mask。
2. **哈希与聚合碰撞**：raw identifier 折叠到 4096、level 求和和 motif pooling 均可能把不同几何映成相同状态。
3. **单构象问题**：E3FP 原论文按每个构象生成 fingerprint，并讨论柔性分子的困难；PCQM4Mv2 的一个计算构象是可扩展近似，不是分子唯一真实构象。
4. **MSE 度量语义**：learned E3FP embedding 的欧氏距离不天然等价于 RMSD、TFD、扭转角或能量差；因此 MSE 只能称 E3FP-derived latent-state prediction。
5. **缺少显式跨 motif 几何**：HoliMol 预测 fragment 间扭转，LSR-MP 保留 fragment 坐标/长程消息；当前模型对相对姿态只能间接建模。

### 3.5 资源受限情况下的最小证据阶梯

先做无需训练或 CPU 小样本门禁：

- 刚体旋转/平移后 E3FP ID 与 pooled state 一致；
- atom renumbering 后，映射同步变化时 pooled state 一致；
- 镜像立体异构体按预期区分；
- raw ID → 4096 的 type/occurrence collision；
- motif 外原子进入 shell 的比例与 mask leakage 比例；
- 1,000–5,000 个柔性分子的多构象 E3FP/pooled 距离，与 TFD、内部距离、扭转及相对能量的相关性。

训练只按门禁逐级进入：

| 条件 | 目的 |
|---|---|
| M0：motif-only | 2D 主干基线。 |
| M1：motif + ECFP/E2FP，同聚合器 | 排除额外拓扑信息。 |
| M2：motif + mean-pooled E3FP | 证明简单 3D 聚合是否已足够。 |
| M3：motif + attention-pooled E3FP | 仅在 M2 有收益后证明 attention 的增量。 |

短跑阶段一个 seed 只用于淘汰；最终 Full 和最关键因果对照再补独立预训练 seed。另做低成本机制扰动：分子内打乱 atom-to-motif、全局广播 E3FP、跨分子打乱 E3FP、center-only/halo mask 对比。

**裁决**：整体思路不需要推翻，但创新表述必须收窄。建议论文候选表述为：

> 我们提出一种与 CAMT5 motif 序列一一对齐的、SE(3) 不变的原子中心 E3FP 局部聚合机制，并在统一 T5 中研究 motif 身份与其上下文化三维状态的互补掩码学习。

在实验成立前不要使用“首次 motif-level 3D”“无损三维表示”“重建真实几何”或“最优 motif 划分”。

## 4. 是否现在确定下游任务和 vocabulary 政策

### 4.1 现在必须确定任务，但不必现在跑完任务

现在冻结下游任务的主要原因有三个，词表只是其中之一：

1. **去污染**：必须在 P1/P2 membership 和 vocabulary discovery 前知道哪些 valid/test 分子受保护。
2. **表示覆盖**：在冻结 tokenizer 前统计每个 split 的 motif OOV、fallback 比例、序列 P95/P99 和 round-trip，而不是训练后发现测试集不可表示。
3. **预注册主张**：提前指定哪些任务验证语言、几何、连接和生成，避免训练完成后从大量任务中挑最好结果。

这不等于把所有数据全部物化、批量 tokenization 或立刻消耗 GPU。现在应保存的是任务 registry、split hash 和 identity manifest。

### 4.2 FineMolTex 与 CAMT5 代表两种不同、都可解释的词表政策

| 工作 | 可确认的词表来源 | 适用解释 |
|---|---|---|
| FineMolTex | 从 PubChemSTM 预训练语料提取全部 30,080 个 unique motifs；用于 masked classification 的集合按频率筛到 2,457 个 | 更接近“预训练词表 + 下游零样本/迁移”，但它是图表示模型，不提供我们的序列 fallback 保证。 |
| [CAMT5](https://aclanthology.org/2025.findings-emnlp.1221/) 与[官方代码](https://github.com/Songhyeontae/CAMT5) | 静态 motif vocabulary 使用 ChEBI-20 train、额外 34k PubChem train，并在 PCDes 阶段并入 PCDes train motifs；官方资产最终约 24,735 个 | 证明使用**下游 train** 构建任务相关 tokenizer 可以作为监督 text-to-molecule 协议，但 tokenizer 随任务变化，不能等同严格零样本或统一 foundation tokenizer。 |

因此文献并没有唯一答案；选择取决于论文声明。

### 4.3 本项目推荐的主政策

推荐主线采用：

> **clean pretraining-train vocabulary + deterministic graph fallback + one frozen tokenizer across P1/P2/all downstream tasks**

理由：

- 不需要看到下游 valid/test 就能表示新 motif；
- 能把 OOV/fallback 本身作为组合泛化指标；
- 不因切换下游任务而扩 embedding，避免 checkpoint 不兼容；
- 更容易主张统一预训练模型和严格的零样本/迁移能力。

允许把“加入指定下游 train motifs”的 CAMT5-style policy 做成一个小规模敏感性对照，但必须命名为 `task-aware tokenizer`，且只使用官方 train。任何 valid/test motif、频率、长度或结果都不得参与词表选择。

### 4.4 建议现在冻结的任务组合

用户原定四项是：Text-Based Editing、Zero-Shot Retrieval、Molecule Captioning、Property Prediction。根据补充动机，建议不是删除 Editing，而是替换 Retrieval：

| 论文位置 | 推荐任务 | 主要证明对象 | 备注 |
|---|---|---|---|
| 核心机制 | **Motif Text-Based Editing** | motif—language 对齐、anchor/连接、局部编辑；加入 3D 指令后检验几何机制 | E1 与 FineMolTex 对齐，E2 承担项目特异主结论。 |
| 公共生成基准 | **Text-to-Molecule Generation（ChEBI-20，资源允许再 PCDes）** | decoder、静态词表、fallback 和生成广度 | CAMT5 的直接比较项，也是发现 vocabulary 问题最敏感的任务。 |
| 核心跨模态 | **3D Molecule Captioning** | molecule/3D → text | 除 BLEU/ROUGE 外需事实正确性、幻觉与 no-3D 对照。 |
| 核心几何证据 | **Property Prediction + conformer/geometry probe** | 3D state 是否带来可归因收益 | QM9 HOMO/LUMO/gap 可先做，但必须加同分子多构象或扭转/立体 probe，避免 2D 捷径。 |
| 次级诊断 | Molecule–Text Retrieval | 跨模态全局对齐 | 只有冻结双塔/相似度评分且无任务调参时才称 zero-shot；资源不足时移至附录。 |

若图中只能放四个下游框，建议使用：

```text
Motif Text-Based Editing
Text-to-Molecule Generation
3D Molecule Captioning
3D-sensitive Property Prediction
```

Retrieval 作为 secondary diagnostic。原因不是它没有价值，而是它对当前以生成式 T5 CE 为主的模型不是天然接口；相比之下，text-to-molecule 会直接暴露词表、fallback 和 decoder 的真实问题。

### 4.5 冻结顺序

```text
T0  冻结候选任务 registry、版本、许可、官方 split hash
 ↓
T1  构建全部 valid/test connectivity-identity 保护并集
 ↓
T2  从 P1/P2、tokenizer discovery、频率统计和参数更新语料中排除保护集
 ↓
T3  在 train / P1 held-out / P2 / 各下游 split 上审计 OOV、fallback、长度、round-trip
 ↓
T4  选定 top-16k vs top-32k，冻结一个版本化 tokenizer
 ↓
T5  P1/P2 训练；之后按预注册顺序做下游和机制评测
```

每个任务现在至少保存：

- 数据集名称、来源、版本、许可证和下载校验值；
- 原始官方 split 与不可变行级 hash；
- standardized isomeric SMILES、connectivity identity、stereo identity；
- valid/test 全局保护 manifest；
- train/valid/test 的 motif type/occurrence OOV、fallback、长度和 round-trip 报告；
- 文本 exact/near-duplicate、分子名称和 target leakage 审计规则；
- 主要指标、主对照、seed 和候选预算。

[3D-MolT5](https://arxiv.org/html/2406.05797) 在 ChEBI-20 相关实验中明确处理了与 PubChem 预训练 pairs 的测试重合，这至少说明在大规模分子预训练中，提前定义下游保护边界是必要的；公开材料不足以证明它系统过滤了所有下游，因此本项目应采用更完整且可复核的 manifest，而不是简单声称“遵循 3D-MolT5”。

## 5. 对当前 P1 的直接影响

在以下四项完成前，`P1=false` 保持不变：

1. 四个核心候选任务及 retrieval 次级任务的 registry、official split 与 valid/test 保护并集冻结；
2. hybrid codec 和逻辑 motif span 数据合同确定，并解决 P1/P2 的 stereo 与 anchor 语义差异；
3. 全量或规定支持域内 round-trip、slot pairing、长度和 OOV/fallback 门禁通过；
4. Dataset/Collator/model smoke test 证明高频单 token 与稀有 span 在 masking、CE 和 3D carrier 上语义一致。

这些工作主要是 CPU/代码审计，不需要现在恢复 GPU，也不需要把下游完整训练数据全部下载到本地。远端只需保存不可变 registry、manifest、统计和小样本测试产物。

## 6. 最终研究定位

四个问题合在一起后，主线应写成：

> 本项目不是简单把 3D 特征加入 motif T5，而是构造一个可组合、可严格解码的 motif 分子语言：高频结构以 motif 宏 token 表达，长尾结构以保留 attachment slot 的确定性化学语法回退；原子中心 E3FP 状态只在对应逻辑 motif 内聚合，并通过身份/几何互补掩码学习。公共生成与 3D 下游验证整体能力，受控自然语言编辑专门检验 motif、anchor 与局部三维状态是否真正被使用。

其价值具有合理文献链，但能否成为顶刊级贡献取决于四项实验事实：回退是否真能 round-trip、3D 是否超越 topology-only、local motif 对齐扰动后是否显著下降、编辑是否在保持非目标结构的同时正确执行 motif/anchor/geometry 指令。

## 7. 主要一手资料

- [Group SELFIES: a robust fragment-based molecular string representation](https://pubs.rsc.org/en/content/articlehtml/2023/dd/d3dd00012e)
- [ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models](https://aclanthology.org/2022.tacl-1.17/)
- [CAMT5: Training Text-to-Molecule Models with Context-Aware Tokenization](https://aclanthology.org/2025.findings-emnlp.1221/)
- [CAMT5 official code](https://github.com/Songhyeontae/CAMT5)
- [FineMolTex / Advancing Molecular Graph-Text Pre-training via Fine-grained Alignment](https://smufang.github.io/paper/KDD25_FineMolTex.pdf)
- [FineMolTex official code](https://github.com/liushiliushi/FineMolTex)
- [E3FP: A Simple Representation of Three-Dimensional Molecular Structure](https://pmc.ncbi.nlm.nih.gov/articles/PMC6075869/)
- [3D-MolT5](https://proceedings.iclr.cc/paper_files/paper/2025/file/3d4dc72d715bd6415d356293079adf3d-Paper-Conference.pdf)
- [Molformer: Motif-Based Transformer on 3D Heterogeneous Molecular Graphs](https://ojs.aaai.org/index.php/AAAI/article/view/25662)
- [HoliMol: Holistic 3D Molecular Representation Learning](https://openreview.net/pdf?id=ufDh55J1ML)
- [LSR-MP: Long-Short-Range Message-Passing](https://proceedings.iclr.cc/paper_files/paper/2024/hash/a3eadeebbc9eecd621086f6978865a85-Abstract-Conference.html)
- [MoleculeSTM](https://www.nature.com/articles/s42256-023-00759-6)
- [TOMG-Bench](https://arxiv.org/abs/2412.14642)
- [MolLangBench](https://openreview.net/forum?id=KbXl2jfFRn)
