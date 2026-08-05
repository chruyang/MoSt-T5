# 下游数据、防泄漏、最小消融与锚点因子化决策（2026-08-05）

状态：主干裁定完成；当前机器无卡即可继续执行数据身份、碰撞与可逆性门禁，但 **不得冻结 tokenizer，也不得启动 P1**。本文件回答下游任务、去重边界、资源受限消融，以及锚点表示/辅助任务四组问题。

## 0. 结论先行

1. **现在就准备下游集，但只准备“数据合同和身份清单”**：冻结来源、版本、许可证、官方 split、文件哈希和分子身份；在锚点表示修复前，不批量生成 motif/E3FP 特征，也不启动下游训练。
2. **防泄漏应清理预训练侧，而不是删除官方下游测试集**：保留官方 downstream train/valid/test；把 downstream valid+test 中连接关系等价的全部分子、构象和文本配对从 P1/P2 的训练 membership、词表发现和频率统计中排除。原始文件不删除，只生成 exclusion manifest 与 clean view。
3. **不做全组合消融**：CPU 完成零训练证明；4090 上先做单 seed、预注册固定数据与有效 token/更新步数的短程筛选。正式预算只验证被保留的一个核心因果关系；最终最优模型再覆盖三个任务族。
4. **“使用锚点”本身不是充分创新点**。CAMT5、层次分子生成和 motif 生成模型都已有碎片连接/attachment point。当前更可能成立的贡献是：**位置保持的 motif 化学—attachment slot—跨 motif edge pairing 因子化序列，与 motif 级 3D 状态融合及互补掩码联合建模**；是否能成为论文贡献，取决于可逆性、效率和消融证据。
5. **当前锚点投影存在已实测的信息损失**：在 441,769 个 P1 exact motif lexeme 上，删除内联锚点后共有 604 个歧义投影键，涉及 1,210 个 exact lexeme。必须改成“原位置槽位模板”，即把每个 `<n*>` 原位替换为 `<*>`，并将按出现顺序排列的 ID 单独编码。
6. **锚点辅助任务有条件接受为低成本候选，而非立即纳入主线**：修复可逆表示后，优先测试同时破坏至少两条边的 `Masked Anchor Pair Assignment`，继续使用 T5 的 CE，不新建 loss/head；先排除计数、遍历顺序和人为编号捷径，再决定是否保留。

证据标签：`[实测]` 为本项目数据/代码运行结果；`[代码]` 为直接源码证据；`[文献]` 为原论文或官方资料；`[裁定]` 为基于前述证据的实验设计决策。

## 1. 3D-MolT5 实际做了哪些下游任务

### 1.1 主文任务

[3D-MolT5 论文](https://arxiv.org/html/2406.05797)的主要下游实验均被转换为 instruction text-to-text，并用标准 token CE 微调；论文没有为这些任务单独增加回归头。其 Table 11 给出的数据规模如下：

| 任务族 | 数据集 | train / valid / test（instruction rows） | 主要输出 |
|---|---|---:|---|
| 数值型分子问答 | PubChemQC | 2,463,404 / 308,024 / 308,248 | 数值文本 |
| 数值型分子问答 | QM9 | 347,774 / 1,928 / 1,928 | 多种量化性质的数值文本 |
| 数值型分子问答 | PubChem | 46,532 / 3,885 / 7,746 | 数值文本 |
| 描述型分子问答 | PubChem | 59,775 / 4,980 / 9,940 | 自然语言回答 |
| 3D 分子描述 | PubChem | 11,955 / 996 / 1,988 | 分子 caption |
| 文本生成分子 | ChEBI-20 | 26,407 / 3,301 / 3,300 | SELFIES/分子 |

注意：表中是 E3FP 处理后的 instruction 样本数；论文说明无法被 E3FP 处理的记录已被丢弃，因此它既不等于原始数据集规模，也不能直接当成去重后的 unique molecule 数。例如同一分子可对应多个性质问题。去重与泄漏审计必须回到 molecule identity，而不是按训练行文本比较。

### 1.2 附录扩展任务

论文还报告了：

- USPTO-50k retrosynthesis；
- MoleculeNet 的 BACE、BBBP、HIV、ClinTox；
- Mol-Instructions 的 reagent prediction、forward reaction prediction 与 retrosynthesis。

这些任务扩大了通用性证据，但并非验证本项目“motif 3D 状态 + 双掩码”的首要任务，当前资源条件下应后置。

### 1.3 一个需要纠正的边界

3D-MolT5 **没有报告自身的 molecule-text retrieval 或 zero-shot retrieval 下游结果**；表中的 zero/few-shot 是 Llama/GPT 等生成基线，Text2Mol 是 ChEBI-20 生成质量评价指标。[3D-MoLM](https://arxiv.org/html/2401.13923) 等相关工作包含分子—文本检索，但其相应 checkpoint 仍经过任务数据微调，不能据此把任务称为 zero-shot。因此，本项目不应为了“复现 3D-MolT5 下游”现在额外准备 retrieval pipeline。

## 2. 下游集何时准备、准备到什么程度

### 2.1 现在（无卡阶段）必须完成

每个拟用下游数据集建立不可变 registry：

1. 数据集名称、release/loader 版本、官方 URL、许可证/使用条件；
2. 原始文件相对路径、字节数、SHA-256；
3. 官方 train/valid/test membership，且保留原始 row ID；
4. split 算法与 seed、membership SHA-256，以及 instruction row 到 molecule ID 的映射；
5. 每个分子的多层身份键：
   - 原始 ID/CID；
   - 原始 SMILES digest；
   - 规范化、保留立体化学的 identity；
   - 规范化、忽略立体化学的 connectivity identity；
   - Bemis–Murcko scaffold（只用于报告和分层，不默认据此删除）；
   - 原始 3D record 字节 digest，以及在固定原子顺序与坐标量化规则下的 conformer/state digest；
   - 规范化文本与 molecule-text pair digest。
6. preprocessing 代码 commit、RDKit/标准化工具版本、identity schema SHA、text-normalization 版本；
7. 与 P1、P2 的 overlap report 和用于训练视图的 exclusion manifest。

上述工作只需要 CPU/内存与顺序 I/O。现在做的价值是：在 tokenizer 词表发现和预训练 membership 固定前切断验证/测试泄漏，而不是等训练结束后才发现无法补救。

### 2.2 现在不做

- 不在锚点序列化合同未通过前批量生成下游 motif token；
- 不先算全量 E3FP/3D cache；
- 不为反应、检索和全部 MoleculeNet 任务搭建训练环境；
- 不下载到本地；数据与审计产物继续只在远端工作区和持久化区流动。

## 3. 去重与防泄漏协议

### 3.1 主规则：保护 valid 和 test，清理预训练 membership

[裁定] 在本项目尚未开始正式 P1 训练的情况下，采用以下优先级：

`downstream test > downstream valid > downstream train > pretraining membership`

- 保护集合是**所有已选下游任务 valid/test connectivity identity 的全局并集**；若某分子在任务 A 为 train、在任务 B 为 test，仍从全部 P1/P2、replay、词表发现和辅助预训练语料中排除；
- 官方 downstream valid/test 保持不动，以保留公开基准的可比性；
- 对其 connectivity identity 命中的 P1/P2 记录，排除所有立体异构/构象/文本配对版本；
- 排除发生在训练 view 层，不删除 PCQM4Mv2、3D-MoLM 或 3D-MolT5 原始数据；
- 相同 exclusion 必须应用于 motif vocabulary discovery、频率阈值选择和任何数据驱动 tokenizer 绑定；否则测试分子仍参与表示设计；
- downstream train 与 pretraining 的重合可以保留，但必须报告 unique molecule 数与比例。

保护 valid 不只是“更严格”：valid 参与 early stopping、超参数选择和模型选择，泄漏同样会使最终结论乐观。

### 3.2 为什么不是从下游中删掉重合样本

3D-MolT5 报告排除了 ChEBI-20 test 与 PubChem 3D molecule-text 预训练 pairs 的重合分子，并移除文本中的分子名称，但公开文字没有给出可复核的删除侧、ID 清单或预训练清洗代码。标准 test 规模未显示缩减，与预训练侧排除相容但不足以判定方向；详见文档 33 的证据裁定。无论原论文实际采用哪一侧，我们的正式预训练尚未启动，因此应选择可复核的预训练侧清理。

ChEBI-20 应保存两个零额外训练成本的评测视图：

- `reported-protocol approximation`：按论文文字尽可能重建 3D-MolT5 的 ChEBI overlap 处理；拿不到精确 exclusion manifest 时不得称为严格复现；
- `clean-full-official`：保留完整官方 test，同时从本项目所有预训练 membership 中清除命中分子。

前者负责与论文数值可比，后者负责本项目的 clean/unseen 结论；二者不能混称为同一协议。

大规模分子预训练中，严格移除下游评估分子已有直接先例。[MolE](https://www.nature.com/articles/s41467-024-53751-y)在自监督与监督预训练数据中移除了 TDC 测试分子，并报告了高重合数据集带来的风险。

### 3.3 身份粒度

| 粒度 | 用途 | 是否作为默认排除键 |
|---|---|---|
| source ID/CID | 溯源、快速 exact join | 否；不同来源 ID 不可靠 |
| stereo-aware normalized identity | 精确同一立体分子报告 | 辅助 |
| normalized connectivity identity | 跨来源同连接关系匹配 | **是** |
| conformer/state digest | 识别同一分子的不同 3D 状态 | 随其 connectivity 一并排除 |
| Bemis–Murcko scaffold | scaffold overlap 与难度分析 | 否；默认全删会改变基准分布 |
| normalized text/pair digest | 名称、描述和配对泄漏 | 独立审计 |

对互变异构、盐、溶剂和电荷状态的标准化必须版本化；不能把一次 RDKit canonical SMILES 当成唯一真值。最终 overlap report 至少同时给 source-ID、stereo、connectivity 与 scaffold 四档计数。

### 3.4 下游自身的重复

- instruction QA/caption 必须按 molecule group 划分，不能把同一分子的不同问题随机拆到不同 split；
- 若沿用官方 split，先完整复现官方结果，再提供 clean view；clean view 只根据身份键、不查看标签，按 `test > valid > train` 保留最高优先级 split 并从较低优先级 view 移除重复记录，同时公开各 split 受影响数量；
- 官方未清洁视图只承担 published comparability，clean view 才承担 unseen-molecule 主结论；
- 数值性质出现冲突标签时，先按 `stereo identity + endpoint/property + unit + calculation/assay level + conformer` 分类；只有同类重复才可按预注册规则聚合，异类记录禁止静默平均或任意删一条；
- 若因公开协议无法排除预训练重合，必须把该结果标注为 `overlap-permitted / non-decontaminated reproduction`，不能称为 unseen-molecule generalization，也不能仅因存在重合就自动使用 transductive 术语。

### 3.5 PCQM4Mv2 与 PubChemQC 的特殊风险

[OGB 官方说明](https://ogb.stanford.edu/docs/lsc/pcqm4mv2/)指出 PCQM4Mv2 来自 PubChemQC，并提供 DFT 结构/性质。由于本项目 P1 使用 PCQM4Mv2，而 3D-MolT5 的旗舰数值下游之一也是 PubChemQC，二者属于同一来源族。正式使用 PubChemQC 下游前必须以 connectivity 为主排除键做全量 join，以 stereo/3D state 分层报告，CID 只作为 lineage 辅助键。没有 overlap proof 时，只能把 PubChemQC 标为 overlap status unknown，不能承担“未见分子泛化”的主结论。

P2 来自 PubChem/3D-MoLM，PubChem caption 与 descriptive downstream 也有同源风险，必须通过同一全局保护集合门禁，不能只审计 PCQM4Mv2–PubChemQC。

## 4. 资源受限下的下游任务最小集合

### 4.1 现在建立 registry 的核心集合

| 优先级 | 数据集/任务 | 作用 | 当前处理深度 |
|---|---|---|---|
| A | QM9 HOMO/LUMO/gap 等 | 小规模 3D 相关数值诊断；只有 no-3D/geometry-loss 对照才能归因几何收益 | 完成 identity/split/leakage；后续最先短跑 |
| A | PubChem 3D caption | 检验带 3D 输入的 molecule-to-text 能力；需 no-3D 对照隔离 3D 贡献 | 完成 identity/split/leakage |
| A | ChEBI-20 text-to-molecule | 检验文本到分子与无 3D 路径是否被破坏 | 完成 identity/split/leakage与名称泄漏审计 |
| B | PubChemQC 数值 QA | 在相同公开 split/处理协议下提供 3D-MolT5 参照；clean protocol 需单独报告 | 现在只做全量 overlap proof；预算允许再训练 |
| C | MoleculeNet/反应/检索 | 扩展通用性，但不直接回答首要机制问题 | 暂缓 |

最终论文至少覆盖“3D 数值—3D 到文本—文本到分子”三个任务族；不要求每一个预训练消融都在三类任务上重复。

## 5. 消融如何压缩到资源可承受范围

3D-MolT5 的预训练本身使用 8×80GB A100、400k steps、global batch 768；其核心消融集中在 PubChemQC，而不是把每个变体铺满所有下游任务。论文比较了：在预训练和微调中移除全部 3D 信息、移除 1D+3D joint denoising，以及移除整个 translation-task 类别。Appendix C.2 还在一个 PubChem molecule-to-text 场景比较 embedding summation 与 sequential concatenation，并报告后者达到收敛所需训练时间超过 1.5 倍。这支持“代表性任务上的定向消融”，不支持穷举组合。

3D-MolT5 的 no-3D 不等于本项目的 CE-only。为避免把 3D 输入、geometry loss 和 anchor task 混在一次比较中，冻结以下名称：

| 配置 | motif-3D 输入 | geometry loss | anchor candidate |
|---|---:|---:|---:|
| B0 `no-3D` | 否 | 否 | 否 |
| B1 `3D-input CE` | 是 | 否，`lambda_geom=0` | 否 |
| B2 `CE+geometry` | 是 | 是 | 否 |
| B3 `CE+geometry+anchor` | 是 | 是 | 是 |

- B0 vs B1 隔离 motif-3D 输入/融合的贡献；
- B1 vs B2 隔离 geometry loss 的贡献；B1 必须保留与 B2 完全相同的 3D 输入、架构、样本、mask/sampling 和所有 CE 项，仅令 geometry-loss 梯度为零；
- B2 vs B3 隔离 anchor candidate 的贡献。

若正式预算只允许训练两个配置，就只能保留相应的一个核心因果主张，不能用“完整方案 vs CE-only”同时证明三个模块。

### 5.1 三层证据设计

| 层级 | 资源 | 必做内容 | 放行条件 |
|---|---|---|---|
| C0：CPU 证明 | 当前无卡机器 | overlap、碰撞、可逆 round-trip、token 长度/截断、OOV、anchor pairing、atom permutation/stereo 审计 | 所有硬不变量 PASS |
| S0：短程筛选 | 单张 4090，单 pretraining seed | 缩小预算筛 B0/B1/B2；B3 只在表示修复后加入；只看一个 3D 代表任务 | 只淘汰发散、无收益或明显负迁移方案，不产生有效性结论 |
| F0：论文主对照 | 单张 4090 的可承受正式预算 | 只选择一个核心因果相邻对照；下游微调至少 3 seeds | 主结论方向稳定；报告均值/方差与限制 |

主效能比较预注册并固定 membership、数据顺序、非 padding token 预算、optimizer updates、batch 规则、seed 与 schedule；由于表示长度和额外 loss 会改变每 token 计算量，FLOPs、tokens/s、墙钟和峰值显存作为独立效率指标实测报告。若补 compute-matched 对照，将其作为第二视图，不能替代等数据/等训练协议结果。

若要把“锚点因子化”写成独立论文贡献，增加一项同预算的 `exact anchored motif token` vs `slot-factorized motif` 对照；否则只把它作为可逆、节省词表的实现设计，在 C0 和 S0 证明，不扩大主张。

### 5.2 明确取消/后置的实验

- 不做所有 loss × mask × 数据集 × 权重的笛卡尔积；
- 不为每个消融重跑三次全量预训练；
- 不把所有消融重复到所有下游任务；
- 不默认训练“泄漏版 vs 清洁版”两套完整模型，overlap table 本身是零训练证据；只有重合异常大且影响判断时才做小规模敏感性实验；
- 不先做反应、检索和全部分类任务；
- 不进行大范围 MSE 权重网格，先用梯度范数、loss scale 和 2–3 个短程候选校准。

下游 3 seeds 只估计微调方差，不能替代预训练 seed 方差。投稿版本对最终 Full 和最关键主消融应尽量完成至少 2 个独立 pretraining seeds；若资源确实不允许，必须明确报告单 pretraining seed 限制，不得声称统计显著或充分稳定。

### 5.3 顶刊叙事的资源约束原则

每增加一个“核心创新”主张，就必须增加能隔离它的因果对照。资源有限时，应缩小主张而不是保留大量证据不足的模块：

- 核心主张优先保留 `motif-level 3D fusion / geometry objective`；
- 锚点辅助任务先作为候选；只有通过短程筛选才升级为主线；
- 表示可逆性、tokenizer 确定性和防泄漏是方法有效性的前提，不需要 GPU 消融来证明。

## 6. 官方 CAMT5 与当前 motif/anchor 实现的差异

官方 CAMT5 已在远端隔离拉取：

- 目录：`/root/autodl-tmp/reference-repos/CAMT5-official-20260805`
- commit：`5875a0a6d73756b1204a7600af9c0b773d9e1ae3`
- 未覆盖或修改当前 MoSt-T5 主干。

当前对比基于 [CAMT5 论文](https://aclanthology.org/2025.findings-emnlp.1221/)及上述 commit：

| 维度 | 官方 CAMT5 | 当前 R1/MoSt-T5 候选 |
|---|---|---|
| motif 基础划分 | 环与非单键相连原子形成 motif，未覆盖原子为 singleton | 保留相近化学启发式，但重写为 molecule-native、确定性 union-find/DFS |
| anchor 语义 | `<n*>` 表示 fragment 内局部父/子断键次序 | 为整分子 cross-motif edge 分配全局 ID，同一 ID 出现在两个端点 |
| tokenizer 单元 | 含 anchor 的完整 fragment 作为一个 token 加入 tokenizer | 当前把 anchor 抽成 special token，并把去 anchor motif 加入词表 |
| 立体化学 | fragment token 中保留 stereo | 当前 P1 producer 暂未保留；P2 legacy 中存在 `@`、`/`、`\\`，二者尚不兼容 |
| 掩码 | motif-level MLM，并按 motif 原子数赋权 | 现有旧主干 collator 的普通 identity mask 不掩 anchor；anchor 的 mask weight 为 0，尚非冻结的 R1 策略 |

[代码] 当前相关位置：

- `most_t5_next/r1/adapter/mol_linearizer.py`：全局 cross-edge ID、原子到 motif 映射和序列化；
- `tokenization/motif_tokenizer.py`：抽取全部 anchor，删除其原位置，再生成 anchor special tokens + pure motif；
- `process/build_vocab_pipeline.py`：去 anchor 后统计 motif 词表；
- `dataset/dataset2.py`：anchor mask weight 为 0，普通 MLM 不会掩掉连接骨架。

因此，“2D identity mask”当前更准确的表述是：**隐藏 motif 身份，但保留 motif 间连接 scaffold**。这不一定错误，类似图模型中的 node-attribute masking 可以保留 edges；但论文中不能称为“隐藏全部 2D 拓扑”。

## 7. 当前锚点删除投影的硬阻塞问题

### 7.1 全量实测

审计对象：

`/root/autodl-fs/most-t5-r1/runs/cpu-p0-20260805/incoming/pcqm-geometry-production-v2-20260805T0430Z/motif_census.jsonl`

- bytes：65,655,668
- SHA-256：`165933bf8b148b4163cf3b2ba5d78a173f4522dd98a5bc4967757d183fe8750b`
- unique exact motif lexeme：441,769
- total motif occurrence：24,180,228
- weighted anchor occurrence：41,628,180

当前投影键定义为：

`(按出现顺序提取的 anchor ID 列表, 删除所有 anchor 后的 core motif)`

结果：

| 指标 | 数值 |
|---|---:|
| projected unique keys | 441,163 |
| ambiguous keys | 604 |
| involved exact lexemes | 1,210（0.273899%） |
| involved motif occurrences | 6,520（0.026964%） |
| 单个键最多对应 exact lexemes | 3 |

碰撞示例：

```text
<1*>C1CCC2=C(C=CC=C2<2*>)C1
<1*>C1CCC2=C(C=CC=C2)C1<2*>
```

两者拥有同样的 ID 顺序 `[1, 2]`，删除 anchor 后 core 也相同，但 `<2*>` 所在原子位置不同。在现有“anchor 列表 + pure motif”的拟议序列化方式下，两者会映成同一词元级 token 序列，因而无法从该表示精确恢复原始连接位置。604 个冲突键证明的是词元级信息损失；完整分子级是否出现最终重建冲突仍需 round-trip 实测，不能由该数字单独外推。

碰撞比例虽小，但它否定的是“表示可逆/无损”这一结构性质，不能因频率低而忽略；并且稀有连接恰可能对应困难化学结构。

### 7.2 接受的修复：attachment-slot template

不要删除 anchor；将每个内联 `<n*>` **在原位置**替换为统一槽位 `<*>`，并保存按从左到右顺序排列的 ID：

```text
exact:    <1*>C1CCC2=C(C=CC=C2<2*>)C1
template: <*>C1CCC2=C(C=CC=C2<*>)C1
ids:      [1, 2]
```

逆变换只需按顺序把第 i 个 `<*>` 换回第 i 个 `<n*>`。全量实测结果：

| 表示 | unique motif units | 相对 exact 缩减 | 投影碰撞 |
|---|---:|---:|---:|
| exact anchored lexeme | 441,769 | 0 | 0 |
| 当前删除式 pure motif | 214,554 | 51.4330% | 有 |
| slot template | 229,600 | 48.0271% | **0** |

slot template 只比当前 pure motif 少约 3.4 个百分点的词表压缩，却恢复了词元级精确连接位置，应取代“pure motif”作为正式合同。建议正式命名为 `attachment_slot_template`，避免误导为完全不含连接信息。`<*>` 只是模板内部的原子化占位符：模板必须作为完整 motif unit 直接查表，不能交给 SentencePiece 再拆分，也不能把含 `<*>` 的模板重新交给 RDKit 当 SMILES 解析。

### 7.3 tokenizer freeze 前的硬门禁

1. 版本化 exact ↔ `(slot template, ordered IDs)` 合同；
2. 对 441,769 个 lexeme 做零碰撞证明；
3. 做 byte-exact inverse round-trip；
4. 做 record/molecule 级序列化—反序列化与键一致性测试；
5. 每个 template 的 slot 数必须等于 anchor token 数；
6. 每个全局 anchor ID 在完整未截断分子中恰出现两次；
7. 截断必须以完整 motif/edge group 为单位，禁止产生 orphan anchor；
8. anchor ID 超出预留范围时 fail closed。P1 当前观测最大 ID 为 15、历史预留为 100，但仍需审计所有下游；
9. 固定 stereo policy，并重跑 P1/P2 compatibility；
10. 对 RDKit atom renumbering 做不变性或鲁棒性测试。当前对 3 个分子的反向 `RenumberAtoms` 测试全部改变序列，说明表示仍依赖输入 atom index。

### 7.4 词表缩小不等于计算更省

若每个 anchor 都成为一个独立序列 token，P1 census 的名义 motif 段长度倍率为：

`(24,180,228 motif + 41,628,180 anchor) / 24,180,228 motif = 2.7216×`

这会增加截断风险和 self-attention 成本。修复后必须实测：平均/P95/P99/max token 长度、超过 max length 的分子比例、有效 batch tokens/s、峰值显存，以及 group-aware truncation 后的保留率。仅用“词表从 441k 缩小到 230k”不能证明方案更高效。

## 8. 是否针对锚点额外设计训练任务

### 8.1 裁定

**有研究价值，但暂不批准为主线任务。** 先修复表示并通过第 7.3 节全部门禁，再用一次低成本、固定破坏 token 预算的短程实验决定去留。

不建议最初设想中“同时掩掉一条边的两个同名 anchor，再预测原 ID”的朴素版本：若 ID 按稠密顺序编号，模型可能通过“哪个编号缺失”或固定遍历顺序恢复标签，并未真正学习连接关系；若每次随机重命名且两个端点都被隐藏，目标编号又可能不可辨识。

### 8.2 推荐 V1：锚点配对恢复（Masked Anchor Pair Assignment）

继续使用现有 T5 span-denoising CE，不新增 MSE、分类头或 matching head：

1. 只对至少含两条 cross-motif edges 的分子启用；
2. 每个样本同时选择 `K >= 2` 条 edge，并对每条 edge 随机掩掉一个端点、保留另一个端点；
3. 选边后，对该分子的活跃 anchor IDs 做随机双射重命名，但同一 edge 的两个端点仍同名；
4. 用“每批被破坏的 anchor token 数”固定任务预算，不用含混的 batch/edge 百分比；
5. 保留 attachment-slot template、motif 身份与 3D 状态；
6. decoder 在标准 T5 target 中把多个可见 partner labels 正确分配回多个 masked slots；
7. 截断不得破坏任何参与任务的可见—被掩端点对。

若每次只掩一条 edge 的一个端点，正确标签就是序列中唯一只出现一次的 anchor ID，模型用计数和复制即可完成；随机重命名也消除不了这个频次捷径。`K >= 2` 后，多个单次出现的候选标签必须被分配给多个 slot，目标仍可识别，但不再能仅靠“找唯一缺失编号”完成。随机双射进一步削弱“全局第几个 edge 就是 `<n*>`”的编号/遍历捷径。

该任务恢复的是 slot 与 partner label 的配对关系，不是 anchor ID 的化学“身份”。它与已有两种视图形成较清楚的互补关系：

| 任务 | 隐藏 | 保留 |
|---|---|---|
| motif identity mask | motif 身份 | geometry + connectivity |
| 3D state mask | geometry | motif identity + connectivity |
| masked anchor pair assignment | 多个 slot 的 edge pairing labels | motif identity + geometry + visible partner labels |

### 8.3 必须报告的指标

anchor token accuracy 不足以证明学到了分子连接，还应报告：

- anchor pair consistency；
- 对 anchor ID 全局重命名取商后的 edge precision/recall/F1；
- exact motif 与 molecule reconstruction；
- decoded chemical validity；
- orphan-anchor 与 truncation failure rate；
- 随机 ID 重命名、atom permutation 下的鲁棒性；
- 对 CE 语言建模和 geometry objective 的负迁移；
- QM9/代表性 3D 下游收益。

还必须加入低成本捷径/反事实基线：

- singleton/counting baseline；
- 只看 anchor 序列与遍历顺序、看不到 motif chemistry/3D 的 baseline；
- motif slot templates 打乱；
- E3FP 置零或在分子间打乱。

若 template/E3FP 被打乱后性能几乎不变，该任务只能被称为序列化语法正则项，不能声称学习了化学连接或 3D 几何。

唯一新增训练对照为：

`B2 CE+geometry` vs `B3 CE+geometry+Masked Anchor Pair Assignment`

两者使用相同数据 membership、优化器、数据顺序、有效训练 token/updates 与下游协议，并单独报告增加的 FLOPs/墙钟。若只提高锚点 token 准确率而不提高配对、重构、几何或下游，任务应删除或降格为语法正则化。

未来若资源允许，可研究 permutation-invariant edge matching head；当前不应同时增加新 head、新 loss 和新表示，否则因果归因和调参成本都会失控。

## 9. 创新性可以怎样表述

### 9.1 当前不能主张

- “首次在分子中使用 motif anchor”；
- “首次把 attachment point 编码到 motif”；
- “当前表示无损/可逆”；
- “identity mask 隐藏了全部 2D 拓扑”；
- “额外 anchor loss 已能增强 3D 表征”。

[CAMT5](https://aclanthology.org/2025.findings-emnlp.1221/)、[Hierarchical Molecular Graph Generation](https://proceedings.mlr.press/v119/jin20a.html)、[MiCaM](https://openreview.net/pdf?id=Q_Jexl8-qDi)与 [MGSSL](https://papers.nips.cc/paper/2021/hash/85267d349a5e647ff0a9edcb5ffd1e02-Abstract.html)均覆盖了 motif、attachment/连接关系或 motif-level self-supervision 的相邻思想。

### 9.2 通过修复与实验后可考虑的贡献表述

> We introduce a position-preserving, invertible factorization of motif chemistry, attachment slots, and inter-motif edge pairing, and couple it with motif-level 3D state fusion and complementary identity, geometry, and pairing-denoising objectives in a unified text-to-text model.

这句话包含四个都必须被证明的限定词：

- `position-preserving`：slot 原位保留；
- `invertible`：全量零碰撞与 byte-exact round-trip；
- `factorization`：相对 exact token 的词表/泛化收益及长度代价；
- `complementary`：`K >= 2` 配对任务通过计数/遍历/E3FP 反事实控制，并在严格对照中产生非冗余收益。

在更系统的 novelty search 和同类方法对比完成前，应称其为 **candidate methodological contribution**，而不是已经确定的首创。

## 10. 从当前无卡状态到下一阶段的执行顺序

### R1.1：当前 CPU 阶段，立即执行

1. 将删除式 pure-motif projection 改为旁路候选 `attachment_slot_template_v1`，不覆盖原代码；
2. 全量运行 collision、inverse、pairing、overflow、truncation 与 atom-permutation tests；
3. 建立 QM9、PubChem 3D caption、ChEBI-20、PubChemQC 的 registry/identity/split manifests；
4. 生成 P1/P2 vs downstream valid+test overlap tables 和 exclusion manifests；
5. 基于 exclusion 重新派生“允许进入词表发现”的 P1/P2 census；已有逐 record payload 时只过滤聚合，不重复几何重算；
6. 比较 exact anchored、slot-factorized 两种表示的 vocab size、token length、OOV 与 truncation；
7. 检查 CAMT5 仓库许可证/衍生代码边界；当前官方根目录未见明确 LICENSE，发布前必须解决 provenance。

### R1.2：恢复 4090 后的最小 smoke

1. 只跑 Dataset → Collator → model forward/backward；
2. 检查 slot/anchor 对齐、CE/MSE mask、梯度有限性、单 batch 过拟合；
3. 测 tokens/s、峰值显存和截断率；
4. 未通过不得进入 P1。

### P1-screen：低成本选择

按固定 membership、数据顺序、非 padding token 预算和 updates 做 B0/B1/B2 单 pretraining seed 短跑；B3 只有在 slot 修复和 `K >= 2` 捷径基线通过后加入。只用一个代表性 3D 任务淘汰发散、无收益或明显负迁移的候选；单 seed 不产生有效性或显著性结论。

### P1-final / downstream

只对一个被预先选定的相邻因果对照投入正式预算：B0/B1 证明 3D 输入，B1/B2 证明 geometry loss，或 B2/B3 证明 anchor 任务；资源不允许时不能同时保留三项主张。下游微调至少使用三个 seeds；最终 Full 与最关键主消融尽量补两个独立 pretraining seeds。先完成 QM9、PubChem 3D caption、ChEBI-20；PubChemQC 只有在同源 overlap proof 与预算同时满足时升级为旗舰参照。

## 11. 最终放行条件

进入 tokenizer freeze/P1 前必须同时满足：

- downstream registry 与 protected valid/test membership 已冻结；
- P1/P2 exclusion manifest 可复算，原始数据未改；
- slot projection 在全量 lexeme 上零碰撞且 byte-exact 可逆，并通过 molecule-level round-trip；
- anchor pair、overflow、truncation、stereo、atom-order policy 明确并通过测试；
- P1/P2 的统一 producer/compatibility 有结论；
- tokenizer vocabulary 只来自准入后的训练 membership；
- 4090 单 batch 与短程性能 smoke PASS；
- 消融矩阵及每条论文 claim 的必要对照已经冻结。

在这些条件之前，当前最优决策不是启动训练，而是利用无卡机器完成不可逆训练决策之前的 CPU 证据闭环。

## 12. 主要资料

- 3D-MolT5: <https://arxiv.org/html/2406.05797>
- 3D-MoLM: <https://arxiv.org/html/2401.13923>
- PCQM4Mv2 official documentation: <https://ogb.stanford.edu/docs/lsc/pcqm4mv2/>
- CAMT5: <https://aclanthology.org/2025.findings-emnlp.1221/>
- MolE: <https://www.nature.com/articles/s41467-024-53751-y>
- Hierarchical Molecular Graph Generation: <https://proceedings.mlr.press/v119/jin20a.html>
- MiCaM: <https://openreview.net/pdf?id=Q_Jexl8-qDi>
- MGSSL: <https://papers.nips.cc/paper/2021/hash/85267d349a5e647ff0a9edcb5ffd1e02-Abstract.html>
