# 08 文献视角下的分层代码思路与证据矩阵

> 审查日期：2026-07-17  
> 审查对象：当前工作区代码与远端训练版本的既有分析结论  
> 核心原则：文献能支持一个研究命题，不等于当前代码实现、超参数或训练结果已经被证明正确。

## 1. 结论先行

MoSt-T5 的**总体研究方向可行，模块组合具有明确的文献基础**：已有工作分别证明了文本到文本的分子—语言统一建模、motif/fragment 级分子表示、原子中心三维子结构 token、二维—三维联合预训练以及分子—文本细粒度对齐的合理性。项目把这些方向组合为“motif 拓扑序列 + 多层 E3FP + 局部 atom-to-motif 融合 + T5 多任务训练”，属于有依据的组合创新。

但文献目前**不能证明当前实现已经有效**。以下内容仍是项目自己的工程或研究假设：

- 每个 E3FP shell 独立嵌入后直接求和；
- 仅允许 motif 查询其映射原子的局部注意力；
- 当前 sigmoid 门控的具体形式；
- 非折叠逐 token sentinel mask；
- motif 大小的 `log1p` mask 权重和文本 IDF mask；
- 15%/5% shell dropout；
- 同网络未遮蔽分支 `detach` 产生的移动几何目标；
- `lambda_3d=500`、四任务各 25%、20K/25K 词表等精确数值。

此外，`MotifTokenizer` 的 `set -> list` 会破坏 token ID 的可复现性。这不是“缺少文献”问题，而是会使同一 motif 在不同进程中对应不同 embedding 行的实现错误；在修复并重新核验 checkpoint 之前，历史结果不足以验证理论假设。

## 2. 证据等级

| 等级 | 含义 | 可用于何种表述 |
|---|---|---|
| A 直接支撑 | 同类分子任务中使用了基本相同的表示、目标或评测方法，并报告实验结果 | “已有分子领域工作支持该设计方向” |
| B 间接支撑 | 相邻分子方法或通用机器学习理论支持机制，但与当前实现不完全相同 | “机制合理，但需要本项目实验确认” |
| C 工程/研究假设 | 有直觉或实现需要，尚未找到直接证据，或精确公式/数值是自定的 | “待消融验证” |
| D 不成立或高风险 | 违反可复现性、数据一致性或方法成立前提 | “应先修复，不能作为创新点辩护” |

## 3. 总体层：研究命题是否成立

### L0-1 统一分子与自然语言的 text-to-text 模型

**代码命题**：同一 T5 编解码器处理 motif 去噪、分子描述、文本生成分子和一般文本去噪。

**文献证据（A）**：

- MolT5 把分子字符串和自然语言共同放入 T5，并以 span corruption 预训练，再执行 molecule captioning 与 text-based molecule generation（本地 `MoIT5.pdf`，PDF 第 4–5 页）。
- BioT5/BioT5+ 把 SELFIES、自然语言和生物序列置于统一 T5 框架，并进一步采用多任务 instruction tuning（本地 `biot5.pdf` 第 1–2 页；`biot5+.pdf` 第 1–4 页）。
- T5 原始工作建立了“所有任务统一为 text-to-text”的基础方法，并使用 C4 与 span corruption（补充 `T5_2020_JMLR.pdf`，特别是第 24 页）。

**判断**：总体任务框架可行；但四任务比例与每个任务是否都需要共享全部参数仍需消融。

### L0-2 以 motif/fragment 代替纯字符分子序列

**代码命题**：把分子拆成 motif，并用 anchor token 表达 fragment 之间的拓扑连接。

**文献证据（A）**：

- T-SMILES 明确使用 fragment-based、multi-scale molecular representation，并用树结构和共享/虚拟原子保留片段拓扑（本地 `t-smiles.pdf` 第 1–4 页）。
- CAMT5 使用 motif-wise context-aware tokenization，并以 motif 重要性设计预训练目标（本地 `CAMT5.pdf` 第 1、2、4、8 页）。
- FineMolTex 同时学习 molecule-level 与 motif-level 的分子—文本知识（本地 `finemoltex.pdf` 第 1–3 页）。

**判断**：motif 与连接拓扑的主方向有直接依据；当前 CAMT5 `Frag.encode` 的具体线性化和词表覆盖率仍要用重建率、未知率及立体化学保持率验证。

### L0-3 用三维信息增强二维/一维分子表示

**代码命题**：为每个原子生成多层 E3FP，把三维局部环境对齐到 motif token。

**文献证据（A/B）**：

- E3FP 原始工作把 ECFP 思路扩展成快速、旋转/对齐不变的三维构象指纹，编码原子周围逐层扩大的三维 shell，并在部分靶点上获得更好的 precision-recall（Axen et al., 2017，DOI `10.1021/acs.jmedchem.7b00696`；PMC 正文）。
- 3D-MolT5 直接使用 E3FP 构造原子中心三维 token，并把 1D SELFIES token 与 3D token 在原子层面对齐、相加（本地 `3dMolt5.pdf` 第 2–3、5–6 页）。
- GraphMVP 通过 2D 拓扑与 3D 几何视图的一致性/重建学习增强分子表示，并把三维几何视为训练期 privileged information（补充 `GraphMVP_2022_ICLR.pdf` 第 1–5 页）。
- 3D-MoLM 使用三维编码器、molecule-text projector 和分阶段对齐/指令微调（本地 `3D-MoLM.pdf` 第 1–4 页）。

**判断**：引入 3D 和做局部对齐是强支撑方向；当前“E3FP 四层如何聚合到 motif”是项目新设计，不能由 3D-MolT5 的简单相加自动证明。

## 4. 局部层与细节层：代码—真实操作—证据矩阵

### 4.1 数据与分子预处理

| 粒度 | 当前真实操作与代码位置 | 证据 | 等级 | 严谨判断与必要验证 |
|---|---|---|---|---|
| L1 数据规模 | PubChem/PubChemQC 等大规模无标签分子用于预训练 | MolT5、BioT5、3D-MolT5 都使用大规模分子语料 | A | 数据来源合理；仍需记录去重、过滤、泄漏和版本 |
| L2 文本混合 | Phase 2 加入 C4 去噪 | T5 与 MolT5 使用 C4；3D-MolT5也保留文本去噪 | A | 可支持保持自然语言建模能力；“一定防止遗忘”仍是推断，应测通用文本损失/化学描述指标 |
| L2 构象生成 | RDKit 生成一个廉价构象，再做力场优化 | E3FP 需要构象；Uni-Mol+ 明确讨论廉价 RDKit 构象与高质量构象之间的差异 | B | 可作为可扩展近似，但单构象不等于生物活性构象；应比较多构象、不同 seed 和失败率 |
| L2 E3FP | 为每个原子计算逐 shell 的 3D identifier | E3FP 原始论文；3D-MolT5 | A | 方法本身成立；需固定构象参数、E3FP 版本、半径和 level |
| L3 4 层 shell | 输出 `[num_atoms, 4]` | E3FP 支持迭代 shell；3D-MolT5展示多层 token | B | “四层最优”没有直接证明，需对 1/2/3/4/更多层消融 |
| L3 折叠到 4096 | identifier 映射到 4096 大小词表 | E3FP 支持固定长度 folding | B/C | folding 合理，4096 是超参数；需报告碰撞率和 2048/4096/8192 对照 |
| L2 motif 线性化 | CAMT5 Frag 拆分，anchor 表示连接 | T-SMILES、CAMT5 | A | 应报告可逆重建率、未知 motif 率、token 长度和立体化学丢失 |
| L2 atom mapping | 保存 atom-to-motif 映射并在加 anchor 后重定位 | 3D-MolT5 的原子层面对齐；FineMolTex 的细粒度对齐 | B | 映射思想合理；具体索引变换是工程正确性问题，需逐样本断言和 round-trip 测试 |

### 4.2 Tokenizer 与词表

| 粒度 | 当前真实操作与代码位置 | 证据 | 等级 | 严谨判断与必要验证 |
|---|---|---|---|---|
| L1 共享词表 | 文本、任务前缀、motif 共用 T5 embedding | MolT5、BioT5、3D-MolT5 的统一序列建模 | A/B | 共享空间合理；新增 token 必须稳定、保存且可恢复 |
| L2 任务前缀 | `[MMM]:`、`[Caption]:`、`[Text2Mol]:`、`[Denoise]:` | T5 task prefix；BioT5+ instruction tuning | A/B | 路由方式合理；需检查特殊 token 是否被拆分 |
| L2 数字 token | 把 0–9 作为特殊 token | BioT5+ 指出普通 SentencePiece 对数字切分不稳定并引入数值 tokenization | B | 支持“需要专门处理数字”，不直接证明单个数字 token 足够；需测数值格式有效率和误差 |
| L2 20K→25K | Phase 2 追加约 5K motif | CAMT5/T-SMILES 支持 motif vocabulary | C | 精确容量无文献支撑；应以 coverage、长度、频率和下游收益选择 |
| L3 相似 embedding 初始化 | 用 Morgan/Tanimoto 找旧 motif，为新 motif 复制相似 embedding | ECFP/Morgan 支持结构相似度表征 | C | “结构相似可用于检索”不等于“embedding 可以直接复制”；需随机初始化、均值初始化、相似初始化对照 |
| L3 `set -> list` 注册 | `tokenization/motif_tokenizer.py:79-93` 先存入 set，再 `add_tokens(list(set))` | 无；违反稳定词表 ID 前提 | D | 必须保留文件顺序/显式排序，并随 checkpoint 保存 tokenizer；修复前不得把跨阶段 embedding 继承视为有效 |

### 4.3 Mask 与数据增强

| 粒度 | 当前真实操作与代码位置 | 证据 | 等级 | 严谨判断与必要验证 |
|---|---|---|---|---|
| L1 T5 去噪 | 15% token corruption + sentinel target | T5 第 24 页；MolT5/BioT5 | A | 基线合理 |
| L2 非折叠 mask | 每个被 mask token 留一个 sentinel，以保持 motif 位置不变，`dataset.py:311-377` | T5 使用 span sentinel，但会折叠 span | C | 是满足 mapping 的工程折衷，已偏离 T5 分布；必须与标准 span corruption 比较 |
| L2 motif 重要性 | 按 fragment size 后 `log1p` 加权，`dataset.py:298` | CAMT5 的 importance-based motif masking；FineMolTex 的重要 motif 预测 | B/C | “重要性 masking”有依据；大小与 `log1p` 公式没有直接依据，应与均匀、频率、原子数、化学显著性比较 |
| L2 文本重要性 | TF-IDF/IDF 决定文本 mask 权重，`process/text_weights.py:59-87` | FineMolTex 对重要词做细粒度 masking；TF-IDF 是成熟信息检索方法 | B/C | 可解释但不是分子文本任务的直接最优证据；应测均匀 mask 与术语/实体 mask |
| L2 联合遮蔽 | motif 被遮蔽时，其映射原子的 E3FP 同时置空 | 3D-MolT5 的 1D+3D joint denoising；GraphMVP 的对应区域 mask | A/B | 跨模态联合恢复合理；需与仅 1D、仅 3D、独立 mask 对照 |
| L2 仅几何遮蔽 | motif 保留、E3FP 置空，要求从语义/上下文恢复几何 | GraphMVP 的跨视图重建；3D-MolT5 的 3D-to-1D/联合任务提供相邻依据 | B | 方向合理，当前比例需消融 |
| L3 shell dropout | 15% 去 level 3；5% 去 level 2–3，`dataset.py:445-448` | 通用 dropout/视图增强提供机制类比 | C | 未找到直接分子 E3FP 证据；应验证是否模拟真实缺失，而不是制造分布外噪声 |

### 4.4 表示、局部融合与 T5

| 粒度 | 当前真实操作与代码位置 | 证据 | 等级 | 严谨判断与必要验证 |
|---|---|---|---|---|
| L1 T5 backbone | 保留 T5 encoder-decoder，替换输入 embedding 流 | T5、MolT5、BioT5 | A | 适合统一生成任务 |
| L2 多层 E3FP embedding | 四个 level 各自 embedding 后求和，`model/modeling.py:37-74` | 3D-MolT5 对原子对齐的 1D/3D embedding 求和 | B/C | 文献只支持相邻做法；应比较 sum、concat+MLP、level attention、共享表 |
| L2 局部 cross-attention | motif 只查询映射到自己的原子，`model/modeling.py:109-149` | FineMolTex motif-word cross-attention；3D-MolT5 原子级对齐 | B | 局部约束有化学解释，但精确机制是新假设；需与全局 attention、mean pooling、无 attention 比较 |
| L2 门控融合 | `fused=(1-g)motif+g·3D`，`model/modeling.py:151-168` | Gated Multimodal Unit 证明可学习乘性门能调节不同模态贡献 | B/C | 通用机制支持，不是分子特定证明；应报告 gate 分布、无 3D 退化行为和 residual baseline |
| L3 空映射 gate=0 | 无原子映射的文本/anchor token 强制不注入 3D | 保持模态语义边界的工程约束 | C | 逻辑正确；用单元测试证明，无需声称是文献创新 |
| L3 E3FP padding=0 | `-1` 平移为 embedding padding 行并固定为零 | 标准 embedding/masking 工程规范 | B | 必须验证梯度后 padding 行仍为零，attention mask 与 ID 一致 |

### 4.5 几何目标与两阶段训练

| 粒度 | 当前真实操作与代码位置 | 证据 | 等级 | 严谨判断与必要验证 |
|---|---|---|---|---|
| L1 Phase 1 | 先做 motif 去噪与几何表示恢复 | 3D-MoLM 的分阶段对齐；GraphMVP 的 2D/3D SSL；3D-MolT5 的联合去噪 | A/B | 阶段目标合理 |
| L2 latent reconstruction | 从 encoder state 用 MLP 预测连续 3D latent，`modeling.py:226,312-345` | GraphMVP 明确提出在连续表示空间做重建以避免直接结构重建困难 | A/B | 表示空间重建有依据；当前 target 定义不同，需额外论证 |
| L2 stop-gradient target | 未遮蔽 E3FP 通过同一当前模型产生 target 后 `detach`，`modeling.py:264-287` | SimSiam 证明 stop-gradient 在其孪生视觉架构中对防塌缩关键 | B/C | 只能作为类比；没有证明当前分子/共享融合架构不会塌缩。需监测方差、cosine、rank，并比较 EMA teacher/固定 E3FP target/GraphMVP 目标 |
| L3 `lambda_3d=500` | Phase 1 总损失 `LM + 500·MSE`，`train1.py:118` | 无直接依据 | C | 必须报告两项未经加权/加权的量级、梯度范数，并扫 0/1/10/100/500 |
| L1 Phase 2 四任务 | MMM、caption、text2mol、C4 denoise | MolT5、BioT5、3D-MolT5 | A | 任务集合有直接依据 |
| L2 各 25% | 远端 Dataset 实际四任务等概率 | MolT5 有文本/分子均衡混合的相邻做法 | C | 四任务等权不等于梯度均衡；需按损失、收敛和目标场景调权，并修复失真的 `task_probs` 接口 |
| L2 Phase 2 `lambda_3d=1` | 继续保留较弱几何约束，`train2.py:242` | 持续多任务/防遗忘是一般动机 | C | 精确数值和是否应作用于所有任务需验证 |

### 4.6 下游评测

| 粒度 | 当前真实操作与代码位置 | 证据 | 等级 | 严谨判断与必要验证 |
|---|---|---|---|---|
| L1 MoleculeNet | 分类/回归数据集与任务相应指标 | MoleculeNet 建立数据集、切分、指标和基线体系 | A | 应严格遵循每个数据集的官方指标，并报告多 seed |
| L2 scaffold split | Bemis–Murcko scaffold 划分，`moleculenet/splitters.py:30-98` | MoleculeNet 强调结构泛化与切分规范 | A | 必须检查 train/valid/test scaffold 交集为零，并报告样本量 |
| L2 LoRA | encoder 上低秩适配 | LoRA 冻结预训练权重并注入低秩可训练矩阵，显著减少参数 | A（通用） | 能支持参数高效适配；不能替代全量微调和 frozen-head baseline |
| L2 mean pooling head | encoder masked mean pooling + MLP | 通用序列分类做法 | B | 与首 token、attention pooling、motif-only pooling 比较 |
| L2 生成式数值预测 | 生成字符串后解析 float | BioT5+ 特别处理数值 tokenization | B/C | 必须把格式失败计入结果，报告 valid-format ratio；与回归 head 并列比较 |
| L2 生成指标 | validity、uniqueness、Tanimoto 等 | MolT5、T-SMILES、CAMT5 使用相邻评测 | A | 同时需要 novelty、property relevance、scaffold/fragment 分布和多样性 |
| L3 beam=5 | 生成时固定 beam 数 | 无普遍最优证据 | C | 作为评测协议需固定并报告；至少与 greedy、采样和不同 beam 对照 |

## 5. 从文献角度得到的可行性边界

### 已有文献足以支持的部分

1. 采用 T5 统一分子—文本任务。
2. 用 motif/fragment 和连接拓扑表达分子。
3. 使用原子中心、多 shell、对齐不变的 E3FP 作为局部三维描述。
4. 在原子/motif 层面进行细粒度跨模态对齐。
5. 使用二维/一维与三维联合遮蔽、对齐或表示重建作为预训练信号。
6. 用 MoleculeNet scaffold split、多 seed 和任务相应指标评价泛化。
7. 使用 LoRA 作为参数高效下游适配方案之一。

### 文献只能提供间接支持的部分

1. 受限局部 cross-attention 比全局融合更好。
2. 门控能在 motif 与 3D 之间学到正确权重。
3. 移动 latent target 能稳定学习而不塌缩。
4. 单个廉价 RDKit 构象足以代表与任务相关的三维信息。
5. 数字特殊 token 足以解决生成式性质预测的数值问题。

### 只能由本项目实验回答的部分

1. E3FP 需要几层、多少 bit、几个构象。
2. motif vocabulary 应为 20K、25K 还是按 coverage 自动选择。
3. mask 权重公式、mask 比例和 shell dropout 比例。
4. `lambda_3d` 与四任务采样比例。
5. sum、gate、cross-attention 各自带来的增益及交互作用。
6. Phase 1 是否真实改善 Phase 2 和下游，而非仅增加训练成本。

## 6. 最小充分证据实验

按因果链而不是一次训练全部组合：

1. **可复现性门槛**：修复词表顺序；保存 tokenizer；两个独立进程逐 token ID、编码结果和 checkpoint reload 完全一致。
2. **数据门槛**：报告 motif 重建率/UNK、E3FP 成功率、atom mapping 越界率、构象失败率、重复和数据泄漏。
3. **表示基线**：T5-only、motif-only、motif+E3FP mean、motif+E3FP local attention、local attention+gate。
4. **几何目标消融**：`lambda_3d=0/1/10/100/500`；当前 detach target 对比 EMA teacher、固定 E3FP embedding、GraphMVP-style reconstruction。
5. **遮蔽消融**：标准 T5 span、非折叠均匀、importance mask、联合 1D/3D mask、去掉 shell dropout。
6. **任务消融**：Phase 1→Phase 2 对比仅 Phase 2；等比例对比动态/目标导向采样。
7. **下游与生成**：至少 3 个 seed；scaffold split；均值与标准差；生成任务同时报告有效性、相似度、多样性和格式失败。

只有在步骤 1–2 通过后，步骤 3–7 的比较才具有解释力。

## 7. 当前优先级

| 优先级 | 项目 | 原因 |
|---|---|---|
| P0 | 固定 motif token ID 并审计历史 checkpoint | 否则表示语义可能跨进程/阶段错位 |
| P0 | 建立数据与 mapping 完整性报告 | 局部融合依赖 atom-to-motif 映射绝对正确 |
| P1 | 几何目标与 `lambda_3d` 消融 | 当前最关键且文献支撑最弱的训练环节 |
| P1 | sum/local attention/gate 逐项消融 | 识别真正创新贡献 |
| P1 | 标准化下游 split、seed、metric | 防止偶然结果被误判为可行性 |
| P2 | mask、shell 数、bit 数、任务比例优化 | 在主链成立后做效率与性能改进 |

