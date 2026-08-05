# 13 主干审稿裁定：CE+MSE、3D target 与顶刊证据闭环

> 审核日期：2026-07-30  
> 审核范围：当前工作区的 MoSt-T5 实现、已入库论文和本轮核验的原始论文。  
> 证据原则：文献只能支持其明确采用的表示、目标或实验结论；不能自动证明本项目的 target 定义、超参数或性能。

## 1. 主干结论

### 1.1 结论表

| 问题 | 主干裁定 | 允许的表述 | 当前不能表述为 |
|---|---|---|---|
| T5 中 CE 与 MSE 能否并用 | **可以，且有直接分子与 Transformer 先例。** | “离散 token 学习与连续表示/几何回归可联合优化。” | “联合损失天然提升 T5。” |
| 当前 MSE 是否适合作为论文主方法 | **暂不接收为主方法；接收为必须保留的对照。** | “online detached latent matching baseline。” | “3D 几何重建”或“稳定的 3D teacher loss”。 |
| 如何使 MSE 更融洽 | **采用职责分离、稳定 target、归一化、渐进权重与诊断的版本。** | “masked 3D-aware latent prediction。” | “在 CE 后任意加大一个 MSE 即可融合。” |
| E3FP 是否适合 | **适合做构象条件下、局部、3D-aware 描述符；不等于纯物理几何。** | “conformer-conditioned E3FP local structural descriptor。” | “真实完整 3D 状态”或“纯几何通道”。 |
| 当前 motif 划分是否合理 | **可作为 CAMT5 风格二维 baseline；尚未证明是 3D 聚合的最优划分。** | “ring/non-single-bond structural motifs。” | “化学功能团最优划分”或“3D motif 的唯一合理定义”。 |

**总裁定：有条件接收研究主线，不接收现有实现直接进入大规模定稿训练。**  
主线已经足以构成一个可检验的研究假设：用 motif 序列承载二维化学身份/拓扑，用构象条件下的 E3FP 提供局部三维上下文，再以受控辅助目标学习二者的互补表示。最薄弱的环节不是 CE+MSE 这一数学组合，而是 current target 的语义、单构象来源、mask 定义和可复现性。它们通过 P0 门槛前，任何大规模结果都不能作为方法有效性的证据。

## 2. CE+MSE 的原始先例：能支持什么，不能支持什么

| 工作 | 原始实现 | 对本项目的有效迁移 | 不可迁移的部分 | 证据等级 |
|---|---|---|---|---|
| 3DMolFormer, ICLR 2025 | 在双通道自回归 3D 序列上使用 token CE + 系数乘以坐标 MSE；见本地 paper/3DMolFormer_ICLR2025.pdf，第 6 页，式(2)。 | 直接证明分子 Transformer 可以共同优化离散 token 与连续 3D 数值；系数应显式平衡。 | 它回归可观测坐标，不是 learned E3FP latent；主干是 GPT，不是 T5。 | A：联合损失；B：target 设计 |
| DynaBERT, NeurIPS 2020 | 固定 teacher 的 soft CE 与 embedding/hidden-state MSE 联合；本地 paper/supplementary/DynaBERT_2020_NeurIPS.pdf，第 4 页，式(3)。作者按量级选择 (1, 0.1)。 | 直接证明 Transformer 中 CE 与连续 hidden MSE 可并列，并且权重必须按可比尺度设定。 | 模型压缩、固定 teacher、encoder-only；不是分子生成。 | A：CE+MSE 机制 |
| Patient Knowledge Distillation, EMNLP 2019 | 任务 hard CE、soft-label CE 与归一化 hidden-state MSE 同时优化；论文第 3 页，式(7)–(8)。 | 支持先归一化隐藏表示再做 MSE，并将其作为主任务监督的辅助项。 | 分类蒸馏，teacher 是独立预训练模型；不是 T5。 | A：hard CE+normalized MSE |
| Uni-Mol, ICLR 2023 | masked atom prediction 与连续 3D position recovery 联合。 | 直接支持离散原子身份恢复与连续三维恢复共同预训练。 | 连续项是坐标/距离恢复，损失细节不是本项目的 E3FP latent MSE。 | B |
| C-FREE | 分子 2D/3D 子图 context encoder 经 predictor 预测 target encoder latent，使用 MSE；target encoder 用 EMA 更新；本地 paper/C-FREE.pdf，第 3–4 页。 | 分子领域最接近“masked context → stable molecular latent”设计，支持 EMA target/predictor。 | 不与生成 CE 同时训练，且 target 融合方式不同。 | A：分子 latent target；C：联合 CE |
| data2vec, ICML 2022 | masked student 回归 full-input EMA teacher 的连续上下文 target；target 归一化以避免 collapse，使用 Smooth L1；本地 paper/supplementary/data2vec_2022_ICML.pdf，§3.3–3.4。 | 直接支持 full-vs-masked、EMA、target normalization、只在 masked position 回归，以及把 Huber/Smooth L1 作为 MSE 的稳健对照。 | 预训练阶段不与 token CE 联合；并非分子模型。 | A：稳定 latent 机制；C：联合 CE |
| Improving Language Model Distillation through Hidden State Matching, ICLR 2025 | 在 BART、mBART、T5/Flan-T5 上，将条件语言建模损失与表示匹配联合。 | 直接排除“T5 编解码器不能承受表示辅助目标”的架构性担忧。 | 使用 CKA hidden matching，不是 MSE；其结果也提示简单线性匹配在高压缩 encoder-decoder 中会失败。 | A：T5 多目标；C：MSE 细节 |

### 2.1 由上述先例得到的严格判断

1. **CE+MSE 合法且可发表，但并不自动安全。** 3DMolFormer 是分子领域最直接的 CE+MSE 先例；DynaBERT 与 PKD 则证明“离散监督 + hidden-state MSE”在 Transformer 中是成熟范式。
2. **MSE 的 target 质量决定成败。** C-FREE 和 data2vec 都没有把同一步 online student 的输出仅仅 detach 后当作稳定 teacher；二者都引入了 EMA/独立目标机制。
3. **权重必须以梯度和量级校准。** DynaBERT 选择权重的理由是不同项量级不同，而不是某个固定数值可跨模型迁移。因此 Phase 1 的 lambda_3d=500 目前是待检验超参数，不能写成方法原理。
4. **“MSE”本身不必被教条化。** raw MSE 应保留为对照；normalized MSE、normalized Smooth L1/Huber 必须作为受控消融。data2vec 的 Smooth L1 是针对异常值敏感性的工程论据，不是其必然优于 MSE 的普适证明。

## 3. 当前代码中的 MSE 到底在预测什么

当前 model/modeling.py:274–345 的实际目标可抽象为：

\[
z_m^{cur}=\operatorname{sg}\Big(
\operatorname{Pool}_{\theta}
\big(W_{\theta}(x_m^{in}),E_{\theta}(d^{full}),\operatorname{map}\big)
\Big),
\qquad
\hat z_m=P_{\theta}\big(\operatorname{T5Enc}_{\theta}(x^{mask},d^{mask})_m\big).
\]

其中：

- \(d^{full}\) 是未遮蔽的多层 E3FP ID；
- \(E_{\theta}\) 是正在训练的 E3FP embedding；
- \(W_{\theta}(x_m^{in})\) 是由当前 motif ID 产生的 query；
- Pool 是当前训练中的 Q/K/V local attention；
- sg 仅阻断这一次反向传播，不会冻结下一 step 的 target。

因此它的严谨名称应为：

> **same-online-network, motif-conditioned, detached 3D-aware latent matching**

而不是“坐标回归”或“直接几何重建”。target 同时含有 E3FP 的原子身份/拓扑/构象信息、训练中不断变化的 embedding、以及由真实 motif 或 sentinel 决定的 query。

### 3.1 当前 MSE 的关键语义问题

| 代码事实 | 科学后果 | 主干判断 |
|---|---|---|
| target 经同一套训练中 Q/K/V 与 E3FP embedding 生成，仅在当前 step detach | 跨 step target 会移动；detach 不是 EMA teacher。 | 作为 current baseline 可接受；作为最终稳定 target 不接收。 |
| target query 取当前 input_ids | joint-mask 位点 query 是 sentinel；geo-only 位点 query 是真实 motif。两个 mask 组拟合的 target 定义不同。 | 必须拆开记录与消融；现状不能称单一几何 loss。 |
| collator 内部生成 mlm_mask_pos 与 geo_mask_pos，却只返回后者 | MSE 无法区分“2D+3D 都缺失”和“仅 3D 缺失”。 | P1：显式返回 joint 与 geo-only mask。 |
| E3FP tokenizer 用单一 random_seed=42 的 RDKit EmbedMolecule，再最多 MMFF 100 步优化 | E3FP 是单一生成构象的条件描述符，不是分子真实构象 ensemble。 | P1：构象敏感性/ensemble 消融是顶刊必需证据。 |
| E3FP 本身编码原子不变量、连接和相对取向 | 2D 与 3D 通道不正交；邻居 shell 可能跨 motif 边界。 | 不可称“独立 2D mask 与纯 3D mask”；需量化泄漏。 |
| geometric head 后对 prediction/target 直接 raw MSE | 范数、稀有 hash 或异常 target 可能主导梯度。 | 将 normalized MSE 与 Smooth L1 纳入首轮消融。 |
| 代码注释称“移除 NaN 抑制”，但实际调用 torch.nan_to_num | 非有限值会静默变为零 target/zero prediction，掩盖数据或数值问题。 | P0：改为统计、跳过无效位置、达到阈值即中止。 |
| 每个 DDP rank 对本地有效位置 mean 后由 DDP 平均 | 各 rank 有效 mask 数不同时，rank 等权而非有效位置等权。 | P1：以全局 count 做正确归一化。 |

### 3.2 与“3D 状态”主张的边界

E3FP 原始论文把它定义为对**一个给定构象**的 alignment-invariant 3D fingerprint，并明确讨论多构象、构象生成与立体化学带来的挑战。当前实现还在 motif 生成时调用 RemoveStereochemistry（model/CAMT5/representation.py:114）。因此论文应避免两种过强说法：

1. 不要将单个 RDKit+MMFF 构象上的 E3FP 写成“真实分子 3D 状态”；
2. 不要将移除立体化学后的 motif 序列写成完整 2D identity。

可接受且准确的术语是：

> “基于确定性 RDKit 构象的、E3FP 编码的局部 3D-aware structural context。”

## 4. 推荐的优雅版本：把 MSE 变成职责清晰的辅助任务

### 4.1 推荐目标，而非直接把 lambda 调大

\[
\begin{aligned}
h_m &= f_{T5}(x_{motif}^{mask}, d^{mask})_m,\\
\hat z_m &= \operatorname{norm}\big(P(\operatorname{LN}(h_m))\big),\\
z_m^{T} &= \operatorname{norm}\Big(
\operatorname{Pool}_{geo}\big(E_{EMA}(d^{full}),\operatorname{map}\big)_m
\Big),\\
L_{geo} &= \frac{1}{|\mathcal M_{geo}|}
\sum_{m\in\mathcal M_{geo}}
\ell\big(\hat z_m,\operatorname{sg}(z_m^T)\big),\\
L &= L_{CE}+\lambda(t)L_{geo}.
\end{aligned}
\]

其中 Pool_geo 首版应采用 mean 或 level-aware mean，而非以 motif/sentinel 为 query 的 attention；\(\ell\) 首轮并列比较 normalized MSE 与 normalized Smooth L1。

### 4.2 七项不可省略的设计约束

1. **目标职责独立。** EMA teacher 只读取 full E3FP 与 atom-to-motif mapping；不能读取真实 motif token 或 sentinel query。
2. **mask 职责明确。** collator 返回 joint_mask_positions 和 geo_only_mask_positions。首版 CE 负责 joint mask，MSE 只负责 geo-only mask；joint 位点 MSE 是后续单独消融项。
3. **预测器隔离坐标系。** 使用 LayerNorm → Linear(d,2d) → GELU → Linear(2d,d_target)；不要让 T5 encoder hidden 被 raw MSE 直接硬对齐。
4. **尺度可控。** target 和 prediction 归一化；保留 raw MSE 作为对照，不能只展示选择性最优 loss。
5. **时间可控。** 先 CE warm-up，再渐进增加 lambda(t)；EMA 仅在 optimizer update 后更新，且 checkpoint 保存/恢复 EMA 与 schedule 状态。
6. **梯度可控。** 在共享 encoder/fusion 参数上记录 \(\|g_{CE}\|\)、\(\|\lambda g_{geo}\|\) 和 cosine。首轮让辅助梯度比例处于预设的小范围，例如 0.1–0.3，而不是继承 500。
7. **数值与并行可审计。** 记录有效位置数、非有限 target 比例、prediction/target 范数、方差、effective rank；DDP 以全局有效元素数归一化。

### 4.3 实施顺序

| 版本 | 改动 | 用途 | 何时停止推进 |
|---|---|---|---|
| A | motif+E3FP 输入，仅 CE | 强基线；回答 E3FP 输入本身是否有价值。 | 若 A 未超过 motif-only，不应继续堆叠 MSE。 |
| B | A + 当前 online-detach raw MSE | 忠实评估现有想法。 | 若 B 伤害 CE/下游，不以“还需调大 lambda”解释。 |
| C | B 的 mask 拆分 + predictor + normalization + warm-up | 验证职责分离是否消除冲突。 | 若 C 未改善 B，重新审查 target，而非继续加模块。 |
| D | C + EMA 3D target encoder + geometry-only pooling | 推荐论文候选。 | 若 D 未超过 A/C，MSE 不应留在主方法。 |
| E | D 的 normalized MSE 对 normalized Smooth L1；单构象对多构象 | 验证损失稳健性和 3D 主张。 | 若收益对构象/任务不稳定，缩小 3D claim。 |

## 5. 总体方案的证据分级账本

| 层级 | 当前真实操作 | 文献支撑 | 裁定与所需证据 |
|---|---|---|---|
| 总体 | T5 统一 motif 去噪、caption、text2mol、文本去噪 | T5、MolT5、BioT5/BioT5+ | A：主框架可接收；四任务比例仍是 C，需采样消融。 |
| 表示 | ring/non-single-bond motif + anchor/DFS | CAMT5、T-SMILES | A：可作为二维序列化 baseline；功能团最优与 3D 最优均为 C。 |
| 表示 | 原子多层 E3FP 与 motif 对齐 | E3FP、3D-MolT5 | A：使用 E3FP 的方向可接收；其只代表给定构象的 descriptor，不能替代坐标物理量。 |
| 构象 | 单 seed RDKit embedding + MMFF | E3FP 需要构象；Uni-Mol+ 讨论廉价构象差异 | B：可扩展 baseline；“足够代表真实构象”为未证实，须多 seed/ensemble。 |
| 聚合 | 每层 E3FP embedding 直接相加 | 3D-MolT5 的相邻 embedding 融合 | C：sum、concat+MLP、level attention 都需消融。 |
| 聚合 | atom-to-motif restricted attention | 原子/motif 细粒度对齐的相邻工作 | B：局部性有可解释性；优于 mean/global attention 仍是 C。 |
| 融合 | sigmoid gate 的凸组合 | GMU 等通用门控 | B/C：门控可用，具体公式不是已证实创新；报告 gate 分布并与 residual/no-gate 对比。 |
| mask | 非折叠、逐 token sentinel，以维持映射位置 | T5 支持 span corruption，但采用折叠 spans | C：合理工程折衷，不能声称标准 T5；需标准 span 对照。 |
| mask | motif size 的 log1p 权重、文本 IDF 权重 | CAMT5/FineMolTex 的重要性 masking | B/C：重要性理念有据；精确权重公式与比例需消融。 |
| mask | joint 2D+E3FP mask 与 E3FP-only mask | 2D/3D joint pretraining、GraphMVP/3D-MolT5 相邻证据 | B：方向可接收；当前代码合并两组且存在 shell 泄漏，不能称独立 2D/3D 操作。 |
| 增强 | E3FP shell dropout | 通用 dropout 类比 | C：需证明它模拟有意义的不确定性，而非引入分布外损伤。 |
| 目标 | same-online detached E3FP latent MSE | C-FREE/data2vec 仅提供相邻机制 | C：必须以 current baseline 身份报告，并比较 EMA/frozen target。 |
| 优化 | Phase 1 lambda_3d=500、Phase 2=1 | 无直接权威依据 | C：不能作为设计理论；由梯度与验证集选择。 |
| 工程 | set → list 注册 motif token | 无；违反 checkpoint/tokenizer 同一语义的前提 | D/P0：排序并保存 tokenizer 前，历史训练证据不可信。 |
| 评测 | MoleculeNet/scaffold split/生成指标 | MoleculeNet、分子生成基准 | A：协议方向可接收；必须报告多 seed、方差、格式失败、数据泄漏和 3D-sensitive endpoint。 |

## 6. 顶刊所需的最小证据闭环

### Gate 0：可复现性与数据完整性（必须先过）

- tokenizer 文件顺序或显式排序固定；独立进程、断点恢复后的 token ID 和编码完全一致；
- 保存 tokenizer、motif vocabulary、E3FP 参数、RDKit/E3FP 版本、构象 seed 与训练数据版本；
- 逐样本检查 atom-to-motif 覆盖率、重复映射、越界、anchor 成对、E3FP 失败率；
- stereo-preserving round-trip 单列统计，不再用先删除立体化学后的“100% 重建”替代。

### Gate 1：target 是否真的含有可用的构象信息

- 对同一分子的多个 RDKit seed/构象，量化 E3FP target 的变化；与不同分子的距离分布比较；
- 测量 E3FP shell 跨 motif 边界率，以及被 mask motif 可由邻居 shell 恢复的泄漏率；
- 记录 target/prediction 的均值、方差、协方差 effective rank、范数和不同构象的可分性；
- 将结果解释为“conformer-conditioned representation”，除非多构象实验显示与任务相关的稳定性。

### Gate 2：CE 与几何辅助目标是否兼容

- 用完全相同的初始化、数据 token 数和训练步数比较 A–E；
- 记录 validation CE/perplexity、生成有效性和任务特定结构指标，而非只看 total loss；
- 在共享参数上报告梯度范数比与 cosine；若长期负 cosine 或 geometry 梯度主导，降低 lambda/收缩 MSE 作用层；
- 非有限值不能用 nan_to_num 静默抹平，必须形成可追溯失败日志。

### Gate 3：因果机制而非堆叠收益

- motif-only CE → motif+E3FP CE：测试输入 3D 信息；
- current MSE → split-mask/normalized MSE → EMA teacher：测试每项机制；
- mean pooling → local attention → gate：测试融合机制；
- 单构象 → 多构象/seed ensemble：测试构象假设；
- 当前 CAMT5 motif → stereo-preserving CAMT5 → 至少一种功能团/BRICS 对照：测试划分假设。

### Gate 4：论文主张与统计

- 预先指定主要 3D-sensitive endpoint，不能在大量下游任务后挑一个最好结果；
- 每个关键受控消融至少多 seed，报告均值和不确定性；小数据集宜增加重复次数；
- 同时报生成、性质预测和 3D-sensitive 任务，避免用 latent MSE 自证；
- 清楚报告失败模式：构象生成失败、invalid generation、解析失败、OOV、mapping 无效均不可隐藏。

## 7. 主干的下一步决策

1. **先修 P0：** tokenizer 稳定性、NaN 可观测性、数据/mapping 审计。没有这一步，不运行昂贵的完整 Phase 1。
2. **再跑短程诊断：** A 与 B，以及 lambda 的小型扫参和梯度记录；其目的不是刷最好分数，而是判断 current target 是否有害。
3. **只有 A/B 通过后实施 C/D：** 把 MSE 变为可解释的 EMA 3D-aware latent prediction，并保留 current MSE 作为消融。
4. **只有 D 在预注册的 3D endpoint 上稳定超过 A，且不损害生成能力时，才将 MSE 放入论文主方法。**

若这些条件不满足，最严谨、也最有价值的结论是：E3FP 输入本身可能有效，但本项目的 MSE 辅助目标没有被证实，应移除或降为附录消融，而不是继续通过复杂化机制挽救它。

## 8. 本轮新增或重点核验的来源

- 3DMolFormer：Hu et al., ICLR 2025，本地 paper/3DMolFormer_ICLR2025.pdf；原始公开版本：<https://openreview.net/notes/edits/attachment?id=Mer4HrLWeI&name=pdf>
- DynaBERT：Hou et al., NeurIPS 2020，本地 paper/supplementary/DynaBERT_2020_NeurIPS.pdf；原始版本：<https://proceedings.nips.cc/paper_files/paper/2020/file/6f5216f8d89b086c18298e043bfe48ed-Paper.pdf>
- Patient Knowledge Distillation：Sun et al., EMNLP 2019：<https://aclanthology.org/D19-1441.pdf>
- data2vec：Baevski et al., ICML 2022，本地 paper/supplementary/data2vec_2022_ICML.pdf；原始版本：<https://proceedings.mlr.press/v162/baevski22a/baevski22a.pdf>
- E3FP：Axen et al., J. Med. Chem. 2017：<https://pmc.ncbi.nlm.nih.gov/articles/PMC6075869/>
- Uni-Mol：Zhou et al., ICLR 2023：<https://openreview.net/forum?id=6K2RM6wVqKu>
- T5 hidden-state matching：Dasgupta & Cohn, ICLR 2025：<https://proceedings.iclr.cc/paper_files/paper/2025/file/2fb462e23667ad5e6471a4e9af8e4774-Paper-Conference.pdf>
