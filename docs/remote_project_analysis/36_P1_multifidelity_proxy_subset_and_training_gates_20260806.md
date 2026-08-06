# P1 嵌套代理子集、多保真训练预算与晋级门禁

日期：2026-08-06  
状态：PF-CANARY/PF-1/PF-10/PF-FULL 的多保真定义继续有效；2026-08-06 起候选矩阵改为[文档 41](41_scientific_design_comparison_dataset_and_execution_plan_20260806.md)的 A0/A1/M0/M1 四格，不再默认展开 C0/C1-G/C1-L/C1-R/C3。当前只允许生成清单与实现门禁，`P1_ADMISSION=false` 时禁止启动正式训练。

## 0. 主干裁决

可以先使用部分数据，但**不直接从随机 10% 开始，也不把 10% 当成最终科学证据**。当前采用：

> 全量只做数据审计 → 256 条工程 canary → 约 1% 淘汰明显无效方案 → 10% 确认候选排序 → 全量只训练胜出方案与最近因果对照。

这套流程称为**嵌套式多保真验证（nested multi-fidelity evaluation）**，具体淘汰方式是 **Successive-Halving-style gate**。它借鉴逐级增加数据/计算预算并淘汰候选的思想，但不是完整实现 Hyperband。

需要严格区分四个术语：

| 术语 | 本项目中的含义 | 可以宣称什么 |
|---|---|---|
| pilot / canary | 极小真实样本上的工程检查 | 数据流、loss、梯度、显存和断点是否正常 |
| proxy subset | 用小于全量的数据对候选作低成本排序 | 只支持该数据规模、该预算下的局部判断 |
| multi-fidelity evaluation | 逐级增加数据量、训练 token 或更新步数，只晋级少数候选 | 资源受限的候选筛选过程 |
| learning curve | 同一方案在多个数据/计算规模上的性能曲线 | 只有多个规模点才能讨论尺度趋势 |

单独一个 10% 点不是 learning curve；10% 也不是行业统一阈值。随机/分层子集更不能称为 `coreset`：严格的 coreset 通常带有专门的选择目标和对全量目标/梯度的近似要求。

方法学依据：

- [Hyperband, JMLR 2018](https://jmlr.org/papers/v18/16-558.html) 把迭代数、样本数或特征数都视为可逐级分配的资源，并用 Successive Halving 淘汰低表现配置；
- [FABOLAS, AISTATS 2017](https://proceedings.mlr.press/v54/klein17a.html) 明确把验证误差和训练成本建模为数据规模的函数，说明“小数据代理全量”需要被校准，而非天然成立；
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) 与 [Chinchilla, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/c1e2faff6f588870935f114ebe04a3e5-Abstract.html) 表明数据量、训练 token 与计算量共同影响结果，因而只报记录比例不足以保证公平；
- [Neural scaling of deep chemical models, Nature Machine Intelligence 2023](https://www.nature.com/articles/s42256-023-00740-3) 在化学模型中也先验证早期表现与终点排序的关系，再淘汰配置，并系统改变模型与数据规模；
- [Does GNN Pretraining Help Molecular Representation?, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/4ec360efb3f52643ac43fda570ec0118-Abstract-Conference.html) 说明分子预训练收益会随目标、划分、规模、输入和超参数改变，不能把小规模预训练 loss 直接等同于下游收益。

## 1. 分母先冻结，不能直接按旧的 3,119,717 计算

当前主线 profile 是 `pcqm4mv2_candidate`，不是旧的 `legacy_3dmolm_control`：

| 口径 | 记录数 | 用途 |
|---|---:|---|
| PCQM4Mv2 train-3D 原始记录 | 3,378,606 | 原始来源口径 |
| production-v2 geometry admitted | 3,365,577 | 当前已通过几何 release/复算的候选成员 |
| legacy 3D-MoLM/3D-MolT5 P1 source | 3,119,717 | 只保留作 legacy control |
| legacy geometry admitted | 3,119,714 | 只保留作 legacy control |

当前 3,365,577 条仍不是最终训练分母。令：

\[
N_{\mathrm{train\_permitted}}
=N_{\mathrm{geometry\_admitted}}
-N_{\mathrm{downstream\_protected}}
-N_{\mathrm{internal\_validation}}
\]

其中还要落实 P1/P2 overlap 策略。只有下列事项完成后，才冻结 `N_train_permitted`：

1. 冻结下游任务版本、split 和 protected valid/test identity；
2. 对 protected identity 完成连接身份（connectivity）级泄漏过滤；
3. 决定 P1/P2 overlap 的保留/排除规则；
4. 决定 P2 重线性化到 P1 投影域，或明确采用 P1-only motif vocabulary；
5. 冻结 tokenizer、motif identity codec、fallback 与 ordered vocabulary；
6. 划出固定且与训练集 connectivity-group-disjoint 的内部验证集。

这不是要求删除所有 P1 重复记录。相同 connectivity/stereo/conformer 的成员必须作为同一分组处理并报告 multiplicity，不能无声明地把全部预训练数据去重。

按本文统一的向下取整规则、以当前尚未过滤的 3,365,577 条暂算，1% 的目标容量为 33,655 条、10% 为 336,557 条；它们只是容量预估。正式目标必须由最终 manifest 中的 `N_train_permitted` 重新计算；为保持 connectivity group 完整，实际成员数允许在 manifest 中报告极小偏差，不能为了凑整拆组。若单独复现实验使用 legacy profile，3,119,714 或 3,119,717 的 10% 向下取整均为 311,971 条，两个 profile 不得混写。

## 2. 五个数据保真层级

为避免与现有 P1-S0/P1-S1/P1-S2 代码阶段混淆，本节的数据预算统一使用 `PF-*` 命名。

| 层级 | 目标成员规模 | 目的 | 允许的结论 |
|---|---:|---|---|
| `PF-AUDIT` | 全量流式 census | 身份、泄漏、覆盖、分层统计和 manifest 冻结；不训练 | 数据合同是否成立 |
| `PF-CANARY` | 32 条重复过拟合 + 256 条边界样本 | round-trip、mapping、mask、loss、梯度、OOM、save/load | 工程链路是否正确 |
| `PF-1` | `floor(0.01*N_train_permitted)` | 一次配对 seed 淘汰 C0/C1-G/C1-L 中明显失败者，并运行一个冻结的低成本 downstream dev probe | 仅作淘汰，不作显著性或全量主张 |
| `PF-10` | `floor(0.10*N_train_permitted)` | 严格包含 PF-1；确认晋级候选与最近对照 | 10% 规模下的候选效应与排序 |
| `PF-FULL` | 全部 `N_train_permitted` | 只跑最终方案与最近因果对照 | 最终预训练结论的必要证据 |

`PF-30=floor(0.30*N_train_permitted)` 是**条件层级**，不是默认支出。只在以下任一情况出现时加入：

- PF-1 与 PF-10 的候选排序反转；
- PF-10 的效应接近基线波动或 CE 非劣边界；
- 直接进入全量的预估成本明显高于一次 30% 判别实验；
- 论文希望把“小规模结果可预测大规模排序”作为一个方法学主张。

如果论文要正式讨论 learning curve 或 scaling law，应另行建立 1%/3%/10%/30%/100% 多点设计和重复实验；当前的最小资源路线不作该主张。

## 3. 子集不是复制数据，而是冻结引用清单

所有 `PF-*` 子集只生成成员 manifest，不复制 LMDB/SDF，不重新下载数据。每个 manifest 至少保存：

- release/profile/contract/tokenizer/vocabulary SHA-256；
- `N_train_permitted` 与当前 fraction；
- selection seed 与 selection algorithm version；
- 每个成员的 `member_id`、storage key、connectivity group；
- ordered member-list SHA-256 和分层统计；
- 与上一级的 strict-subset proof；
- protected overlap count（必须为 0）；
- internal validation overlap count（必须为 0）。

固定选择方式：在每个分层单元内部计算

```text
SHA256(release_id || stratum_id || connectivity_group_id || selection_seed)
```

按 hash 排序后，在各分层配额内选取完整 group 前缀；若下一个 group 会越过目标配额，则按预注册规则停止或接受并报告最小超额。禁止使用 Python `hash()`，禁止看到结果后更换 selection seed。相同 connectivity group 的所有成员共同进入或共同不进入，以防止相同分子跨 proxy train/validation；组内成员是否全部保留由 P1 membership 政策统一决定。manifest 必须同时记录目标数和实际数。

严格嵌套：

```text
PF-CANARY ⊄ PF-1（canary 是目的性边界样本，可独立）
PF-1 ⊂ PF-10 ⊂ PF-30（若启用）⊂ PF-FULL
```

## 4. 分层与尾部覆盖

在 tokenizer 冻结前，只用 codec-independent 字段建立主分层：

- `model_atom_count` 分箱；
- `motif_count` 分箱；
- attachment 状态：无跨 motif attachment / 只有 attachment 或 core / 同时包含 attachment 与 core；
- geometry validity 与 source shard/ordinal 区间；
- connectivity/stereo group multiplicity。

tokenizer 冻结后，再审计：

- input/target token length P50/P95/P99；
- macro/fallback 数量及序列膨胀；
- motif frequency bucket 与 vocabulary coverage；
- 稀有 anchor/interface role 覆盖；
- E3FP level/状态覆盖。

主训练子集应近似总体分布，避免用大量稀有样本过采样改变训练问题。同时另建固定 `TAIL-AUDIT`，覆盖长序列、稀有 motif、多 attachment、fallback、几何边界和重复组；它只用于分桶评估与错误定位，不混入主验证指标。

不要把所有字段做笛卡尔积。主分层键先限定为 `(atom_count_bin, motif_count_bin, attachment_status)`，其他变量只做边际覆盖审计；某个关键尾部覆盖不足时，再预注册最小的二级约束。

## 5. 公平预算不是“都看 10% 条记录”

同一保真层内，各模型条件必须共享：

- 完全相同的 membership、样本顺序与随机种子；A0/A1 组内和 M0/M1 组内共享 exact mask realization；A/M 因 corruption unit 不同只共享 mask-rate 合同，不能声称 exact realization 相同；
- 同一 frozen union tokenizer snapshot、最大长度、优化器和学习率日程；atom/motif 的序列化与 collator 是研究因素的一部分，不能冒充相同 batch construction；
- 在可行范围内匹配 non-padding encoder tokens/update，并报告实际值；
- A0/A1 与 M0/M1 各自匹配 masked target tokens；跨 A/M 记录并按实际 supervised target tokens 归一化，同时保持相同 optimizer updates 与 early-stop rule；
- 相同初始化 seed；若补第二 seed，所有配对条件一起补；
- 相同 checkpoint/evaluation cadence。

必须同时报告：unique molecules、records、encoder/target tokens、重复次数/epoch、optimizer updates、GPU 小时、tokens/s、峰值显存和失败重启。数据比例描述“覆盖了多少成员”，训练 token/updates 描述“花了多少学习资源”，两者不能互换。

精确 step/token 预算不在 tokenizer 和 4090 吞吐实测前拍脑袋决定：先由 `PF-CANARY` 和一次 `PF-1` 匹配条件学习曲线测得吞吐与最低可判别预算，随后在所有候选训练前冻结预算合同。

## 6. 模型条件的晋级顺序

### 6.1 PF-CANARY

1. 32 条样本重复训练，要求 CE 可明显下降并能复现预期 mask target；
2. 256 条包含 singleton、无 attachment、多 attachment、fallback、长序列和 E3FP 边界样本；
3. 逐字段检查 `identity codec -> token -> logical motif -> atom group -> E3FP motif state -> collator -> model/loss`；
4. hard failure：round-trip 不成立、mapping 越界、空 target、NaN/Inf、错误 padding、manifest/hash 不一致、意外 OOM。

### 6.2 PF-1：先裁决最小科学命题

只先运行：

- `C0`：motif identity + CE，无 3D；
- `C1-G`：molecule-global E3FP mean + CE；
- `C1-L`：motif-local E3FP mean + CE。

相同 seed 的配对比较只用于淘汰。`C1-L` 必须在预注册的 3D-sensitive/locality endpoint 上优于 `C1-G`，同时 CE/identity recovery 不越过预注册非劣界，才允许宣称 motif-local 3D 有候选价值并实现 C1-R。单 seed 不能给出显著性结论。

PF-1 同时只运行一个代表性 downstream dev probe，防止用预训练 CE/3D loss 代替表征价值。当前首选是与既定 Property Prediction 方向一致的 **QM9 HOMO-LUMO gap**：冻结 encoder，只训练统一的小型 linear/MLP probe；所有候选共享相同 train/validation manifest、probe 初始化与预算，预先冻结的 test 保持封存。若采用的基准没有唯一官方 split，应先选择可复现的既有协议或按 connectivity group 一次性冻结 split，不能事后挑选。可附加“保持 2D identity、扰动/置换 3D 输入”的同一 probe 诊断切片。它只是开发门禁，不是最终下游成绩；若后续下游组合裁决更换首要 3D-sensitive 数据集，则在任何模型结果产生前一次性替换并冻结。

### 6.3 PF-10：只确认晋级者

- 默认只保留 PF-1 胜出方案与最近因果对照；
- 若 C1-R 在 PF-1 短测为正，PF-10 必须同时纳入统计/参数匹配的 `C1-Rpseudo`；
- 第一组配对 seed 方向为正且 CE 非劣后，才补第二组配对 seed；
- 分别报告无 attachment、有 attachment、同时含 attachment/core、motif size、fallback 和稀有 motif 桶；
- 重复 PF-1 已冻结的同一个 downstream dev probe，不能用预训练 CE 或 3D loss 单独代替下游表征判断；不在此层展开 captioning/editing/retrieval 的完整任务组合。

### 6.4 teacher/MSE 与 PF-FULL

先选出胜出的 CE-only E3FP 模型，再实现/训练 C3（EMA teacher masked latent prediction）。C3 只与该 CE-only winner 配对比较；若 3D endpoint 不增益或 CE/生成越过非劣界，teacher 从主方法删除。legacy raw online MSE（C2）只在解释历史 checkpoint 时短测。

全量默认只跑：

1. 最终方案；
2. 能隔离核心因果主张的最近对照。

最低工程证据是一组完整配对运行；面向顶刊的提交目标应是这两个模型至少 2 个独立配对 seed，若 baseline 方差较大或效应接近非劣界则补到 3 个。不会把 C0/C1-G/C1-L/C1-R/C1-Rpseudo/C2/C3 全部在全量上重跑。

## 7. 预注册指标与停止规则

主指标必须至少分成四类：

1. **语言/身份保持**：held-out CE、motif identity recovery、生成有效性；
2. **3D 使用**：预注册 3D-sensitive endpoint、atom-to-motif E3FP shuffle delta、同一 2D identity 的构象敏感 probe；
3. **motif-native 归因**：C1-G vs C1-L、C1-L vs C1-R、C1-Rpseudo vs C1-R；
4. **工程成本**：吞吐、峰值显存、fallback 膨胀和错误率。

CE 非劣 margin 必须先由 baseline 重复的自然波动确定，不能看完候选结果后再设。以下任一项触发硬停止：

- downstream protected overlap 非 0；
- proxy train/internal validation connectivity overlap 非 0；
- tokenizer/release/member manifest hash 不匹配；
- identity/motif/atom/E3FP 映射断裂；
- NaN/Inf、空 target、严重 collapse；
- PF-1 与 PF-10 排序反转却仍直接外推全量。

## 8. 从当前状态开始的执行顺序

| 顺序 | 工作 | 资源 | 出口 |
|---:|---|---|---|
| 1 | 冻结下游任务版本/split/protected identities | CPU | downstream protection manifest |
| 2 | 裁决 P1/P2 overlap 与 P2 projection/vocab 路线 | CPU/文档 | membership + vocab policy |
| 3 | 实现并验证 hybrid motif identity codec、ordered vocabulary、fallback | CPU | deterministic tokenizer gate |
| 4 | 生成 `N_train_permitted` 与 PF-1/PF-10 manifests；全量 census/覆盖审计 | CPU/顺序 IO | `PF-AUDIT=PASS` |
| 5 | 在保留原代码的 `most_t5_next` 中实现 BoundRecord/Collator 和 C0/C1-G/C1-L | CPU | unit/contract tests |
| 6 | PF-CANARY forward/backward/overfit/save-load | RTX 4090 | engineering gate PASS |
| 7 | PF-1 配对筛选 | RTX 4090 | locality gate |
| 8 | PF-10 只确认晋级者；必要时补 seed 或 PF-30 | RTX 4090 | scale-confirmed candidate |
| 9 | CE-only winner 后再裁决 C3 teacher | RTX 4090 | MSE 增量裁决 |
| 10 | PF-FULL：winner + nearest control | RTX 4090 | 最终预训练证据 |

因此当前**不是立刻复制 10% 数据并开训**。先完成 1–4；随后生成的只是对现有 release 的可复现引用清单，不增加一份数据盘副本。当前状态依据见：[R1 semantic recompute](29_R1_semantic_recompute_and_worker_IPC_boundary_20260805.md)、[P1/P2 identity overlap](30_R1_P1_P2_identity_overlap_facts_and_parallel_extraction_20260805.md)、[P2 projection compatibility](31_R1_P2_motif_census_and_projection_compatibility_20260805.md) 和 [统一 motif 架构路线](35_unified_motif_identity_codec_ema_teacher_and_integrated_roadmap_20260805.md)。

## 9. 论文表述边界

PF-1/PF-10 可以用于开发期模型选择和报告透明的资源门禁，但不能把“10% 上更好”写成“全量可行已证明”。只有最终全量配对实验和冻结下游任务上的一致收益才能支撑主结论。

若后续确实验证 PF-10 能稳定预测更高保真层的候选排序，可以报告候选排名的 Spearman/Kendall 相关或 selection regret，并称其为“经验证的 proxy”；在此之前只称“低保真筛选层”。
