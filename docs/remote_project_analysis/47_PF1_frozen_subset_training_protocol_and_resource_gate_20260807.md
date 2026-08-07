# PF-1 子集、训练协议与资源切换门槛（2026-08-07）

状态：**PF-1 的 33,600 条 paired release、全量 collator 门禁、34,666-token union-init 与单卡四格训练均已完成。A0/A1/M0/M1 各完成 1,000 次 optimizer update，8 个 checkpoint 与最终 manifest 齐全。当前 raw E3FP sum + direct residual addition + CE-only 的几何路径没有通过 PF-1：A1/M1 到 step 500 后几何打乱 ΔNLL 已接近 0，且最终表现不优于各自无 3D 对照；该融合不能原样进入 PF-10。**

## 1. PF-1 的角色

PF-1 是约 1% 数据上的单 paired-seed 淘汰层，只运行 A0/A1/M0/M1。它用于发现无法稳定训练、3D 完全不被利用或 motif 表征明显失败的方案；不能给出显著性、全量性能或顶刊主结论。

四格必须共享：

- 同一 train/dev membership 与顺序；
- 同一 union tokenizer 和 union-init；
- 同一 optimizer、schedule、effective batch、更新次数和验证节奏；
- 配对条件内部相同 corruption：A0=A1、M0=M1；
- 同一个训练 seed，不允许结果产生后为某一格追加 steps 或单独调参。

## 2. group-complete PF-1 membership

final-v4 permitted P1 成员数为 3,360,067，1% 的向下取整目标为 33,600。`permitted_member_ids.jsonl` 只含 member ID，因此需要与冻结的 PCQM pretrain identity rows 流式关联，取得 `connectivity_identity` 和 SDF ordinal。

选择规则：

1. 以 connectivity identity 聚合完整 group，任何 group 不拆分；
2. 使用 NumPy PCG64、seed `20260807` 对 group 做一次确定性置换；
3. 在该置换顺序上选择最接近 33,600 members 的完整 group 前缀；
4. PF-10 以后继续使用同一 group 顺序的更长前缀，保持严格嵌套；
5. 对 PF-1 已选 groups 使用独立 PCG64 seed `20260808` 做 group-disjoint 90/10 train/dev，目标约 30,240 / 3,360，实际成员数随完整 group 略有偏差并如实报告。

这里采用普通科研随机化，不使用 Python `hash()`，也不把安全哈希映射当作抽样方法。production release、final-v4 membership 与原 SDF archive 均只读。

实测冻结结果为 33,600 members、31,445 个完整 connectivity groups；train/dev 恰好为 30,240 / 3,360 members，且 connectivity group 不交叉。后续 PF-10 仍须从同一冻结 group 顺序延长前缀，不能重新抽样以追逐结果。

## 3. 1,024 条真实物化基准：CPU 路线成立

修复 codec 后，`westc` 16-worker 基准对同一冻结 1,024-member 前缀完成了 production release 只读读取、完整 SDF 顺序投影、raw/inherited E3FP、motif linearization、AtomSELFIES 与 GraphPorts surface：

| 指标 | 实测结果 |
|---|---:|
| members | 1,024 |
| pass / reject | 1,024 / 0 |
| 总墙钟 | 96.970 s |
| SDF 顺序扫描 | 92.225 s |
| 16-worker pool | 2.947 s |
| worker CPU 累计：E3FP | 30.653 s |
| worker CPU 累计：AtomSELFIES + GraphPorts | 9.976 s |
| 同吞吐下投影的 PF-1 worker 增量 | 96.695 s |

基准当时给出的“两次全 SDF 扫描 + worker”投影为 281.146 s，但明确不包含 tokenizer 与 LMDB publication，因此只用于决定是否值得启动正式构建，不能冒充端到端耗时。92.225/96.970 秒来自顺序解压与 RDKit 扫描，说明负载不能长期达到 32 核满占用并非 worker 配置失效；可并行化的分子处理已由有界进程池承担。

该基准使“另租 GPU 加速物化”失去必要性；GPU 不参与 RDKit/E3FP 数据构建。随后正式 run3 已完成，所以这张表现在是资源决策的实测依据，而不是待执行计划。

## 4. run1 reject taxonomy、严格修复与 SELFIES 2.2.0 覆盖

首次 33,600-member 构建共出现 25 个 reject。按真正故障域拆分，而不是把所有特殊分子混为一类：

| 故障域 | 行数 | 事实分类 | 处理 |
|---|---:|---|---|
| AtomSELFIES 本地结构角色解析 | 3 | 合法 `[-/Ring1]` 双字符方向键前缀未被旧正则接受 | 窄化修复 ring grammar，并保留真实分子 E/Z 回归 |
| SELFIES 2.1.1 支持/回环边界 | 19 | 11 条 strict encoder kekulization failure；8 条 strict round-trip 改变自由基芳香图或桥环立体身份 | 正式固定到 SELFIES 2.2.0；不使用 `strict=False`、全 `-1` 对齐或静默跳过 |
| GraphPorts 跨端口立体语义 | 1 | SDF ordinal 63,194 的跨 port `C=N/O` E/Z 在重连后丢失 | 保存 support-relative CIS/TRANS，并在重连后恢复 |
| GraphPorts 对称多环规范表面 | 2 | ordinals 2,447,063 与 3,140,645 的 mapped fragment 在对称图上出现非幂等首选 spelling | 将严格身份表面定义为明确 fixed point |

AtomSELFIES 的 19 条 SELFIES 2.1.1 边界样本跨 18 个 connectivity groups；这曾触发“support-aware re-freeze”备用方案。但正式 codec 在 SELFIES 2.2.0 下已经覆盖原冻结 cohort，因此主线没有换样本，也没有通过删除困难分子得到零拒绝。

正式 AtomSELFIES 2.2.0 实现同时固定发行版元数据和官方语义约束，采用 parse-reserialize fixed point，并组合原子输出次序以保持 token 到冻结原子行的一一映射。全 33,600-member 独立扫描结果为：

- strict PASS `33,600/33,600`，fail `0`；
- `strict_false_used=false`；
- 28 workers，总墙钟 159.238 s，其中 SDF generator 推进 154.611 s；
- 三个 `[-/Ring1]` 真实记录全部保留。

因此当前结论是“在这批冻结 PF-1 分子上，正式 AtomSELFIES 2.2.0 + GraphPorts codec 满足严格可逆合同”，而不是“SELFIES 或 GraphPorts 对所有化学空间无条件完备”。

## 5. PF-1 run3 paired release

正式输出 `pf1-paired-release-v1-run3` 已通过 publication 与完整 LMDB decode replay：

| 项目 | run3 实测 |
|---|---:|
| scheduled / paired / reject | 33,600 / 33,600 / 0 |
| train / dev | 30,240 / 3,360 |
| union tokenizer vocab | 34,666 |
| base / added rows | 32,100 / 2,566 |
| AtomSELFIES syntax：cohort / train / dev observed | 107 / 105 / 76 |
| train-only motif macro registry | 2,150 identities |
| materialized macro / lossless fallback occurrences | 235,517 / 6,282 |
| LMDB `data.mdb` | 351,236,096 bytes |
| 完整 decode replay | 33,600 / 33,600 |

词表包含两类统计边界不同的对象，必须分开解释：

1. **AtomSELFIES 语法注册表**来自完整冻结、无标签的 PF-1 cohort，只用于保证输入语言闭合；不使用标签、性能、频率或排序。dev-only 的 `[P@@H1]`、`[\P]` 在优化前注册，但 dev records 不参与优化。这与 CAMT5/3D-MolT5 在 train/eval 前加载固定 SELFIES vocabulary 的接口原则一致。它们在 train 中没有正例/输入出现，仍不能据此声称对应新 embedding 已得到充分学习。
2. **motif macro registry**只由 train split 的 identity 频率拟合，最小出现次数为 2；dev 未见 identity 走显式、可逆 GraphPorts fallback，不读取 dev 频率选 macro。

完整 replay 证明无需截断：Atom/Motif 最大输入长度分别为 40/137，遍历全部允许 mask 后最大 target 长度为 42/92，最大 sentinel 数为 21/17，均低于 T5 的 512-token 与 100-sentinel 容量。这里的 2,150 macros 和 34,666-token union tokenizer 仍是 **PF-1 sample-bound candidate**；它们不是 3,119,717 条第一阶段预训练数据的最终词表或最优 K。

## 6. full-collator gate：完整数据流已经闭合

完整 gate 以 batch size 8、冻结顺序各读取一次 train/dev，对 A0/A1/M0/M1 分别遍历 33,600 members（train 3,780 batches，dev 420 batches，每格共 4,200 batches），结果均为零 member/batch reject。

关键配对合同在全部 batches 上成立：

- `A0 == A1` 的 CE `input_ids/attention_mask/labels`；
- `M0 == M1` 的 CE `input_ids/attention_mask/labels`；
- `A1 == M1` 的原子行、四层 E3FP 与 source mapping；
- CE 张量为 rank-2 `int64`；A1/M1 的 `e3fp_ids` 为 rank-3 `int64`、level width 4，atom mask 为 rank-2 `bool`，carrier mapping 为 rank-2 `int64`；
- 所有 33,600 个成员在本轮确定性 corruption 中至少选择一个 mask。

本轮 collator 的实际最大值为 Atom 输入/target/sentinel `40/20/10`，Motif 为 `137/78/8`；它低于上节“遍历全部允许 masks”得到的最坏上界，两组数字回答的问题不同，不能混用。

### 6.1 mask coverage 的 A/M 差异

| 粒度 | mask-unit coverage | identity-token coverage | atom coverage |
|---|---:|---:|---:|
| Atom（A0=A1） | 15.791% | 15.791% | 15.791% |
| Motif（M0=M1） | 19.816% | 23.976% | 22.007% |

这不是 Motif 条件“天然获得更好指标”的证据。Atom 中一个 mask unit 对应一个原子/identity token；Motif 中一个 unit 可覆盖多个原子，且 macro 与 GraphPorts fallback 的 identity token 数不同，所以 unit、identity-token 与 atom coverage 本来就不会相等。科研比较只允许在各自配对内使用相同 corruption（A0 对 A1、M0 对 M1）；A 与 M 的原始 CE、target 长度或 coverage 绝对值不可直接排序。跨粒度结论必须由相同冻结的 3D-sensitive probe 和下游任务裁决。

该 gate 只验证数据、mask 与 tensor I/O，没有执行模型 forward、optimizer step，也不构成架构效果证据。

## 7. PF-1 统一优化协议

依据 3D-MolT5 与 CAMT5 官方训练代码的共同做法，以及 paired-128 中 M1 在 plain AdamW `5e-4` 无 warmup 时的反弹，PF-1 冻结为：

| 项目 | 冻结值 |
|---|---:|
| optimizer | AdamWScale |
| beta / epsilon | `(0.9, 0.999)` / `1e-6` |
| weight decay | 0 |
| base LR | `1e-3` |
| total optimizer updates | 1,000 |
| warmup | 前 100 updates，线性升至 base LR |
| decay | 后 900 updates cosine 到 `1e-5` |
| global gradient clip | L2 norm 1.0 |
| precision | BF16 autocast |
| effective batch | 128 members |
| 单卡实现 | microbatch 32 × gradient accumulation 4 |

该调整不改变 effective batch、成员顺序、更新次数或四格公平性。它是在任何 optimizer update 或训练结果产生前，由同一 RTX 4090 上的最长 M1 序列资源探针确定；本次正式 PF-1 结果必须按实际使用的 32×4 解释。

正式运行结束后的压力复测使用 run3 中 64 条互不重复的最长 M1 记录（最大 input/target width `137/78`）。batch 48 与 64 均完成真实 BF16 forward/backward/AdamWScale step；batch 64 连续 10 步仍通过，峰值 allocated/reserved 为 `22.852/24.505 GB`，设备总量 `25.251 GB`。因此对同一长度支持域，下一轮单卡可使用 64×2；若 PF-10 出现更长序列，先按其真实最大宽度复测，而不是机械沿用 PF-1 数字。

runner 已加入深度 2 的有序预取。真实 M1 50-update 对照中，同步/预取墙钟为 `43.72/41.35 s`，吞吐为 `146.4/154.8 members/s`，提速约 `1.057×`，最终 cursor 完全一致。该结果说明预取值得保留，但 CPU 解码不是主要瓶颈；刻意占满 16 核没有依据。commit `716e597` 还加入了 `--condition-id` 单格运行与四格 manifest 合并器，4×4090 可各跑一格；顺序单进程仍不会因增加可见 GPU 而自动并行。

1,000 updates 对约 30,240 个 train members 相当于约 4.23 次成员曝光。PF-1 不 early-stop，也不因某格收敛较慢而延长。train corruption 随 epoch/position 确定性变化；dev corruption 使用固定独立 seed。

## 8. 验证、checkpoint 与比较口径

在 step 0、250、500、750、1,000 对同一 dev membership 和固定 mask 评估：

- token-weighted CE/NLL；
- teacher-forced masked-token accuracy；
- non-padding encoder/target token 数。

mask unit、selected atom 与 identity-token coverage 由优化前的 full-collator gate 对冻结 corruption 一次性统计；它们不随模型 step 改变，不在每次 dev evaluation 重复计算。训练全程另汇总吞吐、峰值显存、gradient norm，以及裁剪前全局范数超过 1.0 的更新比例。

A1/M1 在最终 step 增加 aligned E3FP 与完整 dev 内、相同 model-atom-count 的跨分子 E3FP derangement 之间的 ΔNLL 诊断。常规 aligned 指标仍覆盖全部 3,360 条；run3 中 atom-count=4 只有一条，数学上无法无自配对地置换，因此仅从 perturbation matched subset 中预先排除并单独报告，诊断覆盖 3,359/3,360，且不替样、不裁剪、不填充或自配。该诊断只检查模型是否使用 geometry，不能单独证明 3D 因果。checkpoint 只保存 step 500 的可恢复状态和 step 1,000 的最终状态；best-dev 仅记录，不用于回头选择不同 step。

比较边界：

- A0 vs A1、M0 vs M1 可分别比较，因为各自 CE target 与 mask 配对；
- A vs M 的 mask unit、target 长度与实际原子覆盖不同，禁止按绝对 CE 排名；
- 跨粒度优劣只由同一个冻结 3D-sensitive dev probe 和后续下游任务判断；
- PF-1 只做淘汰，晋级方向还需 PF-10 和 full 证据确认。

## 9. PF-1 实际训练结果与架构裁决

正式单卡运行使用 commit `3a74251`、microbatch 32×accumulation 4，约 67 分钟完成四格；状态为 `pass`。以下只做同粒度配对比较，A 与 M 的绝对 NLL 不横向排名：

| 条件 | step-0 NLL | step-1000 NLL | step-1000 accuracy | 最终 shuffled-minus-aligned ΔNLL | wall s | peak GiB |
|---|---:|---:|---:|---:|---:|---:|
| A0 | 92.5890 | 0.6381 | 0.8333 | — | 946.59 | 6.61 |
| A1 | 76.9112 | 0.6756 | 0.8308 | +0.000195 | 992.68 | 6.75 |
| M0 | 67.8328 | 1.4231 | 0.6775 | — | 955.88 | 12.30 |
| M1 | 68.1980 | 1.9156 | 0.5772 | +0.000848 | 1044.79 | 12.43 |

四格均能学习，但所有 1,000 个 update 都触发 global-norm clip 1.0。A1 相比 A0 的最终 NLL/accuracy 没有改善；M1 相比 M0 明显退化。更关键的是几何敏感性轨迹：

| 条件 | step 0 ΔNLL | step 500 ΔNLL | step 1000 ΔNLL |
|---|---:|---:|---:|
| A1 | +0.222940 | −0.001435 | +0.000195 |
| M1 | +0.230766 | +0.003304 | +0.000848 |

这说明几何入口并非从一开始就完全无效，而是训练很快学会忽略它。参数审计进一步排除了“几何表被压到零”：四张表初始 RMS 约 `1.0002`，step 1000 后仍约 `1.0002`；A1/M1 几何参数相对初始 RMS 仅改变约 `0.88%/0.61%`。当前实现把四层默认尺度约 1 的 embedding 直接求和后加到 T5 carrier，没有方差校准、归一化或受控残差；标准 CE 也没有显式约束表示持续保留 3D。两者是下一轮优先验证的机制解释，而不是已经被单次实验分别证明的唯一因果原因。因此 PF-1 的科学裁决是：

1. 保留 A0/M0 作为有效基线；不按其绝对 NLL判断 atom 与 motif 谁更优；
2. 淘汰当前 A1/M1 的 **raw sum + direct add + CE-only** 组合，而不是据此否定 E3FP 或 motif 级 3D；
3. 下一轮先设计一个统一、文献支撑的受控几何残差与明确几何学习信号，再做同样的 A0/A1、M0/M1 配对屏；
4. 在新几何方案通过 sensitivity gate 前，不扩大到 PF-10 或完整预训练。

## 10. 当前资源结论与下一步

paired-128 的 AdamW learnability 实测在 batch 8 时峰值约 5.1–5.3 GB，单步约 0.11–0.15 秒；正式 PF-1 数据与 collator 已在 CPU 端闭合。与 **34,666-token** run3 tokenizer 精确绑定的新 union-init 已在 nmb1 生成并通过验证：base/union vocab 为 `32,100/34,666`，model seed `20260807`，geometry seed `20260808`，E3FP embedding count `4,096`。paired-128 的旧 32,499-token union-init 未被复用。

本轮执行结果为：

1. **已完成：**从同一 `google-t5-v1_1-base` snapshot 建立 34,666-token union-init，复制原 32,100 行权重，按预注册 seed 初始化新增行，并保存 tokenizer/model 绑定合同；
2. **已完成：**A1/M1 使用同一 geometry fusion 初始化并通过 save/load 与 BF16 数据流门；
3. **已完成：**1×4090 顺序完成 A0/A1/M0/M1 的共享 1,000-update 协议；
4. **已完成：**step 0/250/500/750/1000 指标及 step 0/500/1000 geometry perturbation 轨迹均已获得；当前几何融合被 PF-1 淘汰；
5. **下一步：**在文献与机制诊断基础上冻结新的受控几何残差/学习信号，只重跑最小必要配对格；代码已支持 4×4090 各跑一格并合并结果。

当前不需要 8×4090。单卡已证明 PF-1 可在约一小时内完成；若下一轮需要同时比较四格，4×4090 的单格进程接口已经实现，8 卡没有额外收益。最终预训练再单独实现/验证数据并行，不能把四格并行与单模型 DDP 混为一谈。

最后必须保持证据边界：PF-1 是约 1% 样本上的 failure screen。run3 的零拒绝证明这批数据可训练，未来 PF-1 loss/probe 只能淘汰明显失败条件；二者都不能推出 3,119,717 条第一阶段预训练已可行、motif/3D 已有效或最终词表已冻结。最终方法主张仍需 PF-10 尺度确认、胜出架构的完整预训练以及冻结下游任务结果。

## 11. 机器可读证据锚点

- 1,024 基准：`tmp/pf1_materialization_benchmark_1024_westc_w16_run2_manifest.json`
- AtomSELFIES 分类：`tmp/pf1_atom_selfies_reject_taxonomy_v1.md` 与 `tmp/pf1_atom_selfies_reject_diagnosis_v1.jsonl`
- SELFIES 2.2.0 全 cohort 扫描：`tmp/selfies220_full33600_formal_summary_v2.json`
- run3 manifest：`tmp/pf1_paired_release_run3_manifest.json`
- full-collator gate：`tmp/pf1_full_collator_gate_v1_run3.json`
- PF-1 训练 manifest：`tmp/pf1_training_manifest_run3_mb32_3a74251.json`
- M1 batch 48/64：`tmp/pf1_gpu_batch_ceiling_m1_48_64_20260807_v2.json`
- M1 batch 64 连续 10 步：`tmp/pf1_gpu_batch_ceiling_m1_b64_repeat10_20260807.json`
- ordered prefetch 实测：`tmp/pf1_gpu_prefetch_benchmark_m1_50updates_20260807.json`
- geometry sensitivity 轨迹：`tmp/pf1_geometry_sensitivity_trajectory_20260807.json`
- geometry 参数轨迹：`tmp/pf1_geometry_parameter_trajectory_20260807.json`
- 远端正式 release：`/root/autodl-tmp/most-t5-r1-pf1/pf1-paired-release-v1-run3`
- 本地便携归档：`dataset/pf1-run3-transfer-20260807.tar.gz`
- nmb1 持久化副本：`/autodl-fs/data/most-t5-r1/pf1-run3-transfer-20260807.tar.gz`

便携归档包含 run3 paired release、冻结 membership、完整 collator 报告与 SELFIES 2.2.0 全量扫描证据；已在本地与 nmb1 分别验证目录可读。因此 westc 临时 CPU 实例不再是这些结果的唯一载体，可以关闭而不会中断下一阶段。
