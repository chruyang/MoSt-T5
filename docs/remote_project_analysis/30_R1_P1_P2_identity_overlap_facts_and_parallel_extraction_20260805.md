# R1 P1/P2 身份交集事实与并行提取（2026-08-05）

状态：P1 parallel identity extraction PASS；P2 geometry-ready candidate identity extraction 双跑确定性 PASS；P1-vs-P2 candidate overlap facts 已报告。`P1_ADMISSION=false`、`P2_ADMISSION=false`、`all_downstream_exclusion_proven=false`、`p1_p2_policy_compliance_proven=false`。

## 1. 为什么先报告事实，不直接裁决策略

P1 PCQM4Mv2 与 P2 PubChem 都属于预训练阶段。二者存在同分子并不自动构成 evaluation leakage；对于第二阶段跨模态对齐，受控复用同一分子还可能提供阶段间桥接。相反，任一预训练集合与 downstream validation/test 的 connectivity overlap 必须单独证明为零。

因此本轮将两件事分开：

1. 严格生成并绑定 P1、P2 的共享 connectivity/stereo identity collection；
2. 只报告 P1↔P2 的 exact set facts，不输出 overlap policy PASS，也不冒充 all-downstream proof。

## 2. P2 geometry-ready candidate identity collection

源：

`/root/autodl-fs/most-t5-r1/sources/p2-pubchem-evidence-r0-v1/pubchem/pretrain/phase2_pubchem_ready.lmdb`

- source bytes：1,909,297,152
- source SHA-256：`465d89f4aafb36043a5964441feffceb3e3e6493fe2ffee9d53190ec7587d5e5`
- payload records：301,655
- observed metadata keys：0
- RDKit：2024.03.5
- identity normalization contract：`5f9be346294e08bf73d47c089a00be4c2f19d89612b5e4c09d0d7f5f6b23b044`

首次输出：

`/root/autodl-fs/most-t5-r1/reports/cpu-p0-20260805/p2-ready-identity-collection-v1-destination-20260805T101301Z`

fresh-process rerun：

`/root/autodl-fs/most-t5-r1/reports/cpu-p0-20260805/p2-ready-identity-collection-v1-rerun-destination-20260805T102803Z`

两次运行分别使用不同 `PYTHONHASHSEED`，均 PASS。以下四个确定性产物逐字节一致：

- molecule rows：113,107,471 bytes，301,655 rows，SHA-256 `34cc1a0d839afd7d5e3b463c51b1dbccc11e823596c4383f94ee0b4550c13af7`
- collection manifest SHA-256：`8c4ed46098088e16add92262209f3cd95f26846e683d0abfed21071320248d7c`
- resolved config SHA-256：`ac36e19a292e59655a89dcb18770a9625b563f8040b468e0a1be2e22916105b9`
- source lock SHA-256：`c9433742bb40570296b1b983853239a559f3711ecd833ae3a75226dbf11f6346`

`extraction_report.json` 只因生成时间及相应自哈希不同；移除 `generated_at_utc` 与 `report_canonical_payload_sha256` 后，两份报告完全相同。

## 3. P1 串行瓶颈与并行旁路

原 extractor 对 136 个 release shard 串行处理，Linux `ps` 显示约 60%–100% CPU 只代表不足一个核；在 16 vCPU 上，整机利用率只有约 4%–6%。临时 SQLite 又位于普通速率 `autodl-fs`，墙钟时间预计超过 70 分钟。

新增并行旁路保留原串行程序不变：

- 完整 shard 是最小并行单位；
- 默认 8 个 `spawn` worker；
- 每个 worker 保持 membership/reject/payload-index/LMDB 锁步、安全 payload 独立解码、identity/contract/RDKit 绑定及 shard artifact 前后哈希；
- scratch shard JSONL 与全局 SQLite 放入 `autodl-tmp`；
- 父进程要求 136 个结果完整、唯一、连续，重新读取并哈希 scratch，检查全局 member 唯一性，再按 SQLite `BINARY` 顺序输出；
- worker 失败不创建 output；scratch 不自动删除，便于失败诊断。

主干发现并修正了 Python 3.8 兼容点：不得调用 Python 3.9+ 才支持的 `shutdown(cancel_futures=True)`；远端 Linux/Python 3.8 专项测试随后为 10/10 PASS。

并行证据：

- script SHA-256：`e6b0a05275aa1e688c91df87c71cb9f7bb63502db35c62a361108bab20ad135f`
- parallel contract SHA-256：`f716c48562b2251ff64d3e1653ef4b2a144776707efa1f082d079f917acfff7a`
- config SHA-256：`0291440ff70b561e8a8181759cb2076b386f8033cafea5f2629d66c8413848ef`
- worker scan：约 7 分 37 秒
- 完整运行：约 11 分 24 秒
- 相对串行预计总时长：约 6 倍加速

输出：

`/root/autodl-fs/most-t5-r1/reports/cpu-p0-20260805/pcqm-identity-collection-parallel-v1-destination-20260805T105845Z`

- source membership：3,378,606
- admitted/emitted：3,365,577
- rejected：13,029
- shard count：136
- molecule rows：1,348,489,549 bytes
- molecule rows SHA-256：`44f98d41de48a7b81b3315b79cb80ab8fbb63a4d43792c45edd5569e7cbe47c4`
- key-LF SHA-256：`159b1688effe34d75be6613368cc9a0e08bbea1c398baccba392a1541de1b001`
- collection manifest SHA-256：`2b50b7455e36c7d14476d69572f68ae04f87d87e66e8745760a5c5c4f5cd5dcb`
- extraction receipt SHA-256：`69b2e554d2ce55d43f4f3977bc012637b2916ea6e09300c401c38821ba632d67`
- receipt canonical self-hash：独立复算匹配

原串行任务继续作为真实基线；完成后只比较 core deterministic boundary（molecule rows 的 bytes/SHA/row count/key-LF），不要求 serial/parallel manifest 字节相同，因为两者的 extractor 与 execution contract 不同。

## 4. P1/P2 candidate overlap facts

诊断工具直接复用正式 proof consumer 的严格 `load_collection()`，不复制或弱化 manifest、JSONL、artifact、key digest 与 identity spec 校验。

报告：

`/root/autodl-fs/most-t5-r1/reports/cpu-p0-20260805/p1-p2-candidate-overlap-facts-v1-destination-20260805T111155Z/p1_p2_candidate_overlap_fact_report.json`

报告 canonical self-hash、两侧外部期望 manifest SHA、实际 manifest/molecule-row bytes/SHA/count/key-LF 均已独立复核。

### 4.1 集合内部规模

| 指标 | P1 PCQM admitted | P2 geometry-ready candidate |
|---|---:|---:|
| molecule members | 3,365,577 | 301,655 |
| unique connectivity | 3,145,515 | 208,661 |
| unique stereo | 3,243,454 | 301,571 |
| connectivity duplicate groups | 165,389 | 49,336 |
| stereo duplicate groups | 105,360 | 81 |

P2 unique connectivity 显著少于 stereo，说明 connectivity 归一化按设计折叠了大量立体变体；不能把 connectivity duplicate 数直接解释成重复原始记录。

### 4.2 跨阶段交集

| 指标 | 数值 | 相对比例 |
|---|---:|---:|
| unique connectivity overlap | 8,600 | P2 unique connectivity 的 4.1215% |
| P1 members impacted by connectivity | 17,313 | P1 members 的 0.5144% |
| P2 members impacted by connectivity | 12,452 | P2 members 的 4.1279% |
| unique stereo overlap | 5,676 | P2 unique stereo 的 1.8821% |
| P1 members impacted by stereo | 10,349 | P1 members 的 0.3075% |
| P2 members impacted by stereo | 5,676 | P2 members 的 1.8816% |
| P1 connectivity overlap without stereo match | 6,964 | report-only |
| P2 connectivity overlap without stereo match | 6,776 | report-only |

Conformer 与 text identity 在两份 membership collection 中均 unavailable，本报告不对其作出结论。

## 5. 策略判断与实验要求

当前事实排除了“P1/P2 天然零交集”的假设，但不支持直接宣称该交集有害。建议主线先登记 `explicitly_declared` 或 `replay_permitted` 候选，而不是未经实验证据强制 `disjoint_required`。

顶刊级消融至少比较：

1. **retain-overlap**：保留 301,655 个 P2 geometry-ready candidate；
2. **connectivity-disjoint**：从 P2 中排除受影响的 12,452 个成员，形成新的、哈希闭合的 P2 membership；
3. **stereo-disjoint**：只排除严格 stereo 重合的 5,676 个成员，用于区分 connectivity bridge 与 exact stereoisomer replay 的贡献。

三组必须保持训练 token 数或 optimizer steps 可比，并分别报告：P2 对齐损失、P1 几何保持、caption/text2mol、property prediction 及最终 downstream 泛化。若 retain-overlap 更好，只能解释为“受控阶段桥接与 alignment replay 的实证收益”，不能扩展为允许 downstream evaluation overlap。

## 6. 仍未完成

1. P1 串行基线尚待完成并与 parallel core 逐字节比较。
2. 301,655 vs 301,658 的最终 P2 membership 尚未裁决；当前报告只覆盖 geometry-ready candidate。
3. P2 alignment/text pair collection、geometry replay collection 尚未按真实 Dataset/Collator 路由构建。
4. downstream task/split matrix、文本 identity 规范、MoleculeNet split/seed 尚未冻结。
5. P2 motif census source copy manifest 与 pickle trust-basis 尚未形成正式 artifact，不能用占位 SHA 启动真实 census。
6. P1/P2 motif projection compatibility 预期不通过；现有 P2 legacy anchors 与 P1 global pair semantics 仍未证明等价。
7. 正式 P1/P2/downstream overlap proof 与 tokenizer binding 均未完成，因此训练准入保持 false。

