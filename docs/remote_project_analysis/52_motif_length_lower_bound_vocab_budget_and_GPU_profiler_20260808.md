# Motif 长度下界、词表预算与 GPU 分段 profiler

> 日期：2026-08-08
> 状态：PF-1 run3 的 33,600 条 v2 长度与 train-only 词表预算审计完成；稀有 motif 的 lossless byte-BPE 仅完成长度探针；独立 GPU profiler 已通过无卡代码门，待下一次单卡执行
> 研究边界：本文件不改变 motif partition、GraphPorts v2、正式 tokenizer、优化器或 G-Codec Gate；所有词表数字均为 1% sample-bound 诊断，不是最终 311 万预训练词表

## 1. 结论

1. GraphPorts v2 在当前 33,600 条上已经精确达到 `4+2E`：33,600/33,600 条 contracted motif graph 都是 forest，且没有一条存在可从当前 byte grammar 中继续删除的冗余 edge token。
2. 当前 M 输入均值为 31.215，AtomSELFIES 为 23.321；M>A 的比例仍是 71.256%。这不是单一原因造成的：
   - graph stream 平均 16.393 tokens；
   - motif identity 平均 12.823 tokens；
   - 外层分子边界 2 tokens。
3. 即使每个 motif identity 理想化为一个 token，保留无损 graph 后 M 均值仍为 25.589，56.205% 的记录仍长于 A。因而继续增加整 motif token 不能解决主要长度差。
4. 当前 2,150 个 train-only macro 覆盖 97.402% motif occurrence；剩余仅 2.598% occurrence 的 fallback 却占 45.335% identity tokens。长尾值得压缩，但不应以无界扩张整 motif 词表解决。
5. 一个只由 train fallback 拟合的 64-merge lossless byte-BPE 长度探针，把 M 均值降到 29.551、P95 从 55 降到 46、最大值从 103 降到 79；效果几乎等于把 whole-motif K 从 2,150 扩到 4,096，但只增加 64 个候选 merge token。它仍只把 M>A 从 71.256% 降到 70.911%，所以是长尾优化候选，不是 partition 问题的替代答案。
6. 当前不修改 motif 划分。先完成唯一的 M0-v1/M0-v2 G-Codec Gate；v2 若在质量上通过，再考虑 `v2 + small train-only lossless fallback BPE`。只有下游或结构级 probe 表明当前粒度本身失败，才比较更粗 partition。

## 2. 33,600 条精确长度分解

正式分析器直接读取已发布 paired LMDB，使用 macro registry 或 GraphPorts UTF-8 fallback 无损恢复每个 motif identity；dev 不参与 macro 排名。每条同时复核 identity digest、span 长度、当前 K surface 和 v2 `4+2E` 公式。

| 域 | mean | median | P95 | P99 | max |
|---|---:|---:|---:|---:|---:|
| AtomSELFIES input | 23.321 | 24 | 30 | 32 | 40 |
| motif identity | 12.823 | 8 | 40 | 52 | 82 |
| GraphPorts v2 graph | 16.393 | 16 | 26 | 30 | 34 |
| 完整 M input | 31.215 | 28 | 55 | 70 | 103 |
| atom count | 14.125 | 15 | 17 | 18 | 20 |
| motif count | 7.196 | 7 | 12 | 14 | 16 |
| cross-motif edges | 6.196 | 6 | 11 | 13 | 15 |

contracted graph 的 implied component 数均值为 1.000149、最大为 2；除极少量断连分子外，`E=M-1`。因此在当前 partition 下，每多一个 motif，通常也会多一条 graph edge，即至少增加一个 identity surface 和两个 endpoint tokens。

### 2.1 四个诊断下界

这些是反事实长度分解，不是可训练表示：

| 诊断 | mean M | M>A | 说明 |
|---|---:|---:|---|
| 当前 v2 | 31.215 | 71.256% | 正式 lossless surface |
| 删除固定 4-token graph header | 27.215 | 56.048% | 不可达，只衡量 header 上限 |
| 每 motif 1 token，保留 graph | 25.589 | 56.205% | identity 词表无限理想化后仍常更长 |
| 当前 identity，完全删除 graph | 14.823 | 18.027% | 不可逆，只隔离 graph 成本 |
| 每 motif 1 token且删除 graph | 9.196 | 0% | 同时舍弃词表和拓扑约束的非方案 |

据此可作两个区分：

- **词表问题：**稀有 identity 以 UTF-8 byte fallback 表示，少量 occurrence 产生大量 token。
- **粒度问题：**当前 CAMT5-derived partition 平均产生 7.20 个 motif；forest 拓扑使 graph 成本随 motif 数近似线性增加。

不能只看到 M>A 就把两者混成一次 partition 修改。

### 2.2 哪些记录最受影响

| motif 数 | records | mean A | mean M | M>A |
|---|---:|---:|---:|---:|
| 1 | 377 | 19.817 | 30.019 | 92.84% |
| 2–3 | 2,514 | 21.027 | 29.756 | 65.27% |
| 4–6 | 11,467 | 22.840 | 27.709 | 44.74% |
| 7–10 | 15,045 | 23.836 | 31.576 | 83.91% |
| 11+ | 4,197 | 24.479 | 40.481 | 100% |

单 motif 记录反而也常很长，是因为整个分子 identity 往往是一次性长尾 fallback；高 motif 数记录则由 graph edge 数主导。两端需要不同方法：前者适合 fallback subword，后者只有更粗 partition 或不同拓扑建模才会显著降低长度。

## 3. Whole-motif macro 预算

macro 始终只按 train occurrence 降序、UTF-8 identity 破同分；dev 只用于应用冻结后的字典。结果如下：

| K | occurrence coverage | mean M | P95 | M>A | mean length² |
|---:|---:|---:|---:|---:|---:|
| 0 | 0% | 109.101 | 168 | 100.00% | 13,116.6 |
| 32 | 84.578% | 49.117 | 73 | 99.11% | 2,601.2 |
| 128 | 91.304% | 40.895 | 65 | 92.20% | 1,856.8 |
| 512 | 95.096% | 35.409 | 60 | 81.80% | 1,426.6 |
| 1,024 | 96.289% | 33.368 | 59 | 76.70% | 1,277.7 |
| 2,150（当前） | 97.402% | 31.215 | 55 | 71.26% | 1,120.7 |
| 4,096 | 98.231% | 29.500 | 52 | 66.49% | 1,003.6 |
| 7,580（train 全身份） | 99.709% | 26.208 | 40 | 57.90% | 767.1 |

K 从 2,150 增到 7,580 仍不能把 M>A 降到一半以下，却要增加 5,430 个 whole-motif rows；而且这些数字来自 1% cohort，不能外推为最终全量 K。

参考代码提供了有用的量级对照，但不是直接最优性证明：

- CAMT5 官方 `frag.txt`、`frag_stereo.txt`、`frag_camt5_v3.txt` 分别约 15,114、11,285、22,728 行；它用更大的 fragment vocabulary 换取紧凑 surface。配置实际载入 `frag_camt5_v3.txt`：`reference_repos/CAMT5_official_src_5875a0a/config/task/train/exp/ct_pt_camt5.yaml`。
- FineMolTex 的 graph BPE 默认预算是 500：`reference_repos/FineMolTex_official_src_c976faa/scripts/FineMolTex/datasets/mol_bpe.py`。它支持“有限子结构码本”这一方向，但其表示不是本项目的可逆 T5 graph codec。
- 3D-MolT5 采用 AtomSELFIES，并把 E3FP 放在与 atom carrier 对齐的 side channel；它不需要为 motif graph 支付顺序位置，因此只能作为 atom baseline，不能直接裁决 motif 粒度。

## 4. Lossless fallback byte-BPE 探针

探针保持当前 2,150 个 macro 不变，只在 5,430 个 train fallback identity 的 UTF-8 bytes 上贪心拟合 merge；dev 的 852 个 fallback occurrence 从不参与频率、排名或 merge 学习。基础 byte alphabet 保证可逆，所有长度在相同 33,600 条上重算。

| merge rows | fallback 平均 tokens/occurrence | mean M | P95 | max | M>A |
|---:|---:|---:|---:|---:|---:|
| 0 | 31.092 | 31.215 | 55 | 103 | 71.256% |
| 64 | 22.194 | 29.551 | 46 | 79 | 70.911% |
| 128 | 22.004 | 29.516 | 46 | 79 | 70.851% |
| 250（当前 min-count=2 极限） | 21.922 | 29.501 | 46 | 79 | 70.815% |

64 merges 已获得几乎全部可见收益，继续增到 250 的边际收益很小。若 G-v2 通过，后续最薄候选应固定为：

```text
GraphPorts v2
+ 当前 train-only whole-motif macros
+ 64-row train-only lossless byte-BPE fallback
```

它必须重新经过 tokenizer exact-singleton、save/reload、dev fallback、wire replay 和配对 CE target 门。当前探针没有建立生产 tokenizer，也没有测试模型质量，因此不进入下一轮 G-Codec Gate。

## 5. 是否改变 motif partition

当前回答是“不在现阶段改变”。理由不是回避，而是保持因果问题单一：

1. v1→v2 已在不改变 partition 的情况下把 M 均值降低 37.32%；先判断这种更薄连接语法是否保留拓扑可学习性。
2. fallback BPE 可再隔离长尾 identity 成本。
3. 若做完两者后，11+ motif 桶仍表现出明显吞吐或质量问题，才需要比较更粗粒度；此时对照必须共享 vocab budget、训练 token/member exposure 和下游任务。
4. motif 的研究价值主要是结构级 mask、编辑位置和 atom→motif 3D pooling，不是保证 token 数少于 SELFIES。即使 M 较长，只要在这些任务上获得稳定收益，也可能是合理成本。

后续 partition 比较只需要一个有逻辑的候选，不做无边界枚举：

- 当前 CAMT5-derived partition；
- 一个预先定义的更粗化版本，只合并最常造成链式 singleton motif 的模式，同时保留 ring/non-single-bond、ports 与 strict round-trip；
- AtomSELFIES baseline。

是否启动该比较由 G-v2、motif editing/structure mask 和下游 probe 决定，而不是由长度单指标决定。

## 6. GPU profiler 已准备

新增独立诊断 `most_t5_next/p1/profile_pf1_gpu_pipeline_v1.py`，固定复用 G-Codec Gate 的命名协议 `graphports-codec-screen-64x2-v1`、validated record cache、union-init 和 wrapper，只运行默认 3 个 warmup + 10 个计时 update。历史 PF-1 的 `32×4` 仍以 `pf1-screen-32x4-v1` 保留，checkpoint 可继续按原数字合同恢复；正式 runner 只允许在这两个完整命名协议间选择，不开放任意 batch/LR 参数。它不保存 checkpoint，所有参数更新都丢弃，也不进入 G-Codec 科学 gate。

每个 update 同时记录：

- prepared-data queue wait；
- Python tensor adapter/H2D wall time与 CUDA event；
- forward wall/CUDA time；
- finite-loss `.item()` 同步；
- backward wall/CUDA time；
- gradient clip wall；
- AdamWScale optimizer wall/CUDA time；
- members、encoder/target tokens、throughput、gradient norm、allocated/reserved peak VRAM。

wall 与 CUDA event 不可机械相加；clip 可能等待前序 backward，optimizer wall 则有意包含参考 AdamWScale 中的 host synchronization。profiler 的目的正是区分三种情况：

1. data wait 仍高：再调整 cache/prefetch/worker；
2. data wait 低、optimizer wall 高：另开 optimizer 数值等价实验；
3. forward/backward 本身主导：GPU 锯齿主要是微批计算与同步边界，不能靠增加 CPU worker解决。

## 7. 下一次单卡执行顺序

一张 RTX 4090 足够，不需要为了这个阶段开 4/8 卡：

1. 对 M0-v1 运行一次 13-update profiler；
2. 不根据 profiler 临时修改 optimizer，按已冻结合同顺序运行 M0-v1 与 M0-v2 各 1,000 updates；
3. 依据文档 51 的预注册质量、吞吐和显存门裁决唯一 codec；
4. profiler 若确认 optimizer host-sync 为主要空洞，等 codec 冻结后再做独立等价性 canary；
5. v2 通过后，才决定是否物化 64-merge fallback v3；不同时改变 partition。

当前无卡门：新增分析器与 profiler 的定向测试、完整 P1 回归均通过；完整 P1 为 141/141 PASS。下一轮只需恢复 nmb1 的 1×4090，不需要重新准备数据或下载资源。

## 8. 证据与可复现入口

- 正式长度/词表分析器：`most_t5_next/p1/analyze_pf1_motif_length_budget_v1.py`
- 正式全量报告：`tmp/pf1_motif_length_budget_run3_v5.json`（v1–v4 为迭代过程，不再引用）
- 探索性 fallback 脚本：`tmp/probe_pf1_fallback_bpe.py`
- 探索性 fallback 报告：`tmp/pf1_fallback_bpe_probe_run3_v1.json`
- GPU 分段 profiler：`most_t5_next/p1/profile_pf1_gpu_pipeline_v1.py`
- G-Codec Gate 与阈值：`docs/remote_project_analysis/51_GraphPorts_v1_v2_paired_gate_and_GPU_utilization_plan_20260808.md`
