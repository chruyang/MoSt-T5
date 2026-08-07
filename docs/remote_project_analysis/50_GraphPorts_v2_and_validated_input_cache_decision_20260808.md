# GraphPorts v2 与一次验证训练缓存裁决

> 日期：2026-08-08  
> 状态：GraphPorts endpoint-pair v2 的 CPU 规范门通过；PF-1 run3 的 33,600 条严格解码缓存门通过；尚未进行 v1/v2 效果短筛，也尚未用 GPU profiler 判断优化器同步占比  
> 研究边界：本轮只改变表示冗余和数据供给方式，不改变 motif partition、样本、mask、CE 目标、模型参数或优化器数值

## 1. 结论

1. `train_members % microbatch == 0` 是项目上层人为限制，不是 PyTorch/T5/LMDB 的要求；现已采用标准 `drop_last=False` 可变尾批语义。
2. 当前 motif 输入较长的主因不是 motif identity，而是 v1 的五 token/edge 外部连接表。不能因长度先改 motif 划分，否则会同时改变研究粒度与序列化成本。
3. GraphPorts v2 只删除冗余 edge id 与 A/B marker，复用 v1 的全部化学层。33,600/33,600 条规范连接解码通过，平均完整 M 长度从 49.80 降至 31.22。
4. v2 仍有 71.26% 的分子长于 AtomSELFIES；因此它是更薄的 lossless codec 候选，不是“motif 一定更短”的证据。v1/v2 必须做唯一一个配对短筛后再定主线。
5. GPU 锯齿的首要数据问题是每个 epoch 重复严格 wire 解码，而不是 LMDB 磁盘读取。本机完整实测表明：4 worker 一次预热 33,600 条耗时 21.82 s；此后按冻结顺序回放全部 train+dev 仅 0.049 s。
6. 4 worker 是当前合理起点：公平基准中 4 worker 为顺序解码的 2.79×，8 worker 为 3.17×，多占一倍 CPU 只再提高约 14%。不以“16 vCPU 必须吃满”为优化目标。

## 2. GraphPorts endpoint-pair v2

### 2.1 改了什么

v1 对每条跨 motif 边输出：

```text
EDGE_ENDPOINT_A, edge_id, endpoint_a, endpoint_b, EDGE_ENDPOINT_B
```

v2 输出两个按规范顺序排列、可自定界的 endpoint uvarint。connection id 由边在规范序列中的位置隐式恢复。motif identity、port、component、connection metadata、R/S、E/Z、radical、charge、isotope 和 strict-isomeric reconstruction 均继续由 v1 化学 codec 负责。

一般图流长度为：

\[
L_G^{v2}=3+\ell(\mathrm{port\_radix})+
\sum_e\left[\ell(p_{e,a})+\ell(p_{e,b})\right].
\]

run3 中 port radix 和所有 packed endpoint 都只有一个 byte，所以该数据域精确简化为：

\[
L_G^{v2}=4+2E.
\]

不能把 `4+2E` 写成任意规模 motif 图的一般定理。

### 2.2 owner 与 mask 合同

- 每个 endpoint 的首 byte 是唯一 `connection` carrier，映射到 owning logical motif；
- continuation bytes 是 `boundary/-1`；
- 每条边恰有两个 connection carriers；
- `connection_token_indices` 是 owner/carrier indices，不是完整 endpoint mask span；
- 当前 identity masking 不遮蔽 graph stream，因此接口兼容；未来若做 topology masking，必须按完整 endpoint span 遮蔽，不能只遮首 byte。

### 2.3 小型与全量证据

- v1+v2 远端 RDKit 2024.03.5：43/43 tests PASS；
- 覆盖零边、断连组件、tree、cycle/nonforest、同 motif pair 多边、多字节 endpoint、端口复用拒绝、R/S、E/Z、charge/isotope 和 atom renumber；
- run3 全量 33,600 条：
  - decoded connection equality：33,600；
  - general varint length gate：33,600；
  - exact two owner carriers/edge：33,600；
  - tokenizer 所需 261 tokens 全部复用冻结 snapshot，0 个新增 ID；
  - maximum packed endpoint=92，maximum port radix=9；
  - 416,388 个 endpoint 全为单 byte；
  - 33,600 个 contracted graphs 全为 forest。

全量审计从已发布且完整 source-bound 验证的 run3 wire documents 出发，证明 v2 是连接表的规范重序列化；本轮没有第二次读取 SDF 重跑完整化学链，因此不把它表述为新的全量 source-SDF chemistry replay。

证据：

- `tmp/pf1_graph_ports_v2_full_audit_run3_20260808.json`
- `tmp/audit_pf1_graph_ports_v2_run3.py`
- `most_t5_next/r1/tokenizer/production_graph_ports_codec_v2.py`
- `most_t5_next/r1/tokenizer/tests/test_production_graph_ports_codec_v2.py`

### 2.4 长度结果

| 指标 | v1 M | v2 M | AtomSELFIES |
|---|---:|---:|---:|
| mean | 49.804 | 31.215 | 23.321 |
| median | 49 | 28 | 24 |
| P95 | 79 | 55 | 30 |
| P99 | 91 | 70 | 32 |
| max | 137 | 103 | 40 |
| M>A | 96.59% | 71.26% | — |
| mean M/A | 2.183 | 1.370 | — |

v2 对完整 M surface 的均值降幅为 37.32%；单独 graph stream 从均值 34.98 降至 16.39，降幅 53.14%。这支持“先压缩 graph serialization”，不支持“为了长度立刻重新划 motif”。

CAMT5 以 DFS/anchor 在 fragment 内隐式表达连接，说明连接无需机械展开为冗长边表；FineMolTex 保留 atom graph、motif graph 与 atom-to-motif map，支持 motif 作为语义位置，但不是可逆 T5 线性化先例；3D-MolT5 把 E3FP 作为 SELFIES 对齐 side channel，不增加序列位置，其论文也报告直接拼接 E3FP 序列会明显增加成本。[CAMT5](https://aclanthology.org/2025.findings-emnlp.1221/)、[FineMolTex](https://doi.org/10.1145/3711896.3736834)、[3D-MolT5](https://proceedings.iclr.cc/paper_files/paper/2025/file/3d4dc72d715bd6415d356293079adf3d-Paper-Conference.pdf)。Group SELFIES 与 SPE 可作为后续语义 group/subword 设计参考，但不能替代本项目的 port 和立体化学回环门。[Group SELFIES](https://pubs.rsc.org/doi/d3dd00012e)、[SPE](https://pubs.acs.org/doi/10.1021/acs.jcim.0c01127)。

## 3. 训练供数不是“把 prefetch depth 调大”

### 3.1 原热路径

每个 epoch 都执行：LMDB get → UTF-8 JSON decode → canonical JSON 重编码/摘要 → 完整 dataclass/语义 gate → Python collate → 直接构造 CUDA tensor。旧 prefetch depth=2 只有一个 Python producer thread；它能重叠少量工作，但不能并行 Python 严格解码。

PyTorch 的成熟路径是多进程 worker、预取、persistent workers，以及在 CPU tensor 形成后才使用 pinned memory/non-blocking H2D；Python 密集型工作不能依靠增加 OMP/BLAS 线程自动并行。[PyTorch DataLoader](https://docs.pytorch.org/docs/stable/data.html)、[performance tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)、[pinned memory guide](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)。3D-MolT5 与 CAMT5 的官方代码也都在预处理/DataLoader 使用多 worker；这证明方法常见，不证明 8 一定是本项目最优值。[3D-MolT5 repository](https://github.com/QizhiPei/3D-MolT5)、[CAMT5 repository](https://github.com/Songhyeontae/CAMT5)。

### 3.2 公平 worker 基准

本机 16 logical CPU、同一 2,048 条完整 wire record；每个并行分支都返回完整不可变 record 到主进程并核对顺序：

| 模式 | records/s | 相对顺序解码 |
|---|---:|---:|
| sequential strict decode | 519 | 1.00× |
| 4 threads | 623 | 1.20× |
| 2 processes | 916 | 1.77× |
| 4 processes | 1,450 | 2.79× |
| 8 processes | 1,647 | 3.17× |

内存追踪独立于吞吐计时，估算全 33,600 条缓存约 0.64 GiB。4 threads 收益小而 4 processes 已接近平台，符合 GIL 与多进程 DataLoader 的预期。

证据：`tmp/pf1_decode_cache_worker_benchmark_local_20260808.json`。

### 3.3 正式缓存边界与全量结果

当前 1% screen 采用最薄优化：

1. 主进程顺序读取 LMDB raw value；
2. 4 个 `spawn` CPU workers 严格 wire decode；
3. 结果按冻结顺序返回；
4. 主进程重新检查 membership join 后写入 process-local dict；
5. mask 与 collate 仍按真实 epoch 在线生成；
6. 缓存不序列化、不成为新数据发布、不进入 checkpoint；
7. worker 使用 `spawn`，不会继承训练主进程 CUDA context。

训练 manifest 将缓存信息拆成两层：`data.validated_record_cache` 只保存四条件必须相同的静态合同；每个进程不同的 hits、misses、预热 wall time 和吞吐只写入对应 condition 的 `input_pipeline_telemetry`。这样四个单条件进程可以严格比较共同科学合同，同时保留各自运行时证据，不会因计时差异而错误拒绝合并。

全量实际结果：

- 33,600 entries，strict decode misses=33,600；
- 4 workers / max pending 16；
- warmup 21.82 s，1,539.8 records/s；
- train 30,240 + dev 3,360 冻结顺序回放 0.049 s；
- hits=33,600，0 reject，cache complete=true；
- 训练语义、record order、mask RNG、resume cursor 均未改变。

证据：`tmp/pf1_validated_record_cache_full_audit_local_20260808.json`、`tmp/run_pf1_cache_warmup_audit.py`。远端缓存/runner/merge 定向回归为 35/35 PASS，完整 P1 包为 123/123 PASS。

## 4. GPU 锯齿仍需区分两个来源

缓存能去掉数据重复验证，但不能先验保证 GPU 始终满载。当前训练还有独立同步点：

- 每 microbatch 的 loss `.item()`；
- gradient clip norm 回传 CPU；
- AdamWScale 对大量参数逐个执行 parameter RMS `.item()`。

3D-MolT5/CAMT5 参考优化器也有类似 RMS 写法，说明它有来源，但不代表吞吐最优。不能在同一次实验中同时改 loader 和 optimizer，否则会改变数值轨迹且无法归因。

下一次 GPU 的固定顺序：

1. 先用当前优化器 + validated cache 跑 50–100 update timing canary；
2. 记录 data wait、tensor/H2D、forward/backward、clip、optimizer 的同步 wall time和 GPU trace；
3. 若 data wait 已接近零而 GPU 仍显著锯齿，再单独设计 GPU-resident/foreach AdamWScale 等价性实验；
4. optimizer 数值实现一旦改变，所有参与效果比较的架构必须从同一初始化重新训练。

若下一轮使用 4 张卡，当前 runner 采用“一进程一条件”，不是 DDP。必须给 A0/A1/M0/M1 四个进程分别设置 `CUDA_VISIBLE_DEVICES=0/1/2/3`；每个进程内部仍只看到自己的 `cuda:0`。四进程各启用 4 个 decode workers，总计 16 个，正好对应当前 16 vCPU；不要在四卡模式下再把每进程 worker 提高到 8。

## 5. 后续实验拆分

### A. codec 短筛

只比较 GraphPorts v1/v2：同一 motif partition、macro/fallback、成员顺序、mask、初始化、优化器、molecule exposure。报告 strict decode/reconstruction、NLL/accuracy、molecules/s、non-padding tokens/s、显存和 wall time。若 v2 效果下降，优先判断删除显式 endpoint marker 是否降低拓扑可学习性，而不是立即重划 motif。

### B. F-Gate

继续使用当前 v1 codec，独立测试几何融合机制。不得把 v2 长度变化与 F-Gate 融合变化放进同一条件。

### C. 正式全量训练

33,600 条 process-local cache 只适合当前 1% failure screen。扩展到 311 万条时应使用 map-style Dataset、每 worker 独立延迟打开只读 LMDB、CPU tensor/pin/non-blocking H2D 和有界预取；不能把 0.64 GiB 小样本缓存机械放大到全量。
