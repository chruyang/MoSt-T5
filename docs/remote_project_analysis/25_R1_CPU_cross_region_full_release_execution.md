# R1 CPU 跨区全量 release 执行记录

状态：全量 production release 与跨区传输结构门禁已通过（2026-08-05）；整体 P0 仍为 `PARTIAL`，`P1_ADMISSION=false`。  
范围：PCQM4Mv2 train-3D 几何侧车的数据侧 P0；不授予 P1 训练准入。

## 1. 本轮目标与边界

本轮把 CPU 密集的 SDF 顺序读取、身份核验、motif 线性化、E3FP 计算、几何侧车写入和独立结构审计迁移到 96 vCPU / 180 GiB CPU 实例。原 4090 实例保留，用其跨实例持久化目录接收不可变分片；两地区不共享 `autodl-fs`，因此采用 CPU 实例到原 4090 地区的 SSH/rsync 直传，本地电脑不承载数据流量。

本轮只回答以下问题：

1. 3378606 条 train-3D 源记录能否形成完整、可追溯、可续跑的几何 pretokenizer release；
2. 每条记录的输入、身份、motif、E3FP 和拒绝处置是否闭合；
3. 全量 motif census 能否支持之后冻结 tokenizer，而不重算 E3FP 或重跑 linearizer。

本轮不回答 tokenizer 冻结、数据集重叠、CE+MSE 训练行为或正式 P1 准入。

## 2. 数据与运行时锁

- CPU 实例 cgroup 可见 96 个逻辑 CPU，内存上限 180 GiB；工作盘为 `/root/autodl-tmp`。
- 输入在源地区完成哈希核验后由 CPU 实例直接拉取；canonical 输入合计约 1.71 GB。
- staging receipt：`pcqm_staging_receipt_v1_20260805T0358Z.json`，结果 `pass=true`。
- 最终生产/审计 runtime attestation：`cpu_runtime_attestation_production_audit_v2_20260805T0420Z.json`，结果 `pass=true`。
- E3FP vendored source closure SHA-256：`46818c2b74a962159a1bac8d386e2a75545d3a730fe64ebd6b55b9f536400510`。
- production contract SHA-256：`4193794a6300e78ef6ed4bf66de0ea9f24aea4e9a1ca034d5ec85425404adc2b`。
- production 前使用的独立审计 v2 contract SHA-256：`12043dcc3757469b9e5610768a3ed114755f861435ebba3bb44b40cf4754644c`。
- 最终双端审计使用的 v3 contract SHA-256：`a3a109edcb9ed424c84462cfd51f3cda62e6be607a4af6a7c1f7393828b0ab9d`；v3 进一步绑定审计器字节、运行环境、计划和报告自哈希。

运行环境固定为 CPU-only，并把 `OMP_NUM_THREADS`、`MKL_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`NUMEXPR_NUM_THREADS` 全部固定为 1，防止 64 个进程内再次产生 BLAS/OpenMP 线程过量订阅。

## 3. 全量前发现并修正的问题

### 3.1 逐记录 motif 关系不能丢失

production v1 只累计全局 fragment 频次，未保存每条记录的 motif 序列。若直接全量运行，最终 tokenizer 冻结后无法把每个 motif 绑定 token ID，必须再次扫描 SDF 并重跑 linearizer。

修正后的 production v2：

- 每个 admitted record 在 `topology.motif_lexeme_sha256` 保存与 `motif_atom_indices` 一一对应的有序 digest；
- digest 定义为 `SHA256(exact motif fragment UTF-8 bytes)`，不做额外规范化；
- 每条记录禁止保存 raw motif fragment 或 raw SMILES；
- shard/global census 保存 `{motif_lexeme_sha256, motif_fragment, count}`；
- digest 格式错误、digest 与 fragment 不一致或同 digest 对应不同 fragment 时 fail closed；
- payload logical hash、wire hash 和 membership content hash共同绑定该有序关系。

因此 tokenizer 冻结阶段可执行 `record digest -> global lexeme -> frozen token ID`，无需重算 E3FP 或 motif 划分。

### 3.2 runtime attestation 必须核对当前文件

原结构验证器只检查 attestation JSON 内部自洽。production v2 额外重新哈希 attestation 中列出的当前 bundle 文件，并要求 runner、builder、linearizer、codec、staging/runtime gates 及全部关键合同都在锁内；任一现场文件变化都会阻止运行。

### 3.3 真实 LMDB 接口差异

第一次 128 条真实写入在 `Environment.sync(force=True)` 停止；当前 `python-lmdb` C 扩展不接受关键字参数。修正为 `Environment.sync(True)` 后重新采集代码/运行时证明，并使用新输出目录重跑。失败的 partial attempt 被保留，没有覆盖、重命名为成功产物或删除。

## 4. 全量前门禁结果

### 4.1 128 条确定性门禁

- 单进程：128 admit / 0 reject，`benchmark_non_release`；
- 16 进程：128 admit / 0 reject，`benchmark_non_release`；
- 两次使用相同 release ID；membership、payload index、global motif census 和 LMDB `data.mdb` 均字节级一致；
- 独立审计：全部 128 个 membership/LMDB key 闭合，4 条预注册分层样本由独立 wire decoder 解码并通过。

### 4.2 10000 条并发与确定性门禁

| workers | 端到端墙钟时间 | 含输入复核吞吐 | admit | reject |
|---:|---:|---:|---:|---:|
| 16 | 19.4 s | 约 515 records/s | 9942 | 58 |
| 32 | 12.2 s | 约 820 records/s | 9942 | 58 |
| 48 | 10.7 s | 约 935 records/s | 9942 | 58 |
| 64 | 9.9 s | 约 1010 records/s | 9942 | 58 |

四种并发的 membership、reject ledger、payload index、global motif census 和 LMDB `data.mdb` 五类核心文件哈希全部一致。58 个 reject 均为 `PCQM_STEREO_2D3D_DIVERGENCE`，即严格立体身份不一致但 connectivity 一致；它们未被伪装为有效几何目标，并全部进入独立语义复核计划。

10000 条产物实际占用约 100.9 MB。线性外推全量约 34.1 GB；启动前 CPU 快速盘可用约 51.6 GB。全量过程中仍需持续监测非线性尺寸增长。

独立 10000 条审计通过：9942 个 admitted LMDB key、58 个 reject 和 10000 个 membership 完整闭合；global census 等于 shard aggregation；4 条固定分层样本的有序 motif digest 均可解析到全局字典。

## 5. 全量执行参数与回传

全量 release ID：`pcqm-geometry-production-v2-20260805T0430Z`。

- source range：`[0,3378606)`；
- workers：64；
- max pending：192；
- shard size：25000；
- LMDB map upper bound：1024 MiB/shard（稀疏映射，不等于实际占用）；
- resume granularity：仅 completed shard boundary；
- partial policy：保留失败 attempt，新 attempt 不复用旧目录；
- raw SDF extraction：禁止，直接流式读取经核验的 tar.gz member；
- LMDB merge：禁止。

CPU 工作目录：

`/root/autodl-tmp/most-t5-r1/runs/pcqm-geometry-production-v2-20260805T0430Z`

跨区持久化目标：

`/root/autodl-fs/most-t5-r1/runs/cpu-p0-20260805/incoming/pcqm-geometry-production-v2-20260805T0430Z`

镜像任务每 15 秒执行 append/copy-only rsync，明确排除 `shard-*.partial-attempt-*`，不使用 `--delete`、`--remove-source-files` 或任何源端删除。最终必须在持久化目标再次执行独立审计，不能把 rsync 成功等同于 release 审计成功。

### 5.1 实际运行结果

- 开始：`2026-08-05 12:26:27 +08:00`；完成：`2026-08-05 13:12:54 +08:00`；主体墙钟时间约 `46 min 27 s`；
- source / membership：`3378606 / 3378606`；
- admitted：`3365577`；rejected：`13029`；二者严格相加等于 membership；
- shard：`136`，覆盖 `[0,3378606)`，无 gap、无 overlap；
- unique motif：`441769`；motif occurrence：`24180228`；
- release apparent bytes：`34918330652`，约 `32.52 GiB`；
- final manifest SHA-256：`4db380c63b00f2a595e3a86f70f434a059a6ca724fe35dee94ae2bdafb7d5a2d`；
- logical release root SHA-256：`898f7400843ac1a99013da31d75e73854da18a5c3b4309af348f839854deb412`；
- `range_no_gap_no_overlap=true`、`lmdb_merged=false`、`tokenizer_binding=absent_and_forbidden`、`p1_training_admission=false`。

## 6. 完成条件

本轮 production-release 子门禁的完成条件及结果如下：

1. 136 个 shard 无 gap/overlap 覆盖全部 3378606 个 ordinal；
2. `membership = admitted + rejected = 3378606`；
3. 每个 admitted 记录恰有一个 payload-index row 和一个 LMDB key；
4. 每个 rejected 记录恰有一个 closed reason ledger row；
5. shard/global motif census 内容寻址、聚合和碰撞检查通过；
6. E3FP 参数 hash 全局唯一；
7. CPU 工作盘独立审计通过；
8. 跨区持久化副本独立审计通过；
9. 完整 manifest、audit report、semantic review plan 和运行时/输入 receipts 返回持久化目录。

以上 1--9 项均已满足，因此可判：

```text
DATA_RELEASE_STRUCTURE = PASS
CROSS_REGION_TRANSFER = PASS
INDEPENDENT_STRUCTURAL_AND_SAMPLED_PAYLOAD_AUDIT = PASS
OVERALL_P0 = PARTIAL
P1_ADMISSION = FALSE
```

## 7. 双端独立审计证据

审计器 v3 SHA-256：`230bcae05849e8d92624c1ca59ea58be738b25af686f491afef424b7033597d7`。它不导入 producer、生产 codec 或项目 E3FP 实现；对完整 release envelope、全部工件哈希、membership/reject/payload-index 分区、全部 LMDB key、motif census 聚合与内容地址执行全量检查，并用独立 decoder 检查预注册分层样本。

| 证据 | CPU 快盘源端 | 4090 区域持久化副本 |
|---|---:|---:|
| audit status | `pass` | `pass` |
| audit elapsed | `238.6 s` | `442.5 s` |
| release manifest SHA-256 | `4db380...d5a2d` | `4db380...d5a2d` |
| membership | `3378606` | `3378606` |
| admitted / LMDB / payload-index | `3365577` | `3365577` |
| reject ledger / scheduled review | `13029` | `13029` |
| sampled admitted payload | `544` | `544` |
| shard | `136` | `136` |
| audit runtime SHA-256 | `ee02b176...c0622f` | `dcb8462f...fd358b` |
| report canonical payload SHA-256 | `0794c92d...b20db0a` | `ca590f3e...02555c` |

两端 runtime、完整 plan 和 report hash 不同是预期结果：v3 把各机内核、Python 路径等运行环境写入 header/report。去除唯一含 runtime provenance 的 plan 首行后，两端其余 `13573 = 544 + 13029` 行字节投影 SHA-256 均为 `dbd39f4a3db158476fd0da05b35438875a5e8ec4fefc213b02bd6dd23c38502e`。这证明两端预注册样本和全部 reject 复核行的内容、顺序及序列化字节一致。

持久化证据位置：

- release：`/root/autodl-fs/most-t5-r1/runs/cpu-p0-20260805/incoming/pcqm-geometry-production-v2-20260805T0430Z`；
- 目标端审计：`/root/autodl-fs/most-t5-r1/reports/cpu-p0-20260805/audit-full-production-v3-destination-20260805T0519Z`；
- 源端审计、runtime/staging receipts 和日志副本：`/root/autodl-fs/most-t5-r1/reports/cpu-p0-20260805/source-host-evidence`；
- 128/10k 确定性产物及相应审计：`/root/autodl-fs/most-t5-r1/reports/cpu-p0-20260805/preflight-artifacts`；
- 本轮后加固代码、合同与本地文档快照：`/root/autodl-fs/most-t5-r1/cpu-bundles/post-p0-hardening-20260805T0535Z`。

## 8. 本轮不能推出的结论与下一门禁

当前证据不能推出：

1. `13029` 个 reject 的化学语义已由独立 RDKit/E3FP 全部复算；当前只证明它们被一致识别、关闭并列入复核计划；
2. motif 划分和 E3FP 状态在科学意义上已被文献或外部参考实现充分证实；
3. 下游数据重叠已排除；
4. tokenizer 已冻结并绑定；
5. CE+MSE 的 Dataset/Collator/model 数据流与梯度行为已通过；
6. 可以启动正式 P1。

因此下一顺序仍是：独立化学语义复核策略裁决 -> P1/P2 训练集合边界与 overlap proof -> tokenizer 冻结和 digest 绑定 -> 4090 上 128 条 Dataset/Collator/forward/backward/save-reload 候选门禁。

## 9. 本次运行后发现的发布竞态与后续修复

本次运行使用的已证明副本在写 `full_release_manifest.json` 时是直接 exclusive-create 写入，旧镜像脚本也未在退出前比较目标端 manifest 哈希。为避免把半写 manifest 或文件列表竞态带入下一次运行，本地后续版已增加：

- 同目录临时文件完整写入、`flush`/`fsync` 后，以 `os.link` 原子 no-replace 发布 immutable JSON；
- 镜像退出前要求 rsync 前后源 manifest SHA-256 稳定，且目标端 SHA-256 与源端一致；
- runner 原子发布回归 `11/11` 通过，Linux fake-transport 镜像竞态回归 `3/3` 通过，`bash -n` 通过。

这些修改发生在本次 release 完成后，没有追溯性改变本次已锁定的 production bytes。本次 release 通过“生产进程完全退出 -> 强制最终 rsync -> 目标 manifest 对比 -> 目标端全量独立审计”闭合该风险。
