# 执行启动与远端运行时检查点

时间：2026-08-06 00:44–00:50 CST  
远端：`connect.nmb1.seetacloud.com:36874`  
状态：连接与只读检查 PASS；阶段 A–C 已开始；没有启动正式 P1 或 GPU 训练。

## 1. 实际资源配额

宿主机工具显示 128 logical CPUs / 1TiB RAM，但容器 cgroup 才是有效配额：

| 项目 | 实测 |
|---|---|
| `cpu.max` | `1600000 100000`，即 16 vCPU quota |
| `cpuset.cpus.effective` | `0-127`，但仍受 16 CPU quota 限制 |
| `memory.max` | `128849018880` bytes，即 120 GiB |
| GPU | NVIDIA GeForce RTX 4090，24,564 MiB |
| GPU 初始状态 | memory 0 MiB，utilization 0%，31°C |
| driver | 580.105.08 |

因此 CPU 使用率 1600% 才代表容器约 16 vCPU 满载，不能按宿主机 128 线程解释。

## 2. 存储边界

| 挂载 | 容量/使用/剩余 | 裁决 |
|---|---|---|
| `/` | 30G / 18G / 13G | 不放大型 scratch |
| `/root/autodl-tmp` | 112G / 93G / 20G | SQLite、sort、compact summary；必须控制峰值 |
| `/root/autodl-fs` → `/autodl-fs/data` | 200G / 153G / 48G | 持久化合同、报告和最终小型 bundle；避免高并发随机 I/O |

production-v2 release 约 33G，CPU-P0 reports 约 3.9G，legacy/downstream evidence 约 12G，P2 PubChem evidence 约 8.7G。禁止复制 production release 或为不同词表候选重复生成大型数据副本。

## 3. 进程与环境

- 未发现 semantic、identity、overlap、tokenizer 或训练任务；GPU 空闲；
- 仅有平台 supervisord、Jupyter、TensorBoard、autopanel 和本次只读 SSH；
- `3dmolt5` 环境：Python 3.8.20、PyTorch 2.1.0+cu118、RDKit 2024.03.5、Transformers 4.45.2、LMDB 1.7.5、PyArrow available；
- 该环境没有 E3FP package，但当前 A–C 不重算 E3FP，既有 payload 已冻结；
- remote legacy `/root/autodl-tmp/MoSt-T5` 是大量修改/未跟踪文件的 dirty worktree，不允许覆盖或用作新主线；
- 新候选继续以本地 `most_t5_next/` 为源，之后部署到独立 remote staging/bundle。

## 4. 已确认仍在的权威资产

- production-v2 release：3,365,577 admitted、13,029 rejected、136 shards；
- P1 identity rows：约 1.35GB；
- P2 geometry-ready identity rows：约 113MB；
- P1/P2 overlap facts、P2 motif census、semantic r5_2 COMPLETED；
- Google T5 v1.1 base snapshot；
- QM9、ChEBI-20、PubChem property/captioning 的现有 processed/HF/LMDB split；
- P1/P2 identity extractor/proof、tokenizer/auditor contract 工具。

不重做：SDF 扫描、linearizer、E3FP、P1/P2 identity extraction、P2 motif census、semantic replay。

## 5. 已核对的下游源形态

| 任务 | 当前可用 source | validation/test 结构身份获取方式 |
|---|---|---|
| QM9 property | HF Parquet validation/test，各 1,928 rows，含 `smiles` | 直接按 Parquet 行抽取 canonical connectivity/stereo identity |
| ChEBI-20 text-to-molecule | HF Parquet validation 3,301 / test 3,300，含 `cid`、`smiles` | 直接抽取 |
| PubChem 3D captioning | split-specific `3d-pubchem.lmdb`，valid 1,000 / test 2,000，含 `cid`、`smiles`、description | 直接按 split LMDB 抽取 |
| controlled motif editing | 尚未冻结唯一公开 source/split | 仍是 admission blocker，不能事后选择测试身份 |
| retrieval | 次级诊断，尚未冻结 | 不进入第一轮 P1 admission 的主任务集合，除非后续升级为论文任务 |

## 6. 本轮已启动的三条并行支线

1. downstream registry/identity protection contract 与 config；
2. logical-motif、CE-first 的 admission/record/collator contract vNext；
3. hybrid codec、BoundRecord 与 synthetic invariant tests。

合流顺序仍为：

```text
downstream protected identities + P1/P2 policy
→ clean permitted membership
→ clean motif census / codec length gate
→ tokenizer freeze
→ real BoundRecord integration
→ candidate release
→ 1×4090 PF-CANARY
```

