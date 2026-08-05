# 当前 CPU / 单卡 / 8×4090 资源切换裁决

日期：2026-08-06  
裁决：**当前继续使用现有 16 vCPU、120GB RAM 的无卡实例；不立即另租 96 vCPU，也不切换 8×4090。**

## 1. 为什么现在不需要新机器

已经完成且无需重算：production-v2 release、E3FP/motif payload、P1/P2 identity collections、P2 motif census、semantic replay、global motif census 和 T5 base snapshot。当前 A–C 关键路径是：

- downstream registry、split 和 protected identity；
- overlap/exclusion proof；
- contract/schema vNext；
- hybrid codec、round-trip、length/fallback census；
- BoundRecord/Collator/C0-G-L 实现与 CPU tests。

这些任务多数受工程判断、顺序 I/O 或一次性扫描限制，不能因 GPU 数量增加而加速；16 vCPU 已足够并行 2–4 个数据集或以 8–12 workers 顺序处理 shard。

活跃 SQLite、sort scratch 和 compact summary 放 `/root/autodl-tmp`，不要放普速 `/root/autodl-fs`；全量 clean scan 只做一次，并产出可复用 summary，禁止为 16k/32k 等每个候选重复扫描 32.5GiB release。

## 2. 96 vCPU 的条件触发

先在当前机器用真实一个 shard 或至少 100k records 做 benchmark。仅当以下条件同时成立才短租 96 vCPU：

1. 剩余必做全量任务投影墙钟超过 6–8 小时，并阻塞关键路径；
2. 当前 16 vCPU 使用率稳定约 1200%–1500%（上限 1600%）；
3. `iowait < 10%–15%`，确认瓶颈是 CPU 而不是磁盘；
4. 工作能拆为至少 32 个独立 shard，并可确定性 merge；
5. 若跨区，传输、环境 attestation 和结果回传后，端到端时间仍能缩短至少一半。

最可能触发该条件的只有“需要 RDKit 重建的全支持域 codec round-trip/atom-renumber 审计”。若 CPU 只有 300%–700% 且 I/O wait 较高，96 核不会显著提速。

## 3. GPU 数量按阶段选择

| 阶段 | 资源 |
|---|---|
| CPU admission / codec / contract / implementation | 0 GPU，现有 16 vCPU |
| PF-CANARY | 1×4090 |
| PF-1 | 1×4090 |
| DDP smoke | PF-1 有 winner 后，在目标节点短测 1/2/4/8 卡 |
| PF-10 | 根据 DDP 实测选择 1/4/8 卡，不预设 8 卡最好 |
| PF-FULL | 若 PF-10 稳定且单卡期限不可接受，再使用已验证的 4/8 卡 |
| 下游微调 | 多卡拆成多个独立单卡 job，通常比单任务 8 卡 DDP 更有效率 |

8×4090 不等于 192GB 统一显存；普通 DDP 下每个 rank 仍只有 24GB。若 p99 sample 在 microbatch=1 下单卡放不下，增加 DDP 卡数不能解决，必须调整 BF16、gradient checkpointing 或长度政策。

## 4. 长租 8×4090 前的硬门槛

必须全部满足：

1. protected identities、clean membership、tokenizer、PF manifests 均已冻结并 hash 一致；
2. BoundRecord/Collator tests 和单卡 PF-CANARY PASS；
3. PF-1 已压缩为最多一个 winner 和一个 nearest control；
4. p99 batch 的单卡峰值显存最好不超过约 21–22GB；
5. 单卡 PF-10 pair 投影超过约 48 GPU-hours，或 PF-FULL pair 超过两周墙钟；
6. 目标节点有本地高速盘，建议至少 32–64 vCPU、128–256GB RAM；
7. 200–500 optimizer-step 的 1/2/4/8 卡 benchmark 中，8 卡相对单卡吞吐至少 5.2×（约 65% scaling efficiency）、median GPU utilization ≥80%，且 sampler、loss、scheduler、checkpoint/resume 全部一致。

若 8 卡只有约 4× 加速，GPU 成本约为单卡的两倍；优先使用 4 卡。AutoDL 官方也提示 RTX 4090 的多机多卡并行效率相对较低，必须实测而不能按卡数线性外推：[GPU 说明](https://agent-server-security.seetacloud.com/docs/gpu/)。

## 5. 成本边界

AutoDL 当前公开页列出的 RTX 4090 参考按量价为约 1.88 元/卡小时，且实例从开机起计费而不是从 GPU kernel 使用起计费；实际价格以租用控制台为准：[公开价格](https://agent-server-security.seetacloud.com/)、[计费规则](https://agent-server-security.seetacloud.com/docs/price/)。按该参考价简单计算，8 卡约 15.04 元/小时、360.96 元/天，因此不能让 8 卡承担数据审计、codec 调试或 DDP 首次排错。

## 6. 当前执行顺序

```text
现有 16 vCPU 无卡机完成 A–C
→ 100k/单 shard CPU benchmark
→ 只有达到 CPU 触发条件才短租 96 vCPU
→ 1×4090 PF-CANARY
→ 1×4090 PF-1
→ 有明确 winner 才短租目标多卡节点做 1/2/4/8 DDP benchmark
→ PF-10 选择实测最经济的卡数
→ PF-10 稳定后决定 PF-FULL 是否长期使用 4/8 卡
```

