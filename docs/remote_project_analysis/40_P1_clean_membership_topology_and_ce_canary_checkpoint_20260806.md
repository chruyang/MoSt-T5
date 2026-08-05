# P1 clean membership、拓扑回放与 4090 CE canary 检查点

日期：2026-08-06  
远端：`connect.nmb1.seetacloud.com:36874`  
配置：RTX 4090 24GB ×1、16 vCPU、120GB RAM、30GB 系统盘  
裁决：`TOPOLOGY-CANARY=PASS`，`STANDARD-T5-CE-PLUMBING=PASS`；可以进入真实 BoundRecord/batch/model canary，但尚不能记为 `P1_ADMISSION=PASS`。

## 1. 本轮真正解决了什么

本轮没有启动正式预训练，也没有把审计逻辑塞进模型热路径，而是完成了三个相互独立的基础证据：

1. 从当前可见的 ChEBI-20、PubChem captioning 和 QM9 validation/test 身份构造 connectivity 保护并集，派生 P1/P2 的临时 clean membership；
2. 从官方 PCQM4Mv2 SDF 前缀重放 RDKit 传输、氢投影和 motif linearizer，证明冻结的 32+256 样本与 production-v2 的 motif 分组和 lexeme 绑定一致；
3. 用本地 T5 v1.1 base snapshot 在 RTX 4090 上完成标准 CE 的 forward、backward、一次 AdamW step 和 checkpoint 保存重载。

因此，当前已经证明“数据能按预定边界进入后续训练管线”和“标准 T5 CE 训练接口可执行”，但尚未证明真实 motif token、E3FP 几何融合、teacher/MSE 或下游效果成立。

## 2. 代码组织裁决

3D-MolT5 只用于参考工程组织风格：薄入口、Dataset/Collator、`BatchEncoding`、显式 forward 参数和独立训练脚本。没有复制其任务内容、模型融合方式、magic token ID 或实验结论。

当前建议的数据流是：

```text
production-v2 payload + official SDF topology replay
                         │
protected identity union ├──> permitted member filter
                         │
                         └──> validated BoundRecord
                                  │
                         stateless motif-identity mask
                                  │
                            CEFirstExample
                                  │
                           PaddedCEBatch
                       ┌──────────┴──────────┐
                       │                     │
             model channel             audit channel
        input_ids/attention_mask/labels  record_id + hashes
                       │                     │
                standard T5 forward     不进入 forward
```

拓扑、身份、哈希和 release 检查属于离线工件层；训练热路径只接收模型真正需要的 tensor。后续 E3FP/geometry 应通过独立的 model-specific batch 接口进入 C1，而不是扩张标准 CE 三键接口。

## 3. 下游身份与临时 clean membership

### 3.1 已提取的真实身份

| 数据 | train | validation | test | 当前定位 |
|---|---:|---:|---:|---|
| ChEBI-20 text-to-molecule | 26,407 | 3,301 | 3,300 | 真实结构/文本身份 |
| PubChem captioning | 12,000 | 1,000 | 2,000 | 固定 legacy `description` target |
| QM9 property | 347,774 | 1,928 | 1,928 | connectivity 诊断；validation/test 源文件字节相同 |

QM9 train 的 347,774 条提取 `PASS` 只证明源可读且身份可复现。validation/test 字节相同，不能被解释为两个统计独立评测集；不同 collection ID 和前缀仅保留诊断地址。

### 3.2 为什么 exact SMILES 不够

canonical connectivity 比 exact SMILES 揭示了大量此前不可见的等价结构重合。例如：

- P1 对 ChEBI validation/test 分别有 249/271 个 connectivity 重合；
- P1 对 caption validation/test 分别有 85/140 个 connectivity 重合；
- P2 对 ChEBI validation/test 分别有 2,610/2,587 个 connectivity 重合；
- P2 对 caption validation/test 分别有 372/723 个 connectivity 重合。

因此最终预训练排除键使用 canonical connectivity；stereo 只报告，不作为当前硬排除键；文本身份也不参与结构 membership 排除。

P1 与 P2 的跨阶段重合采用 `replay_permitted`：第二阶段本来就允许在同一分子上学习跨模态对齐，不能把这种有意重访错误标记为下游泄漏。

### 3.3 当前三任务保护并集的派生结果

| 阶段 | 原始成员 | 排除成员 | 许可成员 | 排除 unique connectivity | 排除率 |
|---|---:|---:|---:|---:|---:|
| P1 PCQM4Mv2 | 3,365,577 | 2,735 | 3,362,842 | 893 | 0.0813% |
| P2 PubChem | 301,655 | 12,021 | 289,634 | 5,974 | 3.9850% |

两者均满足 `原始成员 = 许可成员 + 排除成员`。同一 connectivity 可以对应多个训练记录或多个受保护 collection，因此成员数大于 unique connectivity 数是正常结果。

这两份 membership 可命名为 `provisional clean-v0`，足以用于 loader/filter/forward-backward canary 和短程探索；不能用于最终论文权重或“全部下游任务无泄漏”的表述。最终保护并集仍需纳入有效 QM9 split、受控 motif editing，以及对 PubChemQC property 是否保留的明确裁决。

## 4. 真实 topology canary

### 4.1 阈值修正是数据驱动的

第一版选择器预设 `model_atom_count >= 35`，在 production-v2 shard 0 上真实失败，因为 24,893 个 admitted 样本的观测上界仅为 18。随后冻结的 shard-0 census 为：

- model atoms：min 6、p50 15、p90/p95 17、p99/max 18；
- motif count：min 1、p50 6、p90 10、p95 11、p99 13、max 16；
- max motif size：min 1、p50 6、p90/p95 11、p99 14、max 18。

因此高原子边界被固定为 `model_atom_count >= 18`，quota 仍为 40。18 是该冻结 census 的观测边界，不是运行时动态分位数，也不会随数据重新选择而漂移。旧失败没有产生可复用 selection 文件。

### 4.2 冻结选择与运行结果

- selection：32 条 smoke + 256 条 canary，互不重合；
- selection SHA-256：`fc7d796cac5fddfa4efdf79e7475e0817949313eb69387250195a75b1ec3b40d`；
- 与 P1 `provisional clean-v0` 排除 ledger 的成员交集：0/288；
- runner：288/288 输出，0 failed；
- 覆盖：model atoms 9–18、logical motifs 1–14；
- attachment profile：`attachment_and_core=239`、`attachment_only=5`、`no_attachment=44`；
- production release 只读；坐标未重算；E3FP 未重算；
- 只扫描到最大选中 ordinal 24,980，共读取 SDF 前缀 24,981 条，没有展开 9.7GB 全量 SDF。

该结果支持：冻结样本上，官方 SDF 重放的 linearizer、原子映射、motif 分组和 lexeme binding 与 production-v2 一致。它不支持把结论外推到全部 136 shards，也不证明 tokenizer/fallback、mask、CE labels 或 E3FP 几何有效。

## 5. 标准 T5 CE 的 4090 运行证据

### 5.1 薄适配层

`training_adapter.py` 只执行：

```text
PaddedCEBatch
  -> BatchEncoding(
       input_ids: torch.long,
       attention_mask: torch.long,
       labels: torch.long
     )
  -> allowlist forward
```

`record_ids`、审计 JSON 和 artifact SHA 不会进入模型。此接口是 C0/standard-CE plumbing baseline；未来 motif-E3FP 融合必须另建模型专用输入，不得把几何字段伪装成标准三键。

### 5.2 首次严格判据与修正

第一次真实运行已经完成 forward/backward/AdamW，但用 `rtol=1e-5, atol=1e-6` 比较两个独立实例的 logits 时，最大绝对差为 `2.19345e-4`，因此旧报告判为失败。

没有直接放宽阈值。进一步验证发现：

- 284 个 state tensors 的 key、shape、dtype 和数值逐张量完全相同，最大权重差为 0；
- 两个全新 reload 实例在 CPU 上输出逐位相同；
- 优化后原实例与 reload 实例存在 219 个 tensor stride/layout 差异；
- 原实例连续两次 CUDA eval 的最大差为 0。

因此旧失败是“序列化正确性”和“不同执行布局的浮点功能一致性”混用了同一判据，并非 checkpoint 权重损坏。报告 schema 升为 v2，最终判据拆分为：

1. 去除加载地址等运行时字段后，functional config hash 必须完全相同；
2. 所有 state tensors 必须逐张量完全相同；
3. CPU 功能输出在显式 `rtol=1e-4, atol=5e-4` 内一致；
4. reload 后目标设备 logits 必须有限；
5. CUDA 同实例和跨 reload 差异继续报告，但不单独充当权重损坏判据。

### 5.3 最终运行结果

| 指标 | 结果 |
|---|---:|
| device | NVIDIA GeForce RTX 4090 |
| PyTorch / Transformers | 2.1.0+cu118 / 4.45.2 |
| forward keys | `input_ids, attention_mask, labels` |
| batch shape | input 2×3、labels 2×3 |
| loss | 18.0196723938，finite |
| gradient tensors | 282，全部 finite |
| gradient norm | 294.8462000032，非零 |
| AdamW steps | 1 |
| functional config | exact |
| state tensors | 284/284 exact，最大差 0 |
| CPU logits max abs diff | 0.0002193451，within tolerance |
| CUDA reload logits | finite，within tolerance |
| final report | `pass=true` |

这是标准 T5 CE 管线证据，不是 motif/3D 模型证据，也不能说明 loss 会下降、模型会收敛或下游指标会提高。

## 6. 当前证据边界

### 已证明

- 当前三类可见下游 validation/test connectivity 可形成可复现的临时保护并集；
- P1/P2 membership 能无复制 payload 地派生许可与排除成员；
- 冻结 288 条上，拓扑 replay 与 production-v2 记录一致；
- 标准 T5 三键 CE 能在 4090 完成一次前向、反向、优化和模型保存重载；
- 本地相关 contract/codec/collator/topology/overlap 回归 85/85 通过，`compileall` 通过。

### 尚未证明

- QM9 当前 split 是有效独立 benchmark；
- 所有最终下游任务已经纳入保护并集；
- 真实 tokenizer/vocab 与 T5 embedding/output vocabulary 已完成 handshake；
- production record 已贯通到真实 BoundRecord、真实 collator 和真实训练 batch；
- motif anchor、fallback 和掩码在真实 288 条上均无误；
- E3FP 对目标几何状态有增益，MSE/teacher 合理，或任何论文主结果成立。

## 7. 下一阶段最小主线

1. 增加一个薄入口 `collate_bound_records(records, epoch) -> (PaddedCEBatch, audit_sidecar)`，显式传入 epoch；
2. 冻结 tokenizer release，做 tokenizer/model handshake：所有非 `-100` ID 必须落入模型词表，pad/eos/sentinel 与 release 一致；若扩词表，先 resize 并保存绑定后的 snapshot；
3. 把本轮 288 条 topology selection 变成真实 BoundRecord：逐条检查 token round-trip、motif-token/atom/anchor 映射、fallback、mask 和非空 labels；
4. 在 256 条上运行真实 collator + T5 forward/backward，并在 32 条上做短程重复过拟合，要求 CE 明显下降且 checkpoint 恢复一致；
5. 并行修复最终下游协议：有效 QM9 split、caption/ChEBI 任务内 connectivity 交叉、controlled editing source，以及 PubChemQC property 的保留/移出裁决；
6. 最终保护集合冻结后重新生成 P1/P2 permit manifests，再进行 clean motif census、词表冻结和正式 PF-1。

teacher/MSE 暂不进入当前标准 CE canary。先让 C0 和真实 motif batch 成立，再以独立 C1 几何接口验证 E3FP，最后才有条件讨论 C3/teacher，避免把补丁堆进未验证的主干。

## 8. 远端持久化证据

- downstream identities：`/root/autodl-fs/most-t5-r1/reports/p1-vnext-20260806/downstream-identities-v1-20260805T182041Z`
- provisional clean-v0：`/root/autodl-fs/most-t5-r1/reports/p1-vnext-20260806/clean-membership-three-task-v1-20260805T182041Z`
- topology canary：`/root/autodl-fs/most-t5-r1/reports/p1-vnext-20260806/p1-topology-canary-v1-20260806`
- CE smoke v2：`/root/autodl-fs/most-t5-r1/reports/p1-vnext-20260806/p1-ce-smoke-v2-20260806`

关键发布文件 SHA-256：

- topology manifest：`5d94761481855020a4bca268dd0cadbf2da0f74bb266ab6277d3f746477d5e5d`
- topology rows：`6509909259a38d482244a31260b67a3b498049a0469d6c33ce3ebd67e36541de`
- CE report：`425ba3f7eb5162354491230690038a24729012475bbb375d2523b023ea5dc1d8`
- P1 clean manifest：`d48781ba33d7719aa5d7c28151e72a3ce3559188db810c30499b1e7d9b33d3e9`
- P2 clean manifest：`02f30b9a77ec158269b094aa051a5e517b0472e883b69e78ec3e1a579af5a5d2`

CE 发布 bundle 仅 159KB，保留代码、最终报告和首次严格容差诊断；没有把约 1GB 的一次性 smoke checkpoint 复制到 `autodl-fs`。原始 checkpoint 暂留在 `autodl-tmp`，避免为一个不具备科学权重价值的 plumbing probe 长期占用文件存储。
