# R1 PCQM 几何语义重算与 worker IPC 精度边界（2026-08-05）

状态：`semantic_recompute_gate=PASS`；`P1_ADMISSION=false`。本结论只证明 production-v2 几何侧车能够按真实生产数据流被独立重放，不等同于 E3FP 科学合理性验证，也不代表已满足 P1 的全部准入条件。

## 1. 审查对象与结论

- production-v2 release：`pcqm-geometry-production-v2-20260805T0430Z`
- 总记录：3,378,606
- admitted：3,365,577
- rejected：13,029
- shards：136
- release manifest SHA-256：`4db380c63b00f2a595e3a86f70f434a059a6ca724fe35dee94ae2bdafb7d5a2d`
- 语义审查计划：13,573 条，其中 admitted 544 条、reject 13,029 条
- r5.2 重算结果：13,573 条通过，0 条失败
- 终态分类：12,978 条 `PCQM_STEREO_2D3D_DIVERGENCE`，33 条 `PCQM_SDF_CSV_CONNECTIVITY_MISMATCH`，18 条 `HYDROGEN_PROJECTION_RESIDUAL_H`，544 条 `strict_isomeric_match`
- 触及全部 136 个 release shard
- 完整运行耗时：286.61 秒；外层 SSH 命令墙钟时间 289.3 秒

## 2. r5.1 失败为何是有效发现

r5.1 在修正 SDF 单记录解析方式后完整运行，但 13,573 条全部未通过。失败目录被保留，且没有生成 `COMPLETED`：

`/root/autodl-fs/most-t5-r1/reports/cpu-p0-20260805/semantic-recompute-v1-r5_1-destination-20260805T091508Z`

它正确区分出了 reject 的终态分类，却在 source identity、geometry identity 或 admitted payload 比较上产生系统性不一致。该现象最终定位为审查器漏掉了生产数据流中的 RDKit worker IPC 往返，而不是 production release 被破坏。

## 3. production 的真实 IPC 数据流

生产构建器不是直接将主进程解析得到的 `Mol` 用于 worker，而是执行：

```text
ForwardSDMolSupplier record
  -> source_mol
  -> bytes(source_mol.ToBinary())
  -> worker: Chem.Mol(mol_binary)
  -> identity / projection / E3FP / payload
```

在当前锁定 RDKit 运行时中，这一往返会将 conformer 坐标确定性地映射为：

```text
raw float64 -> float32 -> float64
```

抽查记录的最大绝对坐标差约为 `1.17e-7` 至 `2.35e-7`，原子、键及立体信息保持不变。模型 payload 本身存储 float32 坐标，因此这不是额外的模型输入精度损失；但 identity 包含坐标字节时，审查器必须重放这个边界才能与 production 完全一致。

r5.2 只增加了这一条真实生产操作：在单记录 `ForwardSDMolSupplier` 选中记录后，先执行相同的 `ToBinary()`/`Chem.Mol()` 往返，再计算 identity、投影与 E3FP。production builder、release 和原始数据均未修改。

## 4. r5.2 固定证据

证据包：

`/root/autodl-fs/most-t5-r1/cpu-bundles/semantic-gate-v1-r5_2-20260805T094053Z`

- semantic script SHA-256：`d8a9740ee2f37a9d0af2d1dbdaa9325164d57550e0737bcc40f0969d056d33b6`
- semantic contract SHA-256：`05bdd2083d5796b8f6368d0bc9862a170ef04a557e456098682638afa17dd490`
- tests SHA-256：`a5a9a8fefaa17844b4e64b8461fb3471c4290f6803d69e971282ccdd8602f2e8`
- non-raw IPC regression fixture SHA-256：`ac48ea44de1adbbb5a6011c7cdc4c6f5c331855e4880fe829c4d7def248530ef`
- unchanged production builder SHA-256：`c949a4fd010aa8460fcb22a3b72b47ea38a6fe2c9a52bf6740dc1654f37d0f5a`
- 本地主干测试：25/25 PASS
- 4090 端同版本测试：25/25 PASS

最终输出：

`/root/autodl-fs/most-t5-r1/reports/cpu-p0-20260805/semantic-recompute-v1-r5_2-destination-20260805T094454Z`

- report SHA-256：`cc56233c6bd1e99276995ed298149f745a45ccb0d71cff6039e6f92dbf4b2fe5`
- result ledger SHA-256：`450735d53f4c32ee63bdba4e7090737a7018965a3224557084a64f909ef1a355`
- completion receipt SHA-256：`0f25668c3517c6a713a4a3e182857e7c39e087808e64e5e6b1c844d8ddb121ba`
- `COMPLETED` SHA-256：`74d8802d9fde56d70915cdc573a71d1e627d586020a1ac2a5010bf835b18fe77`
- 外部消费者以固定 semantic script SHA 调用 `validate_completed_output`，返回 `overall_gate_status=pass`

报告在权威收据生成前保持 `completion_status=pending_authoritative_receipt`；最终 PASS 只由相互绑定的 staged report、result ledger、completion receipt 和 `COMPLETED` 联合表示。开始源重放前与结束源重放后分别复核完整 release artifacts；未进行无必要的第三次 32.5 GiB 全量哈希。

## 5. 当前能证明与不能证明的内容

已证明：

1. production-v2 的源记录选择、worker IPC 精度边界、identity、氢投影、坐标与 E3FP 工程计算可由独立脚本重放。
2. 13,029 条 reject 的分类与 544 条 admitted 抽样的安全 payload 一致。
3. 审查覆盖全部 release shard，且运行前后 release artifact 哈希不变。
4. 失败运行不会遗留可独立采信的 PASS 标志；成功运行具有报告、台账、收据和完成标记的哈希闭合。

尚未证明：

1. E3FP 是本项目几何表征的最佳或充分选择。
2. E3FP/MSE 辅助目标相对 CE 的科学收益、权重和优化稳定性。
3. P1 与 P2、下游数据之间不存在 identity 泄漏。
4. P1/P2 motif 语义兼容，或正式 tokenizer 已与 Dataset/Collator/模型端到端绑定。
5. P1 训练可以启动。

## 6. 下一步门禁顺序

1. 在 production-v2 safe payload 上生成 PCQM identity collection。
2. 对 P2 geometry-ready 集合及冻结的下游划分执行真实 membership/overlap proof。
3. 执行 P2 全量 motif census 与 P1/P2 projection-domain compatibility audit。
4. 冻结 tokenizer policy，构建正式 tokenizer release，并完成 128-record Dataset/Collator binding smoke。
5. 主干联合审查上述证据后，才决定是否将 `P1_ADMISSION` 改为 true 并进入 GPU forward/backward 验证。

