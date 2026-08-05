# R1 暂停检查点（2026-08-05）

状态：已在无运行中全量任务、无 GPU 占用的边界暂停。CPU 实例已关闭；后续只使用当前 4090 实例，除非用户重新授权其他资源。

## 1. 当前资源边界

- 远端实例：RTX 4090 24 GB × 1，16 vCPU，120 GB RAM。
- 4090 尚未用于模型级任务；本轮仅做只读核验、小文件部署和 CPU-only 单元测试。
- 暂停时没有运行中的 semantic、overlap 或 tokenizer gate 进程。
- 不再连接已关闭的 96 vCPU 实例。

## 2. 已冻结的 production-v2 release

- Release：`/root/autodl-fs/most-t5-r1/runs/cpu-p0-20260805/incoming/pcqm-geometry-production-v2-20260805T0430Z`
- PCQM4Mv2 train-3D 总记录：3,378,606。
- admitted：3,365,577；rejected：13,029；shards：136。
- release manifest SHA-256：`4db380c63b00f2a595e3a86f70f434a059a6ca724fe35dee94ae2bdafb7d5a2d`。
- destination v3 audit report SHA-256：`8875f61c0c2691e1805081ee81228fce2c0a218f92ad3bbc095c5946bf430902`。
- semantic plan SHA-256：`822e9a67f73f0f33caa38a8e015365929e87bf946238faa19810827c2dd58781`。
- release/transfer/structural audit 已通过，但 `P1_ADMISSION=false`。

## 3. Semantic gate 检查点

### 3.1 已完成

- r3 失败诊断完整保留，不得作为 PASS 证据。
- r4 修正了逐块 `MolFromMolBlock` 与生产 `ForwardSDMolSupplier` 的 parser 语义偏差。
- 本地和 4090 端 r4 单元测试均为 11/11 PASS。
- r4 script SHA-256：`ed92f88ae4e4881ce18f3789ee7a2a69675afcd35e6891c98016062966e2bfc2`。
- r4 contract SHA-256：`ebc4e5bb701fe839d58f490e5621af27d975550b6d1f1fd4001bdf8c5a2bc3b3`。
- r4 tests SHA-256：`93298f33272803fd6add81fc23e695e68be434d5a926978a9a6541ea2688db3f`。
- r4 已部署但没有启动全量运行：`/root/autodl-fs/most-t5-r1/cpu-bundles/semantic-gate-v1-r4-20260805T075237Z`。

### 3.2 暂停原因与恢复条件

r4 的正常算法路径未发现确定性错误，但独立审查判定它只能作为诊断候选，不能成为最终 P0 放行证据。恢复时先实现 r5：

1. 外部固定 release manifest、v3 audit、semantic plan、gate script 及三份 contract 的 SHA-256。
2. 按每个 shard manifest 重新核验整个 release 的全部声明 artifact，而不是只检查 544 条 admitted sample。
3. E3FP source closure 在导入前、计算后和最终发布前后重复核验，并确认实际导入模块位于 attested root。
4. 小文件由同一份 bytes 同时完成解析与哈希；大文件在使用前后复哈希，封闭 TOCTOU。
5. 使用 staging + completion receipt/`COMPLETED` 原子发布；所有末检通过前不得留下可独立采信的 PASS。
6. 三份 contract 使用精确 byte-SHA pin，并增加 mutation/integration 负测。
7. 补真实异常 stereo/atropisomer 回归样例，以及 plan 替换、未抽中 LMDB 篡改、E3FP 运行中变化、末检失败等主流程负测。

r5 hardening 尚未开始写入；不得把当前 r4 标记为 r5 ready。

## 4. Overlap proof 检查点

已完成 source extractor、SQLite 精确 set-proof consumer、contract/config/tests 和说明文档。4090 小样本环境 15/15 tests PASS；没有启动全量 identity extraction，因此还没有真实重叠计数。

恢复前需冻结：

1. P1 与 P2 的交集策略。
2. P2 membership 使用 301,655 条 geometry-ready，还是 301,658 条并让 3 个 singleton 仅参与 2D/text。
3. 每个 downstream task 的 train/validation/test、MoleculeNet seed 与 scaffold split 语义。
4. Dataset/Collator 实际消费的文本单元及文本规范化规则。
5. PCQM safe-payload 到共享 identity collection 的专用 extractor。

当前 `P1_ADMISSION=false`、`P2_ADMISSION=false`，不得声称 downstream 零泄漏。

## 5. Tokenizer binding 检查点

builder、独立 validator、contract 和 tests 已完成；本地主干复核 9/9 tests PASS。P1 census 候选事实：

- 441,769 exact lexemes -> 214,554 pure motifs。
- min-count 2：74,576 tokens，occurrence coverage 99.4211%。
- min-count 5：27,258 tokens，coverage 98.9246%。
- min-count 10：14,317 tokens，coverage 98.5807%。
- top-20k：98.7626%；top-50k：99.2178%。
- 与真实 T5 base 32,100 个词项 exact overlap 为 0。
- base snapshot tree SHA-256：`71c3fab438d892230c5aa9eaff5c8054518cafc14382a414850d387723b82f02`。

P2 仅完成 512 条边界样本结构核验，尚未做 301,655 条全量 census。P2 legacy anchor 是局部 attachment label，与 P1 全局 bond-ID 语义不一致；未经 compatibility audit 不得直接合并 P1/P2 motif census。

恢复时先做 P2 全量 census 与 projection-domain compatibility audit，再选择 `p1_only` 或允许的 P1+P2 train union、cutoff/top-K、频率权重和 OOV policy；之后才可构建正式 tokenizer release 和 128-record binding smoke。

## 6. 推荐恢复顺序

1. 实现并重新独立审查 semantic r5。
2. 在 4090 实例以 CPU-only 执行 r5 的全 release 哈希与 13,573 条语义复算；GPU 保持空闲。
3. 补 PCQM identity extractor，冻结 P2/downstream scope，执行真实 overlap proof。
4. 执行 P2 全量 motif census 与 P1/P2 compatibility audit，之后冻结 tokenizer policy。
5. 完成 tokenizer 128-record binding smoke。
6. 主干联合审查全部证据；仅在全部硬门禁通过后进入 GPU Dataset/Collator/forward/backward 候选门禁。

本检查点不更改 production、auditor、原始数据或旧训练代码，也不表示 P1 已准入。
