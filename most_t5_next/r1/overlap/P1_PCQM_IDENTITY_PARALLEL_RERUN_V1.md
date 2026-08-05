# P1 PCQM production-v2 identity 并行复跑说明

## 定位

此实现是串行 `extract_pcqm_production_v2_identity_collection_v1.py` 的旁路复跑器，不替换、不修改已运行的串行程序，也不改变 P1/P2 admission 结论。

并行边界固定为完整 shard：一个 worker 独立完成一个 shard 的 manifest、artifact、membership、reject ledger、payload index、LMDB 和安全载荷校验。父进程只接收已验证的 canonical identity rows，随后完成全局闭包、重复 member ID 检查、SQLite `COLLATE BINARY` 排序和最终发布。

## 运行前提

- 等当前串行任务结束后再复跑；不要与正在读同一 release 的旧进程混合操作。
- `--release-root` 仍指向只读 production-v2 release。
- `--scratch-dir` 必须是全新路径，建议位于 `/root/autodl-tmp`；不得位于 release 或 output 内部。
- `--output-dir` 也必须不存在且位于 release 外部。
- 默认 8 个进程；当前 16 vCPU 机器建议先使用默认值。
- 程序不自动递归清理 scratch。成功后保留分片行、SQLite 和 `scratch_manifest.json`，待结果验收后再单独制定清理方案。

## 命令模板

从仓库根目录执行，并把尖括号路径替换成当前机器上的实际路径：

```bash
python -m most_t5_next.r1.overlap.extract_pcqm_production_v2_identity_collection_parallel_v1 \
  --extraction-contract most_t5_next/r1/contracts/pcqm_production_v2_identity_parallel_extraction_contract_v1.json \
  --config most_t5_next/r1/overlap/configs/p1_pcqm_production_v2_identity_parallel_config_20260805.json \
  --release-root <PCQM_PRODUCTION_V2_RELEASE_ROOT> \
  --production-contract <PRODUCTION_CONTRACT_JSON> \
  --payload-contract <PAYLOAD_CONTRACT_JSON> \
  --identity-normalization-contract <IDENTITY_NORMALIZATION_CONTRACT_JSON> \
  --scratch-dir /root/autodl-tmp/p1_identity_parallel_scratch_20260805_v1 \
  --output-dir <NEW_OUTPUT_DIR_OUTSIDE_RELEASE> \
  --processes 8
```

## 验收边界

必须优先比较串行与并行的核心文件：

```bash
sha256sum \
  <SERIAL_OUTPUT>/molecule_identity_rows.jsonl \
  <PARALLEL_OUTPUT>/molecule_identity_rows.jsonl
cmp --silent \
  <SERIAL_OUTPUT>/molecule_identity_rows.jsonl \
  <PARALLEL_OUTPUT>/molecule_identity_rows.jsonl
```

两个文件必须完整字节相等；同时比较两个 collection manifest 中 `molecule_rows` 的 `bytes`、`sha256`、`row_count` 和 `key_lf_sha256`。

receipt 的 `generated_at_utc`、进程数和 scratch 绝对路径属于执行信息，预期不同，不纳入核心字节一致性判定。并行 receipt 仍必须满足：

- `status == "pass"`；
- `p1_training_admission == false`、`p2_training_admission == false`；
- `source_membership_rows` 等于锁定 release 的 membership 总数；
- `rejected_members_filtered` 等于锁定 release 的 reject 总数（当前生产 release 预期为 13,029）；
- `emitted_molecule_rows + rejected_members_filtered == source_membership_rows`；
- shard 数为 136；
- `scratch_manifest` 的 SHA-256 与 scratch 中实际文件一致。

任何 worker 异常、分片缺失/重复/范围不闭合、源文件或目录集合变化、payload 解码失败、reject/payload/LMDB 不同步，都会在 final output 目录创建前失败。失败时 scratch 中可以保留部分诊断文件，但不能视为通过结果。

## 本地验证

并行实现的 hermetic 测试覆盖：

- 串行、1 worker、2 workers 的核心 JSONL 完整字节一致；
- worker 异常时不发布 output；
- 重复 shard 声明失败；
- scratch 必须全新且与 release/output 隔离；
- release 文件哈希在成功运行前后完全一致。

执行：

```bash
python -m unittest \
  most_t5_next.r1.overlap.tests.test_extract_pcqm_production_v2_identity_collection_parallel_v1 \
  -v
```
