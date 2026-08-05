# R1 P1 PCQM4Mv2 几何侧车：v2 数据契约

## 结论边界

本目录的 v2 侧车是 P0 的**有界、不可训练**数据流验证物，不是 P1 数据集、
不是 tokenizer 产物，也不授权任何训练启动器。它只允许确定性的 SDF 前缀
（`1..1000` 条），并且保留历史 v1 smoke/replay 作为历史证据，绝不覆盖或升级其结论。

在以下门全部通过之前，禁止执行全量 census、冻结词表、P1/P2 训练或报告任何
“3D-MolT5 processed corpus 完全复现”结论：

1. 来源锁与内层 SDF member 锁；
2. v2 builder + builder-linked replay；
3. 不导入生产语义模块的独立 reference audit；
4. 完整 runtime lock；
5. 单独审查的全量 release / reject ledger / downstream-overlap gate。

## 严格数据流

```text
hash-locked outer OGB SDF archive
  + hash-locked exact tar member (name, regular-file, bytes, SHA-256)
  + hash-locked official OGB companion ZIP / CSV / split_dict
    -> SDF ordinal i
    -> split_dict['train'][i] = official CSV row
    -> tagged source_mol with finite conformer 0
    -> minimal explicit-H projection -> geometry_mol + model_to_source map
    -> strict 2D identity gate (CSV is comparison-only)
    -> motif groups + coordinates + E3FP, all from the same geometry_mol
    -> admitted logical record OR one raw-free reject witness
    -> safe LMDB v2 wire payload + membership / payload-index / reject ledger
    -> logical release root + post-report handoff root
```

CSV SMILES may be parsed only to compare identity. 它的 atom order、坐标、E3FP、motif
mapping 均不得进入特征分支。任何坐标重建、构象生成、优化、对齐、后投影加氢、
或由压缩后下标推断 source index 都是禁止操作。

## 来源和记录身份

`pcqm4mv2_source_contract.json` v3 同时锁定：

- 外层 `pcqm4m-v2-train.sdf.tar.gz` 的 canonical remote path、bytes、SHA-256；
- 内层 `pcqm4m-v2-train.sdf` 的 tar member name、regular-file 类型、解压字节数和
  streamed SHA-256；
- OGB companion archive、`data.csv.gz`、`split_dict.pt`、ZIP member CRC/size/content，
  以及支持 manifest；
- `3,378,606` 个 official train-3D ordinals。

每个 selected ordinal 的 source address 不是“分子哈希”，而是固定 preimage 的 SHA-256：

```text
archive SHA + locked tar-member name/SHA + SDF ordinal + official CSV row
```

因此 v2 分离三种概念：

| 字段 | 含义 |
| --- | --- |
| `source_address_sha256` | 每条必有的来源地址绑定。 |
| `source_mol_identity_sha256` | SDF RDKit Mol 的结构/坐标身份；仅 `source_mol is None` 时为 `null`。 |
| `geometry_mol_identity_sha256` | H 投影后的 geometry Mol 身份；投影前失败时为 `null`。 |

## v2 安全持久化

历史 v1 LMDB 的 `pickle` 只保留为历史 smoke 证据；v2 builder、replay 与未来
reference audit **不得**提供 pickle fallback。

每个 v2 LMDB value 使用：

```text
MST5PCQM2\0 | big-endian header length | canonical JSON header | C-order raw array blocks
```

`sidecar_v2_codec.py` 只允许 `int32`、`float32`、`bool` 的 C-contiguous arrays，验证
magic、长度上限、canonical JSON、重复 key、descriptor、连续 offset、array SHA-256、
未引用 block 与 logical record SHA-256。`payload_index.jsonl` 把每个 storage key 的
wire bytes/SHA 与 logical record SHA 一对一绑定。

输出目录必须同时具有：

- `membership.jsonl`：每个 selected ordinal 恰一行；
- `reject_ledger.jsonl`：membership 中 reject 子集的一对一 ledger；
- `payload_index.jsonl`：admitted membership 与 LMDB key 的一对一 wire index；
- `smoke_scope_manifest.json`：来源、契约、harness、E3FP 和无训练边界；
- `build_report.json`：计数、各 artifact hash、logical release root；
- `release_root.json`：在 build report 写完后生成，绑定 build report 与核心 artifact。

后者避免把一个文件的 SHA 写回自身：`build_report` 的 logical root 覆盖数据产物，
`release_root.json` 再覆盖 build report 本身。

## reject ledger v2

reject witness 不得包含 raw/canonical SMILES、CSV 内容或异常原文。它必须包含：

```text
source_address_sha256
source_mol_identity_sha256 | null
geometry_mol_identity_sha256 | null
stage, reason_code, action, diagnostic_code, detail_sha256
```

`diagnostic_code` 是闭集 token；`detail_sha256` 的唯一 preimage 为 canonical JSON：

```json
{
  "diagnostic_code": "...",
  "reason_code": "...",
  "source_address_sha256": "...",
  "stage": "..."
}
```

不能以 exception 文本、路径、SMILES 或“source_record_identity”这种混合字段替代它。

## replay 的证据等级

`validate_p1_pcqm_geometry_sidecar.py` v2 的报告必须始终声明：

```text
validation_class = builder_linked_deterministic_replay
independent_semantic_validation = false
p1_training_admission = false
```

它验证 v2 wire payload、payload index、成员/ledger 分区、release roots，并重新流式
读取同一 SDF 前缀后调用生产 builder 重算。该验证很重要，但因为它复用 builder、
linearizer、preflight、identity gate 和 producer codec，不能代替独立 reference audit。

独立 audit 将自行实现 source-map/H projection、身份、motif、E3FP shell matrix、
safe decoder 和 reject witness；它只可共享来源完整性边界。

## 无卡 P0 执行约束

当前远端 P0 不训练，也不做全量扫描。命令必须以如下环境运行：

```bash
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0
```

可接受的工作仅为小样本流式验证、哈希、契约检查和报告写入。远端资源调整、runtime
lock 与 fresh v2 128/1000 运行应在单独确认后继续；停止实例前安全检查点是本地代码、
静态 contracts/locks 已冻结且未启动远端 build/replay。

## P1 未来 batch 约束（未在 v2 sidecar 中实现）

P1 的 tokenizer-bound record 必须是新版本，且明确分离：

```text
joint_mask_positions
geo_only_mask_positions
geometry_input_mask  = joint_mask_positions OR geo_only_mask_positions
geometry_target_mask = geo_only_mask_positions AND token_geometry_valid_mask
```

MSE 只对有效 `geometry_target_mask` 位置计算；当批次没有有效目标时跳过 MSE，绝不
用零向量伪造目标。CE 仍按 T5 label/联合 mask 定义。该未来约束不授权将当前 P0
geometry-only record 接入 P1 loss。
