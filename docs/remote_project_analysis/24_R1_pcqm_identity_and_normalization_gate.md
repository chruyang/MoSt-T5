# R1：PCQM4Mv2 身份关联与氢规范化 Gate

日期：2026-07-31（远端 `autodl-fs` 证据）  
状态：**图身份 smoke 已通过；PCQM4Mv2 仍只是 P1 候选源，未获训练准入。**

## 结论

冻结的 OGB PCQM4Mv2 train-3D SDF 与官方 `data.csv.gz` / `split_dict.pt` 的关联可以继续按

```text
SDF ordinal i -> split_dict['train'][i] -> data.csv.gz 的零基 row i
```

处理。本次前 1,000 条出现的 29 个“连接图不一致”不是源文件或排序错误，而是比较器用 RDKit 默认
`RemoveHs` 后仍保留 stereo-defining 显式 H、而官方 CSV SMILES 将该 H 隐式表示的假阳性。

我们已把比较规则固定为最小改动：解析 SDF 时保留显式 H，随后仅设置
`RemoveHsParameters.removeDefiningBondStereo=True`。该规则在远端 RDKit `2024.03.5` 上将前
1,000 条结果修正为：

| 比较层级 | 条数 | 处理结论 |
| --- | ---: | --- |
| 严格 isomeric canonical SMILES 一致 | 998 | 可进入后续完整 release 审计 |
| 仅 stereo/isomeric 表示差异 | 2 | 默认隔离，不能混入 primary strict 2D/text/3D release |
| non-isomeric connectivity 不一致 | 0 | 图身份 gate 通过 |

这只允许下一步设计一次**远端单流式**完整 identity/reject-ledger pass；不代表 PCQM4Mv2 已进入 P1，更不代表复现了 3D-MolT5 未公开的 processed corpus。

## 证据链

1. OGB 官方页面说明 train-3D SDF 含 3,378,606 条训练分子，公开 MD5 为
   `fd72bce606e7ddf36c2a832badeec6ab`；同时明确约 46 条训练分子的 SDF/SMILES 2D 图可能不一致，且没有原子到原子的对应关系。[OGB PCQM4Mv2](https://ogb.stanford.edu/docs/lsc/pcqm4mv2/)
2. 远端归档已按官方 MD5、字节数和 SHA-256 锁定；官方 companion zip 中的 `data.csv.gz` 与
   `split_dict.pt` 也已锁定。当前版本中 `train` 连续为 `0..3,378,605`，但 release 仍必须以
   `split_dict['train'][i]` 作为正式关联，而不是靠连续性假设。
3. 旧的 v1 identity report（严格默认 `RemoveHs`）被保留为历史诊断，**不得作为数据拒绝账本**。
   它的 29 条 connectivity 假阳性均在最小规范化下恢复为严格一致。
4. v2 graph gate（1,000 条）报告 SHA-256：
   `9c375b3559c0ff0c6ab1d53c7844311023a08ff80ca2a3626e1bd35fd84b3c60`。
   `--require-strict-isomeric` 控制实验报告 SHA-256：
   `4ce4ab34c9af39e15f6737e9ffc635a9bbcacd17810b0dc237c3a1de3059a507`；它如预期因 2 条 strict 差异而失败。

远端报告目录：

```text
/root/autodl-fs/most-t5-r1/reports/r1-pcqm-source-20260731/
```

## 固定的比较契约

实现与政策分别位于：

- `most_t5_next/r1/gates/pcqm_identity_smoke.py`（v2，远端脚本 SHA-256：`c61c661ac5f0c06e5564c07da8275f38b877d83f3e066ea017427cbfd098bf4e`）
- `most_t5_next/r1/contracts/pcqm4mv2_identity_normalization_contract.json`（SHA-256：`5f9be346294e08bf73d47c089a00be4c2f19d89612b5e4c09d0d7f5f6b23b044`）

不可变规则：

```text
ForwardSDMolSupplier(sanitize=True, removeHs=False)
  -> Chem.RemoveHs(copy_of_mol, params.removeDefiningBondStereo=True)
  -> SanitizeMol + AssignStereochemistry
  -> strict / connectivity canonical-SMILES SHA-256
```

- 不使用“全开”的 RemoveHs 参数；这会把同位素、mapped atom、query 等无关语义一起改写。
- 不手工删除原子，不在相邻 CSV 行中搜索“更匹配”的分子，不静默重排。
- 非严格匹配的记录必须进入不可变 reject/quarantine ledger。默认原因名为
  `PCQM_STEREO_2D3D_DIVERGENCE`；真正连接图不一致为
  `PCQM_SDF_CSV_CONNECTIVITY_MISMATCH`。

## 对实际 P1 数据流的约束

官方 OGB 仅提供**分子级**关系，明确不提供 SDF 与 CSV 之间的原子对应。因此 CSV 只能用于身份核验，绝不能提供 atom index。

对一条准入记录，必须从同一个规范化后的 SDF RDKit `feature_mol` 派生：

```text
feature_mol
  -> canonical 2D / motif decomposition
  -> retained conformer coordinates
  -> E3FP per-atom features
  -> atom_to_motif_map
```

这也与原项目的既有实现约束一致：`E3FPTokenizer.encode` 在已有 conformer 时直接对输入 `Mol` 指纹化；它不会为了该路径重新从 SMILES 生成另一套坐标。R1 adapter 必须保持 feature-mol 原子数、E3FP 行数和 atom-to-motif 索引完全同序，并给每个 motif 写出 `geometry_valid_mask`；不允许将 `-1` padding 或无几何 motif 伪装成零 MSE target。

还必须绕开历史 `process_qc_step2_mapping.py -> linearize(smiles)` 路径：它会先从字符串重新 `MolFromSmiles`，因此无法证明返回的 motif atom index 与 SDF 坐标 / E3FP 行是一套索引。R1 已单独规定 `linearize_mol(feature_mol)`：只接收已规范化的 RDKit Mol、保持原 atom index、并对 motif group、cross-edge 和 DFS 顺序做显式确定性排序。完整 record、membership、reject 和 batch mask 规范见 `most_t5_next/r1/R1_P1_PCQM_adapter_schema.md`。

## R1 后续完整 pass（尚未执行）

1. 先把本契约、RDKit/E3FP 版本、归档和 companion 的 SHA-256 纳入 candidate release manifest。
2. 单次流式遍历压缩 SDF；审计 `SDF ordinal`、SDF title、train position、CSV row 和 CSV `idx` 的关联。
3. 使用固定规范化计算严格/连接图 key，并写出仅含 ID、原因、哈希和统计量的 reject ledger / membership manifest；不解压完整 SDF，不先写 LMDB。
4. 对 strict mismatch、真实 connectivity mismatch、解析/规范化异常分别统计和隔离；不能强行令总数等于 OGB 所说的约 46，也不能强行令总数等于 3D-MolT5 论文中的 3,377,055。
5. 仅在完整 ledger、下游 validation/test 零重叠、motif/E3FP mapping gate、稳定 tokenizer gate 全部通过后，才构建可训练 sidecar release 并由 release-manifest validator 远端复核。

## 当前 R1 决策

| 项目 | 决定 |
| --- | --- |
| OGB train-3D 原始来源 | 锁定，可继续审计 |
| 1k connectivity 身份 gate | 通过 |
| strict 2D/text/3D 对齐 | 未通过（2 条隔离候选） |
| PCQM 全量 release | 未开始 |
| P1 launcher / 训练 | 禁止启动 |
