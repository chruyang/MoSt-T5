# R1 — 可复现数据 release 重建

R1 位于 `most_t5_next/`，与历史 MoSt-T5 训练树隔离。它不是 P1/P2
训练阶段，而是为新 P1 提供可复现、可审计且不泄漏的数据 release。

当前只启用两项低风险 gate：

1. `gates/pcqm_archive_gate.py`：核验远端 PCQM4Mv2 原始归档的供应链
   信息、大小和 MD5/SHA-256；不会解压或复制数据。
2. `gates/pcqm_stream_smoke.py`：以 `tarfile` 流式读取归档内 SDF 的前 N
   条，验证 RDKit 可解析、存在有限坐标并记录极小摘要；不会写 LMDB、不会输出
   分子记录，也不会把 SDF 解压到磁盘。

`contracts/pcqm4mv2_source_contract.json` 是候选源契约。它确认该归档与
3D-MolT5 使用的 OGB PCQM4Mv2 具有相同的**上游官方来源**，但明确禁止把它称为
3D-MolT5 的逐样本 processed corpus：OGB 原始 SDF 有 3,378,606 条，而论文报告
其处理后使用 3,377,055 条，且作者未发布该过滤 membership manifest。

R1 后续顺序：

```text
archive lock
  -> stream smoke
  -> OGB SMILES/split companion lock
  -> bounded SDF-to-official-SMILES identity smoke
  -> full single-pass remote adapter + reject ledger
  -> identity exclusion
  -> motif/E3FP policy and membership manifests
  -> frozen tokenizer
  -> P1 admission
```

## PCQM4Mv2 identity smoke

After the official companion is frozen remotely, run
`gates/pcqm_identity_smoke.py`.  It uses the declared OGB relation
`SDF ordinal -> split_dict.pt['train'][ordinal] -> data.csv.gz row`, compares
canonical RDKit graphs for a bounded prefix, and writes only a small JSON
report with aggregate counts and SMILES hashes.  It never extracts the archive
or writes LMDB records.

The comparison uses the locked minimal post-parse projection declared in
`contracts/pcqm4mv2_identity_normalization_contract.json`: only
`RemoveHsParameters.removeDefiningBondStereo=True` differs from RDKit's
defaults.  This removes an explicit-H representation artifact without using a
broad hydrogen, isotope, query, or atom-order rewrite.  The gate distinguishes
two outcomes:

- A connectivity pass permits only the design of one remote full-pass
  identity/reject ledger.
- A strict isomeric mismatch is still recorded and quarantined from the
  primary strict 2D/text/3D aligned release.  `--require-strict-isomeric`
  turns such a warning into a gate failure for a stricter audit.

任何全量 adapter 只能在上述前置门通过后执行一次，必须从同一个 SDF RDKit Mol
同时生成 SMILES、坐标、E3FP 与 atom-to-motif 映射；不得直接把全量 SDF 路径传给
历史 `3D-MolT5/3d_tokenization/3d_tokenize.py:data_process`，该示例只读取
`SDMolSupplier(...)[0]`。
