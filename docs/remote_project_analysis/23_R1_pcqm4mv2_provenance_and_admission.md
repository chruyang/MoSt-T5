# R1 — PCQM4Mv2 来源核验与准入起点

**状态：** PCQM4Mv2 已通过“官方原始归档”和“前 1,000 条流式结构读取”两个
R1 gate；尚未被纳入 P1，也没有解压全量 SDF 或生成训练 LMDB。

## 远端归档是否来自官方？

是。远端文件：

`/root/autodl-fs/most-t5-p0/sources/pcqm4mv2/ogb-pcqm4mv2-train-3d-v1/archive/pcqm4m-v2-train.sdf.tar.gz`

已核验：

| 字段 | 值 |
| --- | --- |
| 官方 URL | `http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2-train.sdf.tar.gz` |
| 压缩包大小 | 1,559,712,928 B |
| MD5 | `fd72bce606e7ddf36c2a832badeec6ab` |
| SHA-256 | `8690db7b573405e3dbd617482fe10092e18be2b86e4d8500c3892d301dd5817c` |
| OGB 官方 train-3D SDF 记录数 | 3,378,606 |

MD5、文件名、URL 和记录数均与 [OGB PCQM4Mv2 官方说明](https://ogb.stanford.edu/docs/lsc/pcqm4mv2/)
一致。该归档只存放在远端共享盘；没有下载或复制到本机。

## 它与 3D-MolT5 使用的数据是否相同？

必须区分两个层次：

| 比较层次 | 结论 | 依据 |
| --- | --- | --- |
| 上游原始来源 | **相同，可确认** | 3D-MolT5 明确说明 1D+3D joint denoising 与 3D→1D 使用 OGB-LSC 的 PCQM4Mv2；本项目归档是 OGB 官方 train-3D SDF。 |
| 最终 processed corpus | **尚不能确认相同** | OGB 原始 SDF 有 3,378,606 条；3D-MolT5 表 10 报告最终使用 3,377,055 条，相差 1,551。作者未发布 PCQM 专用 membership manifest、reject ledger、原始归档 hash 或完整处理流水线。 |

因此，R1 的严谨表述是：

> 从与 3D-MolT5 相同的 OGB PCQM4Mv2 train-3D 原始上游发布独立复建数据。

而不是“完全复刻 3D-MolT5 的 processed PCQM corpus”。3D-MolT5 的论文来源见
[ICLR 2025 论文 §3.4 与附录表 10](https://proceedings.iclr.cc/paper_files/paper/2025/file/3d4dc72d715bd6415d356293079adf3d-Paper-Conference.pdf)。

## R1 已实现并执行的最小 gate

新的隔离实现位于 `most_t5_next/r1/`，原始 MoSt-T5 和 3D-MolT5 代码均未修改：

1. `pcqm_archive_gate.py` 重新计算 archive MD5/SHA-256、字节数并验证 source
   contract。结果：**PASS**。
2. `pcqm_stream_smoke.py` 以 `tarfile` streaming mode 和
   `RDKit ForwardSDMolSupplier` 读取前 1,000 条。结果：**1,000/1,000** 可解析、
   有有限坐标；只输出哈希化样本和计数，没有落盘 SDF、LMDB 或分子 payload。

远端报告：

- `/root/autodl-fs/most-t5-r1/reports/r1-pcqm-source-20260731/archive_gate.json`
- `/root/autodl-fs/most-t5-r1/reports/r1-pcqm-source-20260731/stream_smoke_1000.json`

## 重要实现约束

历史 `3D-MolT5/3d_tokenization/3d_tokenize.py:data_process(sdf_path)` 读取
`SDMolSupplier(sdf_path)[0]`，只适用于单分子 SDF 示例；若把全量 SDF 路径直接
交给它，会只处理第一条记录。R1 必须在原项目之外使用流式 adapter，并让同一个
RDKit `Mol` 生成 canonical SMILES、坐标、E3FP 与 atom-to-motif mapping。

## 下一道门

在任何全量 PCQM adapter 前，先冻结 OGB v2 的官方 SMILES/split companion，逐条
检查 SDF 与官方 SMILES 图关系。OGB 已披露约 46 条图不一致样本（常见 Si）；它们
必须进入 quarantine/reject ledger，不能被静默重排或修复。随后才执行一次完整的
远端流式构建，并写入 membership、identity-exclusion 和 reject ledger。

