# PCQM `STEREO_2D3D_DIVERGENCE` 补算与 stereo-free 主线恢复（2026-08-12）

状态：`classification=PASS`，`supplement_materialization=PASS`，
`strict_production_v2_mutated=false`。

## 1. 问题与旧门禁

PCQM4Mv2 train-3D 共 `3,378,606` 条。旧 production-v2 要求官方 CSV
SMILES 与 SDF 投影分子在 RDKit 2024.03.5 下满足：

```text
canonical isomeric SMILES 完全相等
```

连接图相等而 strict-isomeric 字符串不同的记录统一被标记为
`PCQM_STEREO_2D3D_DIVERGENCE`，共 `12,978` 条（全源的 0.3841%）。该类别
并不等价于“连接错误”；真正的 CSV/SDF connectivity mismatch 只有 33 条。

该旧门禁适合“二维文本与三维几何必须指向同一严格立体异构体”的 release，
但不应无条件约束现行架构：当前模型表面明确采用 stereo-free motif/fragSMILES
作为二维身份，R/S、E/Z 与构象状态由同一 SDF Mol 派生的 E3FP sidecar 承担。

## 2. 远端补算方法

远端 CPU：32 vCPU；锁定 RDKit `2024.03.5`。输入保持不变：

- 原始 PCQM train-3D SDF tar.gz；
- 官方 companion `data.csv.gz`；
- production-v2 的 136 个 `reject_ledger.jsonl`；
- 历史 E3FP 锁定源码。

reject ledger 已保存 `sdf_record_index == official_csv_row_index`，因此无需重建
33 GB production release。分类器只顺序解压一次 SDF，在命中 12,978 个冻结 ordinal
时解析；CSV 同样只顺序扫描一次。逐条继续要求：

1. canonical non-isomeric connectivity 相等；
2. strict-isomeric hash 不等；
3. source/geometry identity 与冻结 reject witness 一致；
4. 比较双方 stereo feature、双向 `useChirality=True` 子结构匹配和 InChIKey。

分类脚本：

`most_t5_next/r1/semantic/classify_pcqm_stereo_2d3d_divergence_v1.py`

远端终态证据：

`/root/autodl-tmp/most-t5-r1/derived/pcqm-stereo-2d3d-divergence-classification-v3`

分类耗时约 80 秒，`12,978/12,978` 闭合。

## 3. 分类结果

| 分类 | 数量 | 解释 |
|---|---:|---|
| official 未声明、SDF 已声明 | 3,276 | 最直接符合 stereo-free 2D identity + SDF state |
| 双方均声明，SDF 是 official 的 stereo refinement | 2,096 | 可在 SDF-authoritative state 政策下恢复 |
| 双方化学等价，仅 strict surface 不稳定 | 1,032 | 双向手性图匹配与完整 InChIKey 均相等，可作规范化恢复 |
| 双方均声明，official 是 SDF 的 refinement | 4,995 | 旧 strict release 继续隔离；stereo-free 主线可使用 SDF state |
| official 已声明、SDF 未声明 | 587 | 旧 strict release 继续隔离；现行主线没有可由 SDF 提供的对应 stereo state |
| 明确冲突或当前 matcher 不支持 | 992 | 旧 strict release 继续隔离；若纳入现行主线必须声明 SDF 是唯一 state 真值 |

保守恢复视图为 `6,404` 条：3,276 + 2,096 + 1,032。其 membership 已冻结为：

`.../pcqm-stereo-2d3d-divergence-classification-v3/recovery_membership.jsonl`

但现行架构的正式身份策略是 connectivity-only 2D identity，且所有几何/E3FP
都从同一个 SDF Mol 派生。因此，最终 supplement 采用更直接且逻辑一致的规则：

> 只要 CSV/SDF connectivity 相等，二维输入移除 stereo，几何状态以 SDF/E3FP
> 为唯一来源；CSV 的 R/S、E/Z 不再作为几何真值或准入门。

这允许重新处理全部 `12,978` 条，同时仍保留旧 strict release 和分类台账，避免
把两种 estimand 混在同一个 schema 中。

## 4. 实际重新处理结果

由于旧 reject 只保存哈希 witness，没有保存坐标/E3FP payload，不能通过简单修改
membership 补回；必须从原始 SDF 重新计算。补全 builder：

`most_t5_next/r1/semantic/build_pcqm_stereo_recovery_supplement_v1.py`

它按 production 的真实 worker IPC 边界执行：

```text
SDF Mol
-> Mol.ToBinary() -> Chem.Mol()（复现冻结的 float32 坐标边界）
-> 显式氢投影
-> coordinates + 四层 E3FP
-> atom-to-source + atom-to-motif
-> motif 摘要
-> wire encode/decode replay
-> 独立 LMDB supplement
```

运行配置为 28 workers / max-pending 84，耗时 `104.19 s`。结果：

- selected：12,978；
- admitted：12,978；
- rejected：0；
- wire payload：65,292,007 bytes；
- LMDB `data.mdb`：98,275,328 bytes；
- atom rows：169,701；
- motifs：60,633。

远端发布位置：

`/root/autodl-tmp/most-t5-r1/derived/pcqm-stereo-recovery-supplement-v2`

全量只读回放逐条核验 membership 顺序、LMDB key、wire SHA-256、logical SHA-256、
坐标/E3FP 形状、四层宽度与 identity policy，结果 `12,978/12,978 PASS`。

## 5. Schema 与后续合并边界

不能把 supplement 原地追加到 production-v2。旧 schema 硬要求：

```text
official_identity_status == strict_isomeric_match
sdf_strict_smiles_sha256 == official_strict_smiles_sha256
```

而这 12,978 条恰好不满足该条件。强行追加会破坏历史 release 的身份合同与独立
审计。因此：

- production-v2 的 136 shards 保持不变；
- supplement 明确记录
  `identity_policy=stereo_free_connectivity_with_sdf_authoritative_state`；
- 下一次物化 stereo-free fragSMILES/E3FP 正式训练缓存时，将主 release 与 supplement
  按 member ID 合并；
- 不得把 supplement 描述成 strict 2D/3D identity 修复；它是面向新架构身份职责的
  重新处理视图；
- 33 条真实 connectivity mismatch 与 18 条残留氢 reject 不在本次恢复范围内。

## 6. 裁决

12,978 条无需继续整体丢弃，且已全部完成几何/E3FP重新处理。旧 strict release 的
隔离判断在其自身合同下仍然正确；现行 stereo-free 2D identity + SDF/E3FP state
架构则使用独立 supplement 恢复这些成员。正式预训练数据重物化时必须显式合并该
supplement，并在最终 manifest 报告基础 release、supplement 和真实不可恢复 reject
三者的成员闭合关系。
