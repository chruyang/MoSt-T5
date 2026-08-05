# 06 后续最小成本验证计划

> 历史计划说明：本文件保留早期验证路线。2026-08-05 起，motif identity hybrid codec、EMA 3D teacher、下游任务冻结与 P1/P2 顺序以 [35_unified_motif_identity_codec_ema_teacher_and_integrated_roadmap_20260805.md](35_unified_motif_identity_codec_ema_teacher_and_integrated_roadmap_20260805.md) 为当前总计划。

## 目标

在不立即重跑昂贵多卡训练的前提下，判断：

1. 历史 checkpoint 是否还能可信使用。
2. tokenizer 与 atom mapping 是否正确。
3. 局部 3D 融合是否有真实增益。
4. 哪些改进值得进入正式训练。

## Stage 0：冻结证据

只读收集或本地复制：

- 远端关键源码。
- `git diff`。
- 20K/25K vocabulary。
- Phase 1/2 config、trainer state、training args。
- tokenizer 相关 notebook/log。
- 下游评估日志。

输出：一个带 SHA-256 的 snapshot manifest。

停止条件：若无法确认源码版本，不进行历史结果归因。

## Stage 1：修复并测试 tokenizer

新增测试：

1. 两个独立 Python 进程产生相同 mapping hash。
2. 两个 DDP rank 产生相同 mapping hash。
3. 20K tokenizer 保存后重新加载，所有 ID 不变。
4. 25K tokenizer 的前 20K motif ID 与 Phase 1 完全一致。
5. 100 个随机 SMILES encode/decode round trip 统计成功率。

输出：确定性 tokenizer v1。

## Stage 2：历史 checkpoint 审计

对 Phase 1/2 checkpoint 做：

- 固定 100–1000 个分子的 motif reconstruction。
- 不同进程重复加载一致性。
- `<unk>`、validity、exact match。
- gate、3D embedding、geometric head 激活统计。
- 随机打乱 E3FP 与 atom mapping 的敏感性测试。

判定：

- 若输出依赖随机 tokenizer mapping，历史 checkpoint 不用于后续结论。
- 若能恢复稳定 mapping 且结构任务明显高于随机基线，可保留为预实验模型。

## Stage 3：数据质量审计

只读扫描小样本和元数据：

- LMDB 总记录数与 key 连续性。
- 必要字段缺失率。
- motif `<unk>` 率。
- E3FP 全 padding 率。
- atom mapping 覆盖率。
- SMILES/分子/scaffold 重复与 split 泄漏。

输出：dataset quality report。

## Stage 4：小规模可行性矩阵

固定同一数据、参数量、步数和 seed，比较：

| 实验 | Motif | E3FP | 局部 mapping | 3D loss |
|---|---:|---:|---:|---:|
| B0 | 否，SMILES | 否 | 否 | 否 |
| B1 | 是 | 否 | 否 | 否 |
| B2 | 是 | 是，全局池化 | 否 | 否 |
| M1 | 是 | 是 | 是 | 否 |
| M2 | 是 | 是 | 是 | 是，固定 target |
| M3 | 是 | 是 | 是 | 是，EMA target |

每项先跑一个 seed 做 smoke test；只有无异常且有方向性增益后，再跑 3 seeds。

## Stage 5：下游最小评估

优先选择：

- BBBP/BACE：scaffold-sensitive 分类。
- ESOL/Lipophilicity：回归。
- QM9 中一个几何敏感性质。
- ChEBI text2mol 小子集。

统一：

- scaffold split。
- 3 seeds。
- 相同微调预算。
- 同时报告均值、标准差和失败率。

## Stage 6：再决定正式训练

进入正式大规模训练的门槛：

- tokenizer 全部确定性测试通过。
- 数据泄漏检查通过。
- M1/M2/M3 至少一个在两个任务上稳定优于 B1。
- 结果能从配置和 checkpoint manifest 完整复现。

## 推荐的近期执行顺序

1. 保存远端源码和差异快照。
2. 修复 tokenizer 并添加 mapping hash 测试。
3. 用当前 Phase 1 checkpoint 做 100 个分子的恢复性审计。
4. 统计 LMDB 的 mapping/E3FP/unknown 质量。
5. 再决定历史 checkpoint 是继续利用还是重新进行小规模 Phase 1。

这一路线优先产生信息，避免在基础映射尚不可信时继续消耗多卡算力。
