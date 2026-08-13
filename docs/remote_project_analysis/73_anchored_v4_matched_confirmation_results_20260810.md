# Anchored V4 matched-state confirmation 结果（2026-08-10）

## 实验合同

本轮只运行三个必要 cell：

1. B2D `l0_l12_mean`；
2. F3D `l0_l12_mean`；
3. F3D `l0_l123_mean`，即 L0 保持独立、L1/L2/L3 取均值。

三格共享 union-init、adapter seed、训练成员、动态 corruption、64 × 2、1,000 updates 和固定 dev。每格发布一个 model-only 最终 checkpoint，并在 update 1000 评估：

- aligned / both；
- aligned / carrier-only；
- aligned / endpoint-only；
- zero / both；
- same anchored-identity matched donor / both。

matched overlay 在同一跨分子 donor 分配下同时生成 E3FP 与 Morgan 反事实。匹配键为 anchored exact motif identity、atom count、canonical-local endpoint-degree pattern；原子按 canonical-local ID 对齐。

覆盖证据：

- 3,360 / 3,360 dev records；
- 22,887 / 24,015 motif occurrences（95.30%）；
- 38,717 / 47,528 atom rows（81.46%）；
- E3FP 实际改变 95,866 个 state slots；
- Morgan 实际改变 79,835 个 state slots；
- donor 均来自另一分子，未匹配 motif 保持 aligned。

## 最终结果

| cell | aligned NLL | accuracy | zero NLL | zero - aligned | matched NLL | matched - aligned |
|---|---:|---:|---:|---:|---:|---:|
| B2D `l0_l12_mean` | **0.55143** | **85.37%** | 1.79550 | +1.24406 | 0.55090 | -0.00054 |
| F3D `l0_l12_mean` | 0.87079 | 78.81% | 1.03028 | +0.15950 | 0.87076 | -0.00003 |
| F3D `l0_l123_mean` | 0.83760 | 78.65% | 1.38573 | +0.54813 | 0.83766 | +0.00006 |

`l0_l123_mean` 相对 `l0_l12_mean` 的 aligned NLL 改善 0.03318（3.81%），并使 zero ablation 的差距从 0.1595 增至 0.5481。L3 因而能够增强状态通道的总体影响，但没有产生可测的 donor-state 对应敏感性。

## carrier / endpoint

| cell | zero | carrier only | endpoint only | both |
|---|---:|---:|---:|---:|
| B2D | 1.79550 | 1.36058 | 1.91208 | **0.55143** |
| F3D `l0_l12_mean` | 1.03028 | 1.07295 | 1.15678 | **0.87079** |
| F3D `l0_l123_mean` | 1.38573 | 1.72385 | 1.51886 | **0.83760** |

两个 F3D cell 中，carrier-only 和 endpoint-only 都不优于 zero，但同时启用明显更好。这证明当前模型学习的是一个分布式联合路由，而不是两个可独立解释的几何预测器。endpoint 不能据此单独宣称有效；也不能删除，因为联合输入确实优于任一单通道。

## 科学裁决

本轮不支持“当前 identity-denoising 已学会对应的3D状态”：

- matched donor 改变了大量 E3FP slots；
- 两个 F3D cell 的 matched - aligned NLL 绝对值仍小于 0.00006；
- B2D 明显优于两个 F3D cell；
- zero 会显著恶化，说明模型使用了状态通道，但无法证明它使用的是对应的三维状态，而非身份、拓扑、尺度或分布先验。

这也不等价于“V4 adapter 或 E3FP 已被否定”。当前目标是恢复 motif identity；same-identity donor 本来就不改变正确身份标签。该目标天然奖励对构象状态不变，并偏爱直接承载二维身份的 Morgan。用它继续筛选 E3FP shell，会把任务偏好误当成3D表征质量。

因此修正上一轮的冻结结论：

- 不再单凭 identity-denoising 冻结 `l0_l12_mean`；
- `l0_l123_mean` 作为合理的 L3 候选保留；
- `l0_l12_mean` 作为更简洁候选保留；
- 可学习 shell attention 和 `l0123_mean` 仍淘汰；
- 正式预训练前，不再增加相同类型的 identity CE shell-screen。

## 下一门：QM9 3D-sensitive property gate

下一轮应固定同一 anchored encoder 和训练预算，在 QM9 train/dev/test split 上比较：

1. B0；
2. B2D `l0_l12_mean`；
3. F3D `l0_l12_mean`；
4. F3D `l0_l123_mean`。

首批指标应至少包含 dipole moment `mu`、electronic spatial extent `R2`、HOMO、LUMO、gap；这些仍受二维身份影响，因此必须报告 F3D 相对 B2D 的配对差异，而不是只看 F3D 的绝对误差。训练集拟合、标准 dev/test、aligned/zero 及 E3FP donor/扰动诊断要分开报告。

只有 F3D 在预注册的3D敏感属性上稳定优于 B2D，且对应状态扰动使性能恶化，才能冻结 shell 方案并进入更大规模预训练。否则应调整训练目标或几何模块，而不是继续扩大 identity-denoising 预算。

## 产物

远端 checkpoint 根目录：

`/autodl-fs/data/most-t5-r1/anchored-v4-confirmation-v1-b0693bb`

本地 manifest、日志、matched overlay manifest 与 smoke report：

`tmp/anchored-v4-confirmation-v1-b0693bb`
