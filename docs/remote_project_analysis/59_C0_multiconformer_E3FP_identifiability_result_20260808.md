# C0：多构象 E3FP 可辨识性审计结果与 G1 放行裁决

> 状态：2026-08-08 完成。该结果是训练前的机制审计，不是模型效果、下游性能或最终预训练准入结论。

## 1. 要回答的问题

PF-1/PF-2 已证明当前 T5 可以接收 E3FP，但没有证明预测真正依赖几何内容。G1 之前必须先回答一个更基础的问题：在同一 2D identity 下，production E3FP 是否会随 3D 构象改变，并同时保持刚体旋转、平移不变。

如果答案是否定的，继续更换融合器或增加 loss 都没有意义；如果答案为肯定，也只能放行“学习如何使用 E3FP”的下一阶段，不能直接声称模型已学会 3D。

## 2. 冻结协议

- 数据域：PF-1 published train membership；不读取 dev/test 标签，不用于模型选择。
- 抽样：固定 seed `20260808`；单组分、8--25 个重原子，并至少含 2 条“非环、非末端重原子单键”灵活性代理边。
- 构象：ETKDGv3 请求 8 个、RMS prune `0.35 Å`，每个分子最多保留 4 个。
- 3D token：production 氢投影 + `duplicate_pointer_inheritance_v1`，4096 folded IDs，level 0--3。
- motif：使用冻结 `mol_linearizer` partition，仅作 atom-to-motif 统计，不改变 partition。
- RMSD：按冻结 atom-row 对应做 Horn quaternion 刚体对齐；不允许对称原子重排改变 E3FP 行语义。
- 主审计：1,000 个成功分子，ETKDG ensemble 不再做力场松弛。
- 独立复核：同一固定抽样前 32 个分子在 Linux/pinned RDKit 环境做 MMFF94s，参数缺失时回退 UFF。
- 实现：`most_t5_next/p1/audit_c0_multiconformer_e3fp_v1.py`。

本机 pinned Conda 同时加载 PyTorch、RDKit/ETKDG 和 E3FP/Scipy 会引入两份 OpenMP DLL。C0 脚本直接执行，不经 `most_t5_next.p1.__init__` 预加载 PyTorch；没有使用 `KMP_DUPLICATE_LIB_OK` 之类可能静默产生错误的绕过。

## 3. 1,000 分子主结果

产物：`tmp/c0_multiconformer_e3fp_train1000_20260808/`。

| 项目 | 结果 |
|---|---:|
| 接收分子 | 1,000 |
| 生成构象 | 3,833 |
| 构象对 | 5,548 |
| 刚体旋转/平移 exact parity | 126 / 126 |
| level 0 populated-row change | 0 / 80,526 = 0% |
| level 1 populated-row change | 8,210 / 80,526 = 10.195% |
| level 2 populated-row change | 69,155 / 80,526 = 85.879% |
| level 3 populated-row change | 69,772 / 69,886 = 99.837% |
| 任一 level 1--3 改变的 atom-pair rows | 98.856% |
| 任一 level 1--3 改变的 motif-pair rows | 98.894% |
| 完整 level 1--3 相同的构象对 | 0 / 5,548 |
| RMSD≥1 Å 且完整 E3FP 相同 | 0 |
| RMSD 均值 / 范围 | 1.254 Å / 0.439--2.588 Å |
| RMSD 与 atom-change-rate Spearman | 0.135 |

全部 1,000 个分子的保留构象均得到互不相同的 level 1--3 状态。额外 22 个候选只因 ETKDG prune 后少于两个不同构象而拒绝；不是 E3FP、motif 或化学 identity 错误。由于抽样预先多取 1.5 倍候选，正式 1,000 个成功样本的次序仍由冻结 candidate rank 决定。

### 3.1 状态分布

| level | unique folded IDs | entropy (bits) | top-1 rate |
|---|---:|---:|---:|
| 0 | 45 | 4.105 | 15.832% |
| 1 | 2,065 | 8.376 | 7.695% |
| 2 | 4,095 | 11.804 | 0.531% |
| 3 | 3,683 | 11.379 | 0.133% |

level 0 是稳定的局部原子身份层，不应作为“3D 学习成功”的主要证据。level 2/3 接近高熵 4096-way categorical target，标准 CE 仍可计算，但必须报告 level-wise NLL/accuracy，并和各 level 的 unigram prior 比较。

## 4. 力场复核

产物：`tmp/c0_multiconformer_e3fp_mmff32_remote_20260808/`。

| 项目 | ETKDG-only，同一 32 分子 | MMFF94s/UFF |
|---|---:|---:|
| 构象对 | 176 | 176 |
| level 0 change | 0% | 0% |
| level 1 populated change | 9.743% | 5.174% |
| level 2 populated change | 84.366% | 66.465% |
| level 3 populated change | 99.739% | 97.900% |
| 完整 level 1--3 collision | 0 | 2 |
| RMSD≥1 Å collision | 0 | 0 |
| rigid exact parity | 16/16 | 16/16 |

MMFF 的两个 collision 对 RMSD 仅为 `6.79e-6 Å` 与 `0.00166 Å`，说明不同初始构象优化到了同一几何极小值，而不是 E3FP 把明显不同的构象压成同一状态。力场松弛降低了 level 1/2 的变化率，但没有改变 C0 结论。

## 5. 科学解释

### 5.1 可以得出的结论

1. 当前 production E3FP 对同一 2D identity 的构象变化具有充分可辨识性。
2. level 0 满足预期的构象稳定性；3D 差异主要进入 level 1--3，尤其 level 2/3。
3. 刚体旋转和平移不改变 E3FP，生产入口的基本几何不变量成立。
4. G1 可以继续使用 E3FP，不需要现在改成坐标 GNN 或重新生成 PCQM 数据。

### 5.2 不能得出的结论

1. 不能据此声称 PF-1/PF-2 的 T5 已经使用 3D；已有 shuffle 结果仍表明它基本忽略几何。
2. 不能把 folded ID 当连续量。RMSD 与 E3FP atom-change-rate 的相关仅 0.135，E3FP 表示的是离散局部状态，而不是带距离度量的坐标。
3. 本轮没有覆盖对映体专项、实验构象能量排序或跨数据集泛化；这些应作为后续 3D claim 的独立证据，而不是阻断 G1 的前置条件。
4. 不能因为 level 2/3 几乎总变化就直接使用 MSE；target 仍是无序 categorical ID。

## 6. G1 的最小实现合同

### 6.1 先做可学习集合编码，不再做 raw mean

主候选采用 level-aware gated set pooling：

```text
h_a = phi(concat_l(E[id_a,l] + LevelEmbedding[l]), valid_shell_mask)
score_a = w^T tanh(W h_a)
alpha_a = segment_softmax(score_a, motif(a))
g_m = rho(sum_{a in motif m} alpha_a h_a)
```

- shell level 在 atom 内按固定顺序编码；不把 `(atom, shell)` 扁平化成无序集合。
- atom 在 motif 内保持 permutation invariance。
- 一张 shared E3FP table；padding 为零，不重新引入 PF-1 的四表求和。
- `g_m` 只进入一个 motif carrier，不增加 T5 token 长度。
- 标准 Deep Sets `phi -> mean -> rho` 是唯一必需结构基线；Set Transformer/PMA 仅在二者失败后考虑。

### 6.2 先做 standalone masked-state gate

在重新接入完整 T5 前，先训练一个小型 motif-state autoencoder：

1. 只在 populated level 1--3 随机遮蔽 E3FP slot；level 0 只报诊断。
2. atom encoder 读取其余 shell，motif pool 提供集合上下文。
3. 对被遮蔽 slot 做 4096-way CE；分别报告 level 1/2/3 token-weighted NLL、accuracy，并同时比较 train-unigram 与 uniform prior，采用二者中更强者作为静态基线。
4. 对同一分子构象对报告 `g_m`/prediction 的变化；对刚体复制应保持一致。
5. 对齐 E3FP 必须优于 same-size shuffled E3FP。

这样先回答“集合编码器能否学习 E3FP 状态”，再回答“T5 生成是否使用该状态”，避免 T5 的 2D 捷径掩盖几何编码器失败。

### 6.3 放行到 T5 bridge 的条件

- 三个 level 的 dev NLL 均优于各自 `min(train-unigram, uniform)` 静态先验；不得只报总平均。
- aligned state 明显优于 same-size shuffled state。
- 多构象表示变化可重复、刚体变换保持一致。
- gated pooling 至少不劣于 standard Deep Sets；若没有改善，保留更简单的 Deep Sets。

通过后再把 `g_m` 接入冻结 M0 主干，增加 geometry-to-motif generation CE。当前不加入 raw-ID MSE、teacher latent、InfoNCE 或新的 motif partition，以保持因果归因。

## 7. 当前裁决

**C0 PASS，进入 G1 standalone masked-state gate。**

该裁决支持“E3FP 输入具有可学习的构象差异”，但不支持“现有融合已经有效”。下一步 CPU 工作是实现、单测并在 PF-1 train/dev 上准备 Deep Sets 与 level-aware gated pooling 的同协议 state-CE runner；只有短程学习性门需要重新租用一张 RTX 4090。
