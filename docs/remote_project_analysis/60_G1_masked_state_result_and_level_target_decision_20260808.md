# G1 masked-state 结果与 E3FP level 目标裁决（2026-08-08）

## 1. 本轮回答的问题

本轮不再用完整 T5 的总 CE 间接判断几何是否有效，而是在冻结的 PF-1
train/dev（30,240/3,360）上单独训练小型 motif-state encoder，直接回答：

1. motif 内的 E3FP 状态能否由置换不变集合编码器学习；
2. learned gated pooling 是否确实优于标准 Deep Sets；
3. E3FP level 1/2/3 是否都适合作为 4096-way categorical reconstruction target。

正式产物：

- Deep Sets：`tmp/g1_motif_state_formal_deep_20260808/manifest.json`；
- gated：`tmp/g1_motif_state_formal_gated_20260808/manifest.json`；
- train-to-dev 条件可预测性审计：`tmp/g1_state_predictability_audit_20260808.json`；
- 修订后的 Deep Sets level 1+2：
  `tmp/g1b_motif_state_l12_deep_v2_20260808/manifest.json`；
- 可读 checkpoint 复跑：`tmp/g1b_motif_state_l12_deep_v3_20260808/manifest.json`；
- aligned/shuffled：`tmp/g1c_aligned_shuffled_l12_v2_20260808.json`。
- 多构象表示：`tmp/g1c_multiconformer_128_fp32_v3_20260808.json`。

这些结果是几何机制门，不是 T5、下游任务或最终预训练性能。

## 2. 同协议 Deep Sets 与 gated 结果

两格使用相同成员、随机种子、15% shell masking、500 updates、batch 1024、
8 workers、64/128 维 embedding/hidden。两格在两张卡上并行完成，但每格只使用
一张 RTX 4090；单格总 wall time 约 23.5 秒，峰值 allocated 显存约 2.44 GB。

### 2.1 step 500 dev

| pooling | level | NLL | accuracy | 最强静态先验 NLL | 相对先验改善 |
|---|---:|---:|---:|---:|---:|
| Deep Sets | 1 | 2.70599 | 36.365% | 5.94633 | +3.24033 |
| gated | 1 | 2.67627 | 36.753% | 5.94633 | +3.27006 |
| Deep Sets | 2 | 8.00761 | 1.865% | 8.28326 | +0.27564 |
| gated | 2 | 8.01415 | 2.074% | 8.28326 | +0.26910 |
| Deep Sets | 3 | 8.43205 | 0.0185% | 8.31777 | **-0.11428** |
| gated | 3 | 8.44006 | 0.0370% | 8.31777 | **-0.12230** |

结论：

- level 1 有强且稳定的可学习结构；
- level 2 改善较弱，但两个模型都能在 NLL 上超过静态先验；
- level 3 随训练反而恶化，两个模型都未超过 uniform prior；
- gated 对 level 1 的优势只有约 0.030 NLL/0.39 个百分点，对 level 2 NLL
  略差，对 level 3 同样失败。因此没有证据为 learned gate 增加复杂度。

原预注册要求三个 level 均超过最强静态先验，所以原始 G1 **整体不通过**；不能把
level 1/2 的成功平均掉 level 3 的失败后宣称 G1 PASS。

## 3. Level 3 失败是目标问题还是聚合器问题

新增的非参数审计只用 train 统计上下文到 target ID 的频率，再在 dev 上评估；
它不训练神经网络。上下文包含同原子的低层 shell、其余 shell、core/attachment
角色、motif size/attachment count，以及 motif 内低层状态的精确多重集。

| level | 最有信息的审计上下文 | dev seen-context | conditional accuracy | conditional NLL | unigram NLL |
|---|---|---:|---:|---:|---:|
| 1 | motif prefix multiset | 93.20% | 32.997% | 2.90089 | 5.96005 |
| 2 | motif prefix multiset | 73.93% | 11.430% | 8.48748 | 8.28115 |
| 3 | motif prefix multiset | 23.18% | 0.116% | 8.82501 | 8.38746 |

解释：

1. level 1 在 train/dev 间有直接复用的条件结构，查表和神经模型都能明显学习。
2. level 2 的精确上下文非常稀疏；查表虽提高 top-1 accuracy，却因罕见类别校准
   较差而没有改善 NLL。神经模型能把 NLL 改善约 0.27，说明需要参数共享和平滑
   泛化，而不是简单记忆。
3. level 3 有 194,588 个 train prefix contexts，dev 只有约一半能在 train 找到；
   加入精确 motif multiset 后覆盖率进一步降到 23.18%。所有条件模型 NLL 都差于
   静态先验。这与 C0 中 level 3 的高熵、99.837% 构象变化率一致：它对构象敏感，
   但折叠后的精确 ID 对“由剩余局部上下文恢复”而言近似稀疏标签。

因此 level 3 的失败不能简单解释为 Deep Sets 表达力不足；继续叠加 Set
Transformer、GNN 或更深 gate 很可能只增加容量，不改变 target 的统计性质。

## 4. 修订后的最小目标

基于上述诊断，正式代码现在允许显式冻结 `target_levels`。后验修订 G1b 只遮蔽
level 1+2，仍使用更简单的 Deep Sets；它不是对原预注册 G1 的重新命名，而是
一次明确记录的目标修订。

| 目标 | step 500 NLL | accuracy | 静态先验 NLL | 相对先验改善 |
|---|---:|---:|---:|---:|
| level 1 | 2.69911 | 36.503% | 5.94050 | +3.24139 |
| level 2 | 8.00356 | 2.074% | 8.28326 | +0.27970 |

与原三层 Deep Sets 相比，level 1/2 都小幅改善；更重要的是删除了会持续产生反向
训练信号的 level 3 精确重建。该结果支持以下科学边界：

- **保留** level 3 作为输入描述符和构象敏感性诊断；
- **不再**把 level 3 folded ID 作为主 categorical reconstruction target；
- state CE 主目标为 level 1，level 2 为辅助目标；
- 不使用 raw-ID MSE；4096 个 folded ID 没有连续距离语义；
- pooling 采用标准 Deep Sets，gated 只保留为已完成的阴性对照。

## 5. G1c 内容使用与多构象门

当前已经证明“level 1/2 state 可重建”，并完成 G1c 的两项内容使用诊断：

1. 完整 dev 的 3,359/3,360 eligible members 使用 same-atom-count 无自配对 donor；
   level 1 aligned/shuffled NLL 为 2.70336/8.56184，`delta=+5.85848`；level 2
   为 7.99151/8.40724，`delta=+0.41573`。唯一排除的是 atom-count=4 singleton，
   正常 aligned dev 评价没有排除它。两层均明确依赖正确配对的 E3FP 内容。
2. 从冻结 C0 顺序重放 144 个候选，1 个因 ETKDG prune 后不足两个构象而明确
   拒绝，取前 128 个成功分子，共 711 个构象对。完整四层输入的 motif 表示在
   711/711 对中变化，median representation RMS 为 0.31621；屏蔽高熵 level 3
   后仍为 711/711，median 为 0.25781。因此差异不只是 level 3 哈希噪声。
   128 个刚体复制的 E3FP 矩阵和 CPU FP32 motif 表示均 128/128 位级相同。

除因首次 `autodl-fs` checkpoint 为零字节而按同一冻结配置复跑一次 G1b 外，G1c
没有重新比较 Deep Sets/gated，也没有重跑 PF-1、GraphPorts 或 T5 四格。三条机制
条件现已闭合：state target 可学、aligned 优于 shuffled、同一 2D 的不同构象改变
表示且刚体复制不变。下一步进入 G2：正常 motif denoising CE、level 1/2 state CE
与 geometry-to-motif generation CE 的清晰任务交替。

GPU BF16 的同批 segment `index_add` 曾使刚体表示只有 97/128 位级相等，最大绝对
差 0.03125；输入 E3FP 本身仍为 128/128 完全相同，CPU FP32 则恢复 128/128。
因此它被记录为浮点归约次序效应，而不是化学不变性失败；后续报告用容差比较 GPU
表示，不把 BF16 位级一致作为模型效果指标。

## 6. 资源裁决

G1/G1b 的单格峰值显存不足 2.5 GB、500 updates 约 20 秒。后续 G1c 主要是推理
与 C0 小样本重放，一张 4090 已足够；无需为该门租 2/4/8 卡。多卡应保留给 G2
通过后的正式多任务训练，而不是用于扩大一个已经被统计性质否定的 level 3 目标。

## 7. Checkpoint 落盘事实

首次把 G1/G1b 输出直接写入 `autodl-fs` 时，manifest 正常完成但 `final_state.pt`
为 0 byte；所以这些 manifest 可作为已完成指标的证据，却不能作为恢复权重。G1b
随后用相同冻结配置在快盘复跑，checkpoint 为 11,165,285 bytes，`torch.load`
回读成功，并用于上述 aligned/shuffled 诊断。runner 现已把非空与结构回读设为
manifest 写出前的必要条件；以后不会再把“指标完成”与“checkpoint 可恢复”混为一谈。
