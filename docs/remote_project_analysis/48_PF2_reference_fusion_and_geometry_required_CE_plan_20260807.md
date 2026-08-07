# PF-2：参考融合、几何必需 CE 与 teacher 后置裁决

> 日期：2026-08-07
> 状态：PF-2A 设计、代码门与 step-0 sensitivity 已冻结；正式 M0-R/M1-F 尚未运行
> 范围：只裁决 3D 融合与训练目标是否要求使用 E3FP，不宣称最终架构或下游效果

## 1. 为什么 PF-1 不能被解释为“E3FP 或 motif 3D 不可行”

PF-1 淘汰的是一个精确组合：四张 level-specific E3FP 表、四层直接求和、motif 内原子均值、无缩放残差相加，以及 15% identity-recovery CE。它没有测试论文中 3D-MolT5 的原始融合。

PF-1 的事实为：

- A1/M1 在 step 0 的 matched-vs-shuffled E3FP ΔNLL 约为 `0.223/0.231`，说明几何入口最初确实影响输出；
- step 1,000 时 ΔNLL 降至 `0.000195/0.000848`，模型几乎不再响应几何；
- 四张几何表 RMS 仍约为 1，没有整体收缩到零；
- M1 相比 M0 的最终 NLL 恶化约 `0.493`，accuracy 下降约 10 个百分点；
- 四格全部 1,000 updates 都触发 global-norm clip 1.0。

对真实冻结 dev 首批 64 条记录的数值诊断进一步显示：

| 条件 | update | base carrier RMS | geometry delta RMS | fused RMS | 每 carrier `||G||/||X||` 中位数 |
|---|---:|---:|---:|---:|---:|
| A1 | 0 | 2.7839 | 1.9438 | 3.3944 | 1.9579 |
| A1 | 1,000 | 2.7893 | 1.9440 | 3.3993 | 1.9228 |
| M1 | 0 | 3.0446 | 1.7951 | 3.5344 | 1.7703 |
| M1 | 1,000 | 3.0473 | 1.7954 | 3.5376 | 1.7461 |

`X` 与 `G` 的 cosine 接近 0。几何分支产生的是一个与原 token 表示近乎正交、幅度并不小的扰动；训练后扰动仍大，而输出对 shuffled E3FP 几乎不敏感。与这些观测一致、但尚未被唯一因果定位的解释是两个问题可能共同存在：

1. **融合尺度/参数化偏离已验证先例；**
2. **普通 identity CE 可由未遮蔽身份与二维拓扑完成，并不迫使模型持续使用 3D。**

这仍不是对二者各自贡献的因果分解，所以 PF-2 必须分阶段，而不能同时加入 gate、T3MI、teacher 和 MSE。

## 2. 文献与官方代码给出的边界

### 2.1 3D-MolT5：最直接的同域参考

3D-MolT5 的官方 `FPT5EncoderStack.py` 使用：

- 一张共享 E3FP embedding 表；
- 对四个固定 E3FP shell slot 直接取均值，缺失 slot 对应零 padding row；
- 1D 与 3D 同时存在时固定使用 `0.5 E_1D + 0.5 E_3D`；
- 3D-only 输入时使用纯 E3FP state。

论文还同时采用 1D+3D joint denoising 和 3D→1D translation；消融显示移除 joint denoising 或 translation pretraining 都会损害 3D 相关性质任务。但其 translation 消融不是单独隔离 3D→1D，所以它只能支持“需要补充一个让结构条件承担预测责任的任务”，不能证明本项目的 T3MI 或混合比例天然最优。

- 论文：[3D-MolT5, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/3d4dc72d715bd6415d356293079adf3d-Paper-Conference.pdf)
- 官方代码：[QizhiPei/3D-MolT5](https://github.com/QizhiPei/3D-MolT5)

### 2.2 gate：能保护语言模型，但不能强迫模型使用 3D

Flamingo 使用零初始化 `tanh` gated residual，使加入视觉条件时初始函数等于原语言模型；LLaMA-Adapter 也使用零初始化 attention gate。这些证据支持 gate 作为“安全注入”机制，但不证明 gate 会保持几何敏感性：如果 CE 可绕过 3D，gate 仍可能长期接近零。

- [Flamingo, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/960a172bc7fbf0177ccccbb411a7d800-Paper-Conference.pdf)
- [LLaMA-Adapter, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/c196239c5f9481e0db2755f31fe4585f-Abstract-Conference.html)

因此 gate 是 PF-2A 失败后的独立候选 `F-Gate`，不能混入严格参考臂。

### 2.3 CE+MSE/teacher：结构上可行，但不是当前第一修复

连续目标与 token CE 可以共存：Uni-Mol 同时使用 masked-token、坐标和距离目标；3DMolFormer 组合 token CE 与坐标 MSE；MolBERT 将 MaskedLM CE 与连续分子描述符 MSE 联合训练。data2vec 与 Mean Teacher 则说明 EMA teacher 可提供稳定的连续 latent target。

- [Uni-Mol, ICLR 2023](https://openreview.net/pdf?id=6K2RM6wVqKu)
- [3DMolFormer, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/449590dfd5789cc7043f85f8bb7afa47-Paper-Conference.pdf)
- [MolBERT, NeurIPS 2020 ML4Molecules](https://arxiv.org/pdf/2011.13230)
- [data2vec, ICML 2022](https://proceedings.mlr.press/v162/baevski22a.html)
- [Mean Teacher, NeurIPS 2017](https://proceedings.neurips.cc/paper_files/paper/2017/hash/68053af2923e00204c3ca7c6a3150cf7-Abstract.html)

证据边界是：MSE 应作用于连续、对齐并适当归一化的表征，不能对 E3FP hash ID 本身做 MSE；EMA 只能平滑已有表示，不能凭空产生模型尚未学习的 3D 信息。当前直接上 teacher 会同时掩盖融合错误和任务捷径，因此后置。

## 3. PF-2A：只测试 reference-calibrated fusion

### 3.1 精确融合

对 atom `a` 的四个固定 E3FP slot：

\[
g_a = \frac{1}{4}\sum_{l=0}^{3} E(d_{a,l}),\quad E(-1)=0
\]

同一 logical motif carrier `t` 上：

\[
g_t = \operatorname{mean}_{a:C(a)=t} g_a
\]

carrier token 使用：

\[
x'_t = 0.5x_t + 0.5g_t
\]

非 carrier token 保持 `x'_t=x_t`。一张共享表参数量为：

\[
(4096+1)\times768=3,146,496
\]

相比 PF-1 的 `12,585,984` 个几何参数减少 `9,439,488`。该变更包同时改变了表共享、shell reduction 和融合系数，所以只作为有直接外部依据的参考整体；不能凭一次结果声称其中某个内部因素单独造成改进。

### 3.2 最小配对，不重跑四格

PF-2A 只运行 motif 两格：

| 条件 | 几何 | 融合 | 目标 |
|---|---|---|---|
| M0-R | 无 | reference wrapper 中几何支路不执行 | 原 15% motif identity CE |
| M1-F | inherited E3FP | 上述 F-Ref | 与 M0-R 完全相同的 CE |

重新运行 M0-R，而不是仅拿旧 M0 作主对照。batch-64 资源探针已经通过，但 `30,240` 个 train members 不能被 64 整除，正式 preflight 在任何 optimizer update 前按合同拒绝。为避免引入跨 epoch 拼 batch 或丢弃尾批的补丁，PF-2A 统一采用 `microbatch 63 × accumulation 2 = effective batch 126`：63 低于已通过的显存上限，并且 `30,240/63=480` 恰好整除。数据、mask、更新数和 LR schedule 不变；但 effective batch 与 dropout forward 分组均不同于旧 `32×4` PF-1，因此旧 M0 只作支持性历史基线。

冻结合同：

- run3 同一 33,600-member release，train/dev `30,240/3,360`；
- 同一 tokenizer、union-init、model/fusion seed；
- 同一成员顺序、corruption、1,000 updates、AdamWScale、warmup/cosine、clip 1.0；
- M0-R/M1-F 均使用 `63×2=126`；
- 同一 step 0/250/500/750/1,000 dev；
- M1-F 的 matched-vs-same-atom-count shuffled E3FP 仍覆盖预注册的 `3,359/3,360` dev 子域；
- 不加入 gate、adapter、MSE、teacher 或第二个任务。

### 3.3 预注册判读

未训练 M1-F 的完整 matched dev 已在任何 optimizer update 前测得 `ΔNLL_init=0.18344068`。因此最终 practical sensitivity gate 已冻结为 `0.01834407`。定义为：

\[
\Delta NLL_{1000} \ge \max(0.01, 0.1\Delta NLL_{init})
\]

同时要求：

- `NLL(M1-F) <= 1.02 × NLL(M0-R)`；
- `accuracy(M1-F) >= accuracy(M0-R) - 0.01`；
- 所有 loss/gradient 有限，无数据 reject、替样或截断。

`0.01` 是本项目预注册的 practical sensitivity floor，不是文献宣称的普适常数；`10%` 防止只保留初始化效应的极小残余。

裁决树：

1. **通过 practical non-degradation gate 且 sensitivity 通过：**保留更简单的 F-Ref；先做统一 3D-sensitive probe，不自动增加 T3MI。
2. **通过 practical non-degradation gate 但 sensitivity 不通过：**融合可训练，但普通 CE 仍允许绕开 3D；进入 PF-2B。
3. **CE 明显退化或数值不稳定：**先测试独立 F-Gate；不允许用更难任务或 teacher 掩盖融合问题。

PF-2A 即使通过，也只证明模型在该任务中使用了 E3FP 条件，不能证明学到了真实构象。E3FP 同时包含原子身份与拓扑，最终必须由 same-2D/different-conformer 或 QM9 3D-sensitive probe 约束 claim。

## 4. PF-2B：仅在任务捷径被触发时启用 T3MI

PF-2B 的唯一新任务命名为 `Topology-conditioned E3FP-to-Motif-Identity Reconstruction (T3MI)`：

- 对所有 logical motif identity span 进行 sentinel corruption；
- 移除 encoder 中所有原 motif macro/fallback identity token；
- 保留 GraphPorts connection skeleton 与 motif 顺序；
- 在每个 motif sentinel carrier 注入 F-Ref E3FP；
- decoder 仍用标准 T5 sentinel CE 恢复完整 motif identity；
- 不增加词表、head、teacher 或 MSE。

它不是“纯 3D→motif”：topology 仍可见，E3FP 也含二维身份信息，所以名称必须保留 `topology-conditioned`。

最小因果控制只有两格：

| 条件 | T3MI | E3FP |
|---|---|---|
| M0-T | 有 | 无 |
| M1-T | 有 | F-Ref |

M0-T 不可省略，否则 M1-T 的收益可能只是更强的 motif curriculum、更多 target token 或任务难度变化。第一轮不搜索多种混合比例；若进入正式混合训练，预注册为按 supervised target-token budget 的 `75%` 普通 identity CE + `25%` T3MI，并同时报告 record、encoder token、decoder token 与 FLOPs。

T3MI 的 identity recovery 只能作为机制指标。只有 M1-T 相比 M0-T 的优势在 shuffled E3FP 后衰减，并在 QM9 gap/HOMO/LUMO 或 same-2D conformer probe 上形成同方向增益，才能支撑 3D 表征主张。

## 5. teacher/MSE 的进入条件

当前优先级冻结为：

```text
F-Ref + ordinary CE
        ↓ 若融合安全但仍忽略3D
F-Ref + T3MI CE
        ↓ 若E3FP已可用但表征仍缺少稳定连续目标
独立 frozen/EMA 3D teacher 预对齐
        ↓ 最后才考虑
joint CE + normalized latent MSE
```

如果 T3MI 已恢复 E3FP 使用、改善 3D-sensitive endpoint 且不损害普通 CE，teacher/MSE 从主架构删除，而不是继续叠加。若最终进入 teacher，必须新增独立 state-visibility mask，分别记录 CE/MSE 与两者梯度；不能复用数据有效性 mask，也不能把总 loss 冒充 CE NLL。

## 6. 代码、真实 smoke 与资源

PF-2A 使用独立 `most_t5_next/p2` 模块，PF-1 legacy fusion 和旧 checkpoint 合同保留。当前代码门：

- 新 reference fusion 与 wrapper factory；
- PF-2A 单格 runner，强制只接受 M0 或 M1；
- step 500/1,000 checkpoint 额外绑定完整 PF-2 fusion 与 optimization contract；
- 配对 merger 先核对 M0/M1 数据、初始化、训练曝光与 final-dev 合同，再自动应用三项 practical gate；单格 `status=pass` 只表示训练完成；
- M0/M1 统一 `63×2`，并在启动前证明 train membership 可整除；
- nmb1 完整 P1+P2 CPU 回归 `134/134 PASS`；
- 真实 run3 首批 32 条 M1 BF16 forward/backward/AdamWScale step PASS；
- smoke loss `70.8554`，preclip norm `7,670.93`，fusion gradient finite/nonzero；
- 几何参数 `3,146,496`；峰值 allocated `6,639,682,560` bytes。

正式 PF-2A 只有 M0-R 与 M1-F 两个短任务。单张 RTX 4090 顺序运行即可，预计约 25–40 分钟；两张卡可一格一卡缩短墙钟，但不会增加科学信息。当前不需要 4 或 8 卡。若进入 PF-2B，M0-T/M1-T 再使用两格并行；只有扩大到完整四格或多 seed 时才建议升至 4 卡。

## 7. 证据锚点

- PF-1 正式 manifest：`tmp/pf1_training_manifest_run3_mb32_3a74251.json`
- PF-1 sensitivity：`tmp/pf1_geometry_sensitivity_trajectory_20260807.json`
- PF-1 几何参数轨迹：`tmp/pf1_geometry_parameter_trajectory_20260807.json`
- 真实 carrier 尺度诊断：`tmp/pf1_geometry_scale_diagnostic_20260807.json`
- PF-2A step-0 sensitivity：`tmp/pf2_reference_fusion_initial_sensitivity_20260807_v2.json`
- PF-2A fusion：`most_t5_next/p2/reference_geometry_fusion_v1.py`
- PF-2A runner：`most_t5_next/p2/run_pf2_reference_fusion_v1.py`
- PF-2A paired merger：`most_t5_next/p2/merge_pf2_reference_fusion_v1.py`
- 本地 smoke 脚本：`tmp/pf2_reference_fusion_real_smoke.py`
- 官方 3D-MolT5 源码镜像：`reference_repos/3D-MolT5_official_src_82dbe088`

最终边界：PF-2 是机制裁决，不是为了堆消融。每一步只引入一个可解释问题，失败即按裁决树转向；不运行 gate × T3MI × teacher × MSE 的笛卡尔积。
