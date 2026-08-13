# Anchored V4 shell-screen 结果（2026-08-10）

> 后续 matched-state confirmation 已表明本轮单 seed 的 `l0_l12_mean` 排名不能作为最终 shell 冻结依据；B2D 在同 shell 接口下明显更强，且两个 F3D 候选对 same-identity donor 不敏感。最终裁决见 `73_anchored_v4_matched_confirmation_results_20260810.md`。

## 执行状态

单张 RTX 4090 顺序完成六个 cell；launcher 与六个 cell 均为 `pass`，无 traceback、OOM 或残留训练进程。

- 数据：PF-1 frozen train/dev，30,240 / 3,360 members；
- 训练：每格 1,000 optimizer updates，microbatch 64 × accumulation 2；
- 动态 corruption、相同初始化、相同成员曝光和相同开发集；
- 总墙钟时间：2,094.83 s（34.91 min）；
- 每格训练时间：317.87--351.08 s；
- 峰值 CUDA reserved：13.77--13.85 GiB；
- 每格实际成员曝光：127,872，短尾 update 的成员数范围为 96--128；
- 本轮仅筛选候选 shell 组织方式，未保存模型 checkpoint。

本地证据根目录：`tmp/anchored-v4-shell-screen-v1-b0693bb`。

## update 1000 主结果

`aligned` 表示使用与分子对齐的状态，`zero` 表示在相同模型上将几何状态置零。正的 `zero - aligned` NLL 表示模型依赖所提供状态。

| cell / shell mode | aligned NLL | aligned accuracy | zero NLL | zero - aligned NLL | clip rate |
|---|---:|---:|---:|---:|---:|
| B0 / no state | 0.7966 | 79.75% | 0.7966 | -- | 0.696 |
| B2D / Morgan control | 0.8713 | 77.85% | 1.3757 | +0.5043 | 0.371 |
| F3D / `l12_mean` | 0.9430 | 75.45% | 1.1623 | +0.2193 | 0.533 |
| **F3D / `l0_l12_mean`** | **0.6605** | **82.85%** | **1.5632** | **+0.9027** | **0.409** |
| F3D / `l0123_mean` | 1.0435 | 73.94% | 1.3805 | +0.3371 | 0.533 |
| F3D / `l0_shell_attention_l123` | 1.4200 | 69.16% | 1.1702 | **-0.2498** | 0.873 |

## shell 选择裁决

冻结 `l0_l12_mean` 作为下一阶段唯一候选：

1. 相对 B0，aligned NLL 下降 0.1361（17.1%），accuracy 提高 3.11 个百分点；
2. 相对同结构的 B2D 控制，NLL 下降 0.2108（24.2%），accuracy 提高 5.01 个百分点；
3. 去除 L0 后，`l12_mean` 的 NLL 退化到 0.9430，说明 motif token 并不能替代原子级身份上下文，L0 的消费有实证必要；
4. 加入 L3 后，`l0123_mean` 退化到 1.0435，当前任务没有显示 L3 带来净收益；
5. 可学习 shell attention 在 update 500 的 NLL 为 1.2468，update 1000 反而恶化到 1.4200，且 aligned 比 zero 更差。它不是“更灵活必然更好”，当前应淘汰，而不是继续增加复杂度。

该结果支持“L0 承担原子身份上下文，L1/L2 承担局部环境状态”的分工；仍不把 L0 计作 3D 证据。

## carrier 与 endpoint

最佳 `l0_l12_mean` 在 update 1000 的组件消融：

| 几何组件 | NLL | 相对 zero 的改善 |
|---|---:|---:|
| zero | 1.5632 | -- |
| carrier only | 1.1611 | +0.4021 |
| endpoint only | 1.4438 | +0.1194 |
| both | **0.6605** | **+0.9027** |

两者独立改善之和为 0.5215，而共同启用改善 0.9027，额外协同约 0.3812 NLL。因此：

- carrier 是主要通道；
- endpoint 单独较弱，但并非无效；
- endpoint 的价值主要在于与 motif carrier 联合表达 attachment/connection 局部状态；
- 下一阶段保留两者，不应把 endpoint 单独替代 carrier，也不应在尚无证据时继续扩展 endpoint 语法。

## 可以与不可以得出的结论

本轮可以支持：

- anchored V4 数据路径、训练路径和组件消融可稳定运行；
- `l0_l12_mean` 明显优于其余 shell 组织方式；
- L0 对当前 motif 表示是必要补充；
- L3 与可学习 shell attention 当前无净收益；
- carrier 与 endpoint 在最佳 F3D 模式下存在强协同；
- 单纯增加坐标无关的 Morgan 状态不能解释最佳 F3D 的全部收益。

本轮仍不能宣称：

- 已证明一般性的 3D 表征增益；
- 已证明构象敏感性；
- 已证明下游 QM9、captioning 或检索能力提高；
- 已选定正式全量预训练协议。

原因是本轮只有一个 seed、PF-1 规模和 identity-denoising 开发集；`zero` 是通道移除，而不是 same-identity donor shuffle 或真实构象扰动。B2D 对照提高了证据强度，但不足以单独排除全部二维身份捷径。

## 下一门

只重跑两个必要 cell 并保存最终 checkpoint：B2D 与 F3D `l0_l12_mean`。保持当前 64 × 2、1,000 updates 与数据顺序，不重跑已淘汰的三个 F3D shell。随后在固定 checkpoint 上完成：

1. aligned / zero / same-identity matched-donor 三联；
2. both / carrier-only / endpoint-only 复核；
3. 若存在可用的真实 3D 敏感标签，再进入 QM9 子集门；
4. 只有 F3D 相对 B2D 在 3D 敏感指标上仍保持优势，才推进正式较大规模训练。

这一步不是推翻 anchored motif 架构，而是将 V4 从四个 shell 候选收敛为一个简洁实现：

`motif phrase carrier + attachment endpoint + L0 identity context + mean(L1,L2) local state`。
