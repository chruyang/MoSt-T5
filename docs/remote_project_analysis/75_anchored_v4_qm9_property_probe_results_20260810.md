# Anchored V4：QM9 属性探针结果与下一步裁决

日期：2026-08-10

## 1. 执行结论

正式四格均完成，launcher 状态为 `pass`。运行目录为：

`/autodl-fs/data/most-t5-r1/qm9-anchored-probe-v1-r2-b0693bb`

- 数据：10,002 个 records；train/dev/test = 8,003/1,000/999；分子身份互斥。
- 目标：HOMO、LUMO、gap；单位为 Hartree；缺失标签只在 loss/metric 中 mask。
- 每格：30 epochs，960 optimizer updates，batch 256，AdamW，LR 3e-4。
- 总墙钟约 811 秒；单格训练和消融评估约 181--207 秒。
- 四格均无 OOM、非有限 loss 或数据 reject。

这轮结果证明 anchored motif + carrier/endpoint 几何接口可以被 T5 下游回归任务有效消费；但它尚未证明性能增益来自三维构象，而不是原子级二维环境信息。

## 2. Test 主结果

下表均为三个任务的宏平均 MAE，越低越好。

| cell | state | shell | test MAE (Hartree) | 相对 B0 |
|---|---|---|---:|---:|
| B0 | zero | 参数拓扑对齐 | 0.0142733 | -- |
| B2D | stereo-free Morgan | radius 0/1/2 | **0.0105741** | -25.9% |
| F3D-12 | E3FP | L0 + mean(L1,L2) | 0.0129552 | -9.2% |
| F3D-123 | E3FP | L0 + mean(L1,L2,L3) | 0.0108830 | -23.8% |

F3D-123 比 F3D-12 低约 16.0%，说明在当前实现中不能删除 L3；但 F3D-123 仍比 B2D 高约 2.9%，没有通过预注册的“F3D 优于坐标无关 B2D”门。

该方向在三个目标上完全一致，而不是被单一目标驱动：

| cell | HOMO MAE | LUMO MAE | gap MAE |
|---|---:|---:|---:|
| B2D | **0.0085148** | **0.0109257** | **0.0122817** |
| F3D-123 | 0.0086280 | 0.0114451 | 0.0125759 |

Dev 方向也一致：B2D 为 0.0106696，F3D-123 为 0.0109991。因此当前差异不是 test-only 反转。不过只有一个固定 seed，2.9% 的 B2D/F3D 差距不能被解释为最终模型排名；它足以否定“本轮已经证明三维增益”，但不足以否定 E3FP。

## 3. Carrier 与 endpoint 的因果消融

同一训练后 checkpoint 在 test 上执行输入组件消融：

| cell | both | carrier only | endpoint only | zero |
|---|---:|---:|---:|---:|
| B2D | **0.0105741** | 0.0129316 | 0.0195565 | 0.0241555 |
| F3D-12 | **0.0129552** | 0.0165947 | 0.0215748 | 0.0249349 |
| F3D-123 | **0.0108830** | 0.0126018 | 0.0190909 | 0.0249889 |

可以作出的结论：

1. `both < carrier_only` 在三格均成立，因此 endpoint 在当前已训练模型中提供互补信息；它不是死代码。
2. `carrier_only < endpoint_only` 在三格均成立，因此 carrier 是主要通道，endpoint 不能单独替代 motif 摘要。
3. carrier 与 endpoint 写入不同 token，各自执行 0.5/0.5 convex interpolation；`both` 并不是在同一 token 上简单加倍几何幅度。
4. `zero` 是对已依赖 side channel 的 checkpoint 做分布外移除，不能代替独立训练的 B0。主质量比较必须使用独立 B0；组件消融只用于判断同一 checkpoint 的通道依赖。

因此 endpoint 应暂时保留，但其科学表述应为“attachment-local complementary state address”，而不是独立 motif 表征。

## 4. L3 结果如何解释

缓存中的非负 E3FP 行数为 L0/L1/L2/L3 = 87,931/87,931/87,914/1,696；L3 只覆盖约 1.9% atom rows。尽管如此，F3D-123 显著优于 F3D-12。

代码审计确认两格具有：

- 同一 union-init、adapter seed、head seed；
- 同一 DataLoader generator、训练顺序、dropout seed、更新数和 schedule；
- 相同参数拓扑；
- 对缺失 L3 采用 masked mean，不会把 `-1` 当作真实状态。

因此差异不是明显的初始化或样本顺序漂移。不过它仍可能来自少量 L3 分子被放大的任务相关性或单 seed 优化方差。当前正确裁决是：

- `l0_l12_mean` 不应作为正式候选；
- `l0_l123_mean` 是下一轮 F3D 的暂定模式；
- 尚不能声称 L3 一般有效，需在更明确的 geometry-sensitive targets 上复核，并报告有/无 L3 子群指标。

## 5. 为什么本轮没有证明三维优势

HOMO、LUMO 和 gap 与电子结构相关，但可由二维拓扑、元素环境和共轭结构强预测。B2D 是严格 stereo-free、坐标无关且参数匹配的原子环境控制；它在 dev/test 和三个目标上均略优于 F3D-123。这说明当前 benchmark 首先奖励“原子级环境补充”，而不是唯一奖励构象状态。

所以本轮支持：

> motif 文本本身不足以完整承载原子级环境；L0/高阶 state side channel 与 endpoint 地址能增强 anchored T5 的属性表征。

本轮不支持：

> E3FP 已经比同容量二维状态提供稳定的三维特异增益。

## 6. 架构裁决

目前不应推翻 anchored motif、T5 主干或 E3FP，也不应回到 motif identity denoising 反复试验。冻结如下：

- anchored pure motif 语言：保留；
- fallback chemical lexer：保留；
- carrier + endpoint 两级地址：保留；
- B2D：保留为所有后续几何结论的必要控制；
- F3D shell：暂定 `l0_l123_mean`；
- GraphPorts：只作为权威无损来源/旧产物兼容，不作为当前 encoder 文本表面；
- “F3D 优于 B2D”仍是三维主张的硬门，不能用“F3D 优于 B0”替代。

## 7. 下一步

下一门改为标准 QM9 中更有几何含义的连续目标，并保持同一模型接口、同一分子切分和同一 B0/B2D/F3D 对照：

1. 优先：dipole moment `mu`、electronic spatial extent `R2`；
2. 次级：polarizability `alpha`、内能 `U0`；
3. 分别报告每个 target，不把性质简单宏平均后掩盖方向；
4. 对 L3-present 与 L3-absent 分层报告 F3D-123；
5. 若能获得同一分子的可信多构象，再增加 same-identity conformer discrimination；RDKit 生成构象不作为主门的真值。

若 F3D-123 在这些目标上仍不能优于 B2D，则下一步应改变 E3FP 的组织/监督方式（例如 shell attention 或更直接的 atom-memory readout），而不是扩大正式预训练规模。若 F3D 稳定优于 B2D，再进入 anchored 3D-MotifT5 的阶段化预训练。

## 8. 证据文件

本地镜像：`tmp/qm9_anchored_probe_v1_r2_b0693bb/`

- `launcher_status.json`
- `B0_probe_manifest.json`
- `B2D_probe_manifest.json`
- `F3D_l0_l12_probe_manifest.json`
- `F3D_l0_l123_probe_manifest.json`

远端 checkpoint 仍保存在正式运行目录；本地只固化小型 manifest，不复制四份约 1.21 GB 权重。
