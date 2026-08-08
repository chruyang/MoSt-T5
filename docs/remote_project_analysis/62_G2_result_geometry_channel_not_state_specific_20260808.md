# G2 结果：冻结 G1b 桥接改善 CE，但未证明使用对应 3D 状态

## 1. 执行状态

G2-C（topology only）与 G2-G（topology + frozen G1b Deep Sets）均在 PF-1 run3 的相同 train/dev 成员上完成 1,000 updates。两格使用：

- 全 motif identity 遮蔽；
- 相同 GraphPorts topology、labels、成员顺序、T5 初始化和 64×2 优化协议；
- 标准 T5 sentinel CE；
- G2-G 仅额外执行冻结 G1b + `LayerNorm(128) + Linear(128,d_model)` carrier 注入。

最终 checkpoint 位于：

- `/root/autodl-tmp/g2-frozen-g1b-20260808-run4/G2-C/M0/step-1000/training_state.pt`，3,022,206,727 bytes；
- `/root/autodl-tmp/g2-frozen-g1b-20260808-run4/G2-G/M1/step-1000/training_state.pt`，3,022,998,117 bytes。

本地证据：

- `tmp/g2_control_manifest_run4_20260808.json`
- `tmp/g2_geometry_manifest_run4_20260808.json`
- `tmp/g2_paired_decision_run4_20260808.json`

## 2. 主要结果

| 指标 | G2-C | G2-G | G2-G 相对变化 |
|---|---:|---:|---:|
| final dev NLL | 1.794966 | 1.742785 | -2.907% |
| final masked-token accuracy | 0.592885 | 0.632908 | +0.040024 |
| wall time | 405.81 s | 422.21 s | +4.0% |
| members/s | 315.10 | 302.87 | -3.9% |
| peak allocated GPU memory | 21.693 GB | 21.694 GB | 近似相同 |

训练轨迹：

- G2-C NLL：57.524 → 38.809 → 8.220 → 2.047 → 1.795；
- G2-G NLL：57.655 → 20.367 → 1.750 → 1.859 → 1.743。

G2-G 显著更快进入可学习区域，并在最终 NLL/accuracy 上优于控制。这说明冻结几何通道对优化和 motif identity 分布建模有帮助。

## 3. 未通过的关键门

完整 dev 的 same-model-atom-count derangement 覆盖 3,359/3,360：

- aligned NLL：1.742874；
- shuffled NLL：1.745014；
- shuffled-minus-aligned delta NLL：**0.002140**；
- 预注册阈值：0.01。

因此：

- NLL efficacy gate：PASS；
- accuracy non-degradation gate：PASS；
- aligned geometry sensitivity gate：FAIL；
- G2 总科学门：FAIL。

不能把 G2-G 的 2.9% NLL 改善解释为“模型利用了该分子的对应 3D 状态”。错配同尺寸分子的 E3FP 后几乎保持相同表现，收益更可能来自几何通道携带的总体 motif/原子统计、通用偏置或优化捷径。

## 4. 排除的实现性解释

最终 checkpoint 只读参数检查表明：

- G2-C/G2-G 内的冻结 G1b 参数均与源 checkpoint 逐 tensor 完全一致；
- G2-C projection RMS：0.051122；
- G2-G projection RMS：0.051132；
- G2-G 相对控制的 projection delta RMS：0.001152；
- G2-G LayerNorm gamma delta RMS：0.023638。

所以失败不是 G1b 被误更新或桥接完全没有梯度。语言模型确实更新了桥接层，但没有形成足够强的“正确 E3FP 状态—正确 motif 身份”对应依赖。

两格 clip rate 均为 1.0，说明该 1,000-step screen 的全模型梯度一直触发 global clip。它可能限制小型桥接层的有效更新，但由于两格优化合同相同，不影响本次配对结论；后续可在不重跑所有旧实验的前提下，为桥接参数采用明确的 parameter group 或直接的对齐目标。

## 5. 科学裁决

当前不进入 3.36M 全量第一阶段预训练，也不因为最终 NLL 略好而接受该桥接。

下一步不应继续随机尝试多个 gate、缩放和残差补丁。FACET 与本项目已有的
多构象结果进一步表明，还应先区分两个问题：G1b 表示本身是否保留了有意义的
构象关系，以及 T5 是否使用该关系。现有 G1b 表示距离与 RMSD 的 Pearson 仅为
0.17--0.21，所以不直接启动 InfoNCE，而改为：

1. 先做不含 T5 的 G3a：在同一分子的多构象之间，以重原子距离矩阵差监督一个
   motif-topology-aware 关系编码器；
2. 只有 held-out molecule 上的 Pearson/Spearman 与排序指标明确超过 frozen G1b
   baseline，才冻结该编码器并进入 G3b；
3. G3b 保留标准 motif identity CE，并增加 matched-vs-shuffled 条件对比 CE；
4. negative 固定 2D motif identity 与 motif atom count，避免任务只靠尺寸或身份判断；
5. G3a/G3b 通过前不引入 raw E3FP-ID MSE，不扩大预训练数据。

这一顺序比直接 MSE 或直接 InfoNCE 更能定位失败来源。详细协议见文档 63。

### 5.1 零桥后验进一步收紧结论

在加载 G2-G step-1000 checkpoint 后，将
`bridge_projection.weight` 置为严格的零，对同一完整 dev 重放一次：

- 原 G2-G NLL：1.742785；
- 零桥 NLL：1.740408；
- zero-minus-aligned NLL：-0.002377；
- 零桥 masked-token accuracy：0.633614。

零化几何桥没有造成性能下降，而是轻微改善。因此，比“同尺寸
shuffle 只有很小差值”更强的结论是：**step-1000 T5 已经可以在推理时完全丢弃
G1b 几何桥**。G2-G 的快速收敛是训练过程效应，不是最终模型对匹配
3D 状态的依赖。

本地证据：`tmp/g2_zero_bridge_ablation_run4_20260808.json`。

## 6. 运行与存储经验

RTX 4090×2、64×2 实际稳定运行，峰值 `nvidia-smi` 约 23.6/24.6GB，不应继续增大 micro-batch。

`autodl-fs` 对 PyTorch ZIP checkpoint 的随机写以及后续顺序复制均发生实际失败。G2 最终采用：

- 输出/checkpoint 写入 `/root/autodl-tmp`；
- step 500 只保留评估 marker；
- step 1000 保存完整恢复状态。

该调整不改变训练、评估或科学判定，只减少分钟级机制筛选的重复 3GB checkpoint。
