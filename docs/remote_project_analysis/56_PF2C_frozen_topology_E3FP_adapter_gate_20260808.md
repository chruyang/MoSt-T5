# PF-2C：冻结拓扑骨干的 E3FP adapter 门

## 1. 触发原因

PF-2B T3MI（全部 motif identity 被遮蔽、GraphPorts 拓扑可见）得到：

- M0-T：dev NLL 1.794966，accuracy 0.592885；
- M1-T：dev NLL 0.973769，accuracy 0.759984；
- M1/M0 NLL 比 0.542500；
- 最终 gate 为 -0.067486；
- 但 same-atom-count shuffled E3FP 的 ΔNLL 仅 0.0000158。

对最终 M1-T 权重进一步做无训练扰动：gate=0、shell occupancy-only、
分子内 atom-row rotation、随机 E3FP IDs 的 NLL 均与 aligned 相差不超过
5.1e-5。故 M1-T 的收益来自不同训练轨迹/正则化，而不是最终推理时使用
对齐 E3FP 内容。不得据此进入 conformer probe，也不得据此引入 teacher/MSE。

## 2. 最小机制修正

PF-2C 不再重复训练 M0：

1. 从已完成的 M0-T step1000 加载完整 T5 与零 gate E3FP 模块；
2. 冻结全部 T5 参数；
3. 仅训练 `shared_embedding.weight` 与 `geometry_gate_logit`；
4. 数据、全 identity mask、标准 T5 CE、64x2、1000 updates、dev 与 shuffle
   诊断均沿用 T3MI；
5. 不增加词表、projection、teacher、MSE 或辅助 loss。

该设计使 T5 不能通过改变自身参数吸收“几何噪声改善优化”的效应。若模型
改善，改善必须由推理时仍存在的 E3FP adapter 提供；same-size shuffle 再判断
该改善是否依赖正确分子配对。

## 3. 预注册门槛

- adapter aligned NLL / frozen M0-T NLL <= 0.98；
- adapter accuracy >= M0-T accuracy - 0.01；
- shuffled NLL - aligned NLL >= 0.01；
- `abs(tanh(gate)) >= 0.001`。

四项同时通过才进入 same-2D conformer probe；否则回到 E3FP 状态定义或
motif carrier reduction，不自动引入 teacher/MSE。

## 4. 解释边界

这是单 seed、1% cohort 的机制筛选，不构成架构优越性或统计显著性证据。
冻结骨干属于 adapter 式归因控制，不是知识蒸馏，也没有第二个 teacher 模型。
