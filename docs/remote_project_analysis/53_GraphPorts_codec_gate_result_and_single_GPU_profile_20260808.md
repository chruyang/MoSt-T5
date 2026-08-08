# GraphPorts codec 配对门结果与单卡 GPU profiler 裁决

> 日期：2026-08-08  
> 状态：M0-v1/M0-v2 配对 GPU gate 与独立 13-update profiler 均已完成；正式保留 GraphPorts v1  
> 范围：1% PF-1 failure screen、单一配对 seed、codec-only 比较；不构成 motif 粒度、3D 表征或下游性能结论

## 1. 执行边界

本轮在 commit `3e6eece7fab2bee8821b57a6fc0d6083e7da8f4c` 上，以同一张 RTX 4090 顺序运行：

1. GraphPorts v1 的 3 warmup + 10 measured update profiler；
2. `M0-v1` 的 1,000-update 训练；
3. `M0-v2` 的 1,000-update 训练；
4. 预注册自动裁决。

两格共享同一 30,240/3,360 train/dev 成员、顺序、corruption、union-init、T5 初始化、AdamWScale、学习率日程、clip、更新数与 target exposure。协议为 `microbatch=64, accumulation=2, nominal effective batch=128`，采用标准可变尾批而非丢弃成员：两格均看到 127,872 个 train-member views，只有 4 个短微批，最小/平均/最大 members per update 为 96/127.872/128，最终 cursor 都是 epoch 4、batch 108。

因此本轮唯一科学自变量是 GraphPorts connection surface：

- v1：显式 A/B endpoint marker 与 edge id；
- v2：按规范 edge order 隐式恢复 id，只保留两个自定界 endpoint。

v2 在进入 GPU 前已经通过 33,600 条 source-bound、decode-to-same-bonds 与全 corruption epoch 配对门。本轮不是以训练效果替代化学正确性验证。

## 2. GPU profiler：worker 不是当前瓶颈

M0-v1 profiler 的 10 个 measured updates 得到：

| 指标 | 实测 |
|---|---:|
| mean update wall | 0.350257 s |
| members/s | 365.446 |
| encoder tokens/s | 17,620 |
| prepared-data queue wait | 0.01445% wall |
| tensor adapter/H2D wall | 4.45% |
| forward wall | 18.44% |
| finite-loss host sync wall | 7.79% |
| backward wall | 51.71% |
| gradient clip wall | 2.01% |
| AdamWScale optimizer wall | 7.33% |
| peak allocated/reserved | 19.256/20.456 GB |

结论不是“CPU 核数越多越好”。正式训练已经使用一次严格解码缓存、4-worker warmup 和 depth-2 ordered prefetch；queue wait 仅占 0.01445%，说明当前 GPU 锯齿主要来自短序列/微批 kernel、forward/backward 边界以及 loss/optimizer 的 host synchronization，而不是 LMDB、JSON decode 或 worker 数不足。继续把 worker 从 4 提到 8/16 不会产生可观收益，也不应为追求 CPU 使用率而增加并发复杂度。

`64×2` 在正式训练计算段多次达到 84%–98% GPU utilization、约 300–341 W。驱动侧显存采样最高约 23.78/24.56 GiB，因此它是当前实现的实测上限附近配置，不再增加 microbatch。profiling 不是科学 gate 输入，正式两格也没有根据 profiler 临时改优化器。

## 3. v1/v2 配对结果

| 指标 | GraphPorts v1 | GraphPorts v2 | v2/v1 或差值 |
|---|---:|---:|---:|
| step-1000 dev NLL | 1.876950 | 2.013537 | 1.072770（恶化 7.28%） |
| step-1000 masked-token accuracy | 0.568741 | 0.561500 | -0.007240（-0.724 point） |
| dev encoder-token ratio | 1.000000 | 0.625107 | -37.49% |
| members/s | 291.463 | 340.995 | 1.169942（+16.99%） |
| peak allocated memory | 21.564 GB | 19.533 GB | 0.905797（-9.42%） |
| full condition wall | 438.72 s | 375.00 s | -14.52% |

v2 的晚期轨迹稳定：NLL 从 step 750 的 2.090757 降至 step 1,000 的 2.013537，accuracy 从 0.552656 升至 0.561500；所以不是训练崩溃或 checkpoint 故障。但是它未通过最关键的 non-degradation gate：最终 NLL 只能容许 2% 恶化，而实际为 7.28%，并超过预注册的 5% 硬拒绝边界。

自动裁决为：

```text
retain_graphports_v1
```

七项 promotion gate 中，长度、accuracy、晚期稳定性、吞吐和显存六项通过，只有 final NLL ratio 失败。因为落在硬拒绝区而非 2%–5% 灰区，不追加第二个 seed，也不继续构建 `v2 + 64-merge fallback BPE`。

## 4. 对 motif 方案的含义

本轮不能用于否定 motif 划分：v1/v2 使用完全相同的 motif partition、identity、macro/fallback、mask unit 和 CE target。它裁决的是连接语法，而非 motif 粒度。

当前最谨慎且可复现的解释是：v2 在可逆性上与 v1 等价，但删除显式 endpoint role/edge boundary 后，当前 T5 在 1,000-update CE screen 中学习拓扑的难度增加。这个解释与 NLL 结果一致，但尚不是唯一机制证明；因此不据此新增更多 codec 变体。

主线冻结为：

- GraphPorts v1 继续作为严格可逆、当前可学习性更好的生产基线；
- GraphPorts v2 保留为 lossless/长度分析候选，不进入当前训练主线；
- 不因 `M>A` 单一长度指标修改 CAMT5-derived motif partition；
- 只有 motif editing、结构恢复或下游 probe 显示当前粒度本身失败时，才比较一个预定义的更粗 partition；不做无边界架构枚举。

这也说明 motif 表示的价值不能被简化为“token 必须比 SELFIES 少”。motif 的主要研究价值仍是结构级 mask、编辑位置和 atom-to-motif 3D pooling；其额外序列成本必须由这些任务上的收益来证明。

## 5. 后续顺序与资源

GPU codec gate 已闭合，当前无需继续占用 4090。下一阶段先在本地/无卡实例完成：

1. 将 GraphPorts v1 重新标记为当前主线，并冻结本轮 manifests/adjudication；
2. 只在 v1 上定义并实现独立 F-Gate，用零初始化 gated residual 检查 PF-2A 的固定 `0.5` 注入是否是 CE 退化主因；
3. F-Gate 必须先通过初始化函数等价、M0/M1 配对、checkpoint/resume 和小型 CPU/GPU smoke，才再次租卡；
4. 只有 F-Gate non-degradation 通过但仍无 geometry sensitivity，才考虑 T3MI；teacher/MSE 继续后置；
5. 下游/结构 probe 再决定是否需要一个更粗 motif partition，而不是继续搜索连接语法。

下一轮 F-Gate 一张 RTX 4090 足够；若单格 runner、共同合同与合并器均准备完成，可用两张卡各跑一格缩短墙钟，但不会增加科学信息。无需 4/8 卡。

## 6. 证据锚点

- profiler：`tmp/gpu_pipeline_profile_m0_v1_20260808.json`
- v1 manifest：`tmp/gcodec_gate_m0_v1_manifest_20260808.json`
- v2 manifest：`tmp/gcodec_gate_m0_v2_manifest_20260808.json`
- 自动裁决：`tmp/graphports_codec_adjudication_20260808.json`
- 预注册协议与阈值：文档 51
- 长度下界与词表预算：文档 52
- PF-2A 与 F-Gate 顺序：文档 49

这些结果仍是 1% failure screen 的单 seed 机制证据，不是完整 311 万预训练、统计显著性或下游优越性结论。
