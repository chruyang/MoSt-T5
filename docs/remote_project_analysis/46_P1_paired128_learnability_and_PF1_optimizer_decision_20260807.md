# P1 paired-128 learnability 与 PF-1 优化协议裁决（2026-08-07）

状态：**A0/A1/M0/M1 已在同一冻结 8 条 paired records 上各完成 20 次真实 AdamW 更新；四格均能降低自身 CE。该结果只证明短程可优化，不是架构排序或泛化证据。**

## 1. 实验目的与边界

P0 的单批 forward/backward 只证明数据能流过模型；本轮增加最薄的 learnability smoke，回答一个更直接的问题：在相同初始化、相同固定 corruption 和相同小批次上，四个条件是否都能通过真实参数更新降低各自的 loss。

固定设置：

- paired release：`paired-identity-128-v1-run3`；
- frozen membership 前 8 条；
- corruption：epoch 0、seed 0、mask probability 0.15；
- 每个条件独立从同一个 `union-init-128-v1-run1` 加载；
- BF16 autocast、AdamW、weight decay 0、20 steps；
- 每一步重复同一个 minibatch 和同一个 dropout realization；
- 无 scheduler、无 checkpoint、无模型权重保存。

A0/A1 使用相同 CE batch，M0/M1 使用相同 CE batch；A1/M1 的 inherited-E3FP atom rows 仍保持一致。A 与 M 的 target 结构不同，因此禁止比较其绝对 loss。

## 2. 真实 RTX 4090 结果

### 2.1 plain AdamW，learning rate `5e-4`

artifact：

```text
/root/autodl-tmp/most-t5-r1-canary/four-grid-learnability-v1-run1
```

| 条件 | initial | final | minimum@step | initial→final | mean step | peak memory |
|---|---:|---:|---:|---:|---:|---:|
| A0 | 63.8726 | 14.6519 | 13.4213 @ 18 | -77.06% | 0.1288 s | 5.101 GB |
| A1 | 63.7115 | 13.5778 | 13.5778 @ 20 | -78.69% | 0.1279 s | 5.292 GB |
| M0 | 55.7026 | 25.6206 | 25.6206 @ 20 | -54.00% | 0.1138 s | 5.124 GB |
| M1 | 59.5192 | 45.9124 | 40.7023 @ 7 | -22.86% | 0.1149 s | 5.319 GB |

四格 loss 均低于自身初值，梯度逐步 finite 且非零。A0/A1/M0 的下降明显；M1 在第 7 步降至 40.70 后回升并振荡，说明 `5e-4` 的无 warmup plain AdamW 对 motif+3D 条件偏激进，不能直接作为 PF-1 优化配置。

### 2.2 plain AdamW，learning rate `1e-4`

artifact：

```text
/root/autodl-tmp/most-t5-r1-canary/four-grid-learnability-v1-run2-lr1e4
```

| 条件 | initial | final | minimum@step | initial→final | mean step | peak memory |
|---|---:|---:|---:|---:|---:|---:|
| A0 | 63.8726 | 50.5168 | 43.7188 @ 18 | -20.91% | 0.1471 s | 5.101 GB |
| A1 | 63.7115 | 39.1815 | 39.1815 @ 20 | -38.50% | 0.1287 s | 5.292 GB |
| M0 | 55.7026 | 44.0570 | 42.2616 @ 18 | -20.91% | 0.1143 s | 5.124 GB |
| M1 | 59.5192 | 47.7518 | 43.9810 @ 13 | -19.77% | 0.1249 s | 5.319 GB |

较低学习率使四格都保持可学习，M1 的最优点推迟到第 13 步，但 20 步小批次曲线仍有噪声。这说明“给 M1 单独设一个更小学习率”不是当前合理结论；PF-1 应使用四格共享、带 warmup/decay/gradient clipping 的领域内标准训练协议。

## 3. 与官方训练代码的关系

[3D-MolT5 官方代码](https://github.com/QizhiPei/3D-MolT5)的默认预训练配置使用 `AdamWScale + cosine + warmup + gradient clipping`，weight decay 为 0；其微调入口也按总步数设置约 6% warmup。[CAMT5 官方代码](https://github.com/Songhyeontae/CAMT5)的 pretrain/continual-pretrain 配置同样使用 AdamWScale、cosine、warmup 和 gradient clipping，weight decay 为 0。

两者的绝对 base LR、batch 和总步数不同，不能机械复制一个数值；但共同实践足以说明，PF-1 不应继续使用本轮为诊断而设的“plain AdamW + 固定 LR + 无 warmup”。本轮两组 LR 也不构成超参数搜索，只用于识别 M1 的短程稳定性问题。

## 4. 当前科学结论

已支持：

- 四格都存在有效梯度，geometry fusion 没有切断主模型训练；
- motif+3D 在较高固定 LR 下更容易振荡，正式训练需标准 schedule；
- 单张 4090 的 optimizer-state 峰值仅约 5.3 GB，PF-1 不因显存必须使用多卡。

不支持：

- A1 优于 A0、M1 优于 M0，或 motif 优于 atom；
- 3D 能提高泛化；
- `5e-4` 或 `1e-4` 是最终最优 LR；
- 8 条样本的下降幅度可以外推到 PF-1、PF-10 或全量训练。

## 5. PF-1 的最短执行决定

PF-1 继续只运行四个必要条件 A0/A1/M0/M1，单个 paired seed，用于淘汰明显失败方案。四格必须共享同一 optimizer family、base schedule、有效 batch、训练成员和总更新预算；不因本轮 M1 曲线单独调参。

优化协议以 3D-MolT5/CAMT5 的共同做法为起点：AdamWScale、weight decay 0、短 warmup 后 cosine decay、全局 gradient clipping。具体 base LR、warmup 比例和 effective batch 只通过一次 PF-1 吞吐/稳定性 pilot 冻结，不做四格分别调参，也不展开网格搜索。

数据侧先冻结 final-v4 permitted 集上的约 1% connectivity-group 完整子集，目标约 33,600 members；从同一批 SDF records 派生 inherited-E3FP 与 paired A/M records。PF-1 只报告训练/验证 CE、各自学习曲线、实际 corruption coverage、吞吐和一个冻结的 3D-sensitive dev probe，不能写成论文最终结论。
