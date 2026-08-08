# T3MI 预注册、运行门与正式训练剩余路径

> 日期：2026-08-08
>
> 当前状态：实现与 CPU 回归完成；真实 RTX 4090 runtime smoke PASS；M0-T/M1-T 待执行
>
> 研究角色：PF-2B 单 seed 机制筛选，不是最终预训练、显著性检验或下游优势证据

## 1. 为什么只增加 T3MI

F-Gate 的 fresh pair 表明，零初始化 gated residual 能避免 PF-2A 的 CE 破坏，但普通 15% motif corruption 下 M1-G 的 shuffled-minus-aligned ΔNLL 为 `-0.000893`。模型改善了语言恢复，却没有形成对配对 E3FP 的可测依赖。

此时最小、逻辑最直接的下一问不是增加 MSE、teacher 或新预测头，而是移除主要的 2D 身份捷径：把每个 logical motif 的完整 identity span 都替换为 sentinel，同时保留 motif 顺序和 GraphPorts connection skeleton。T5 仍以标准 sentinel CE 重建 motif 身份；M0-T 与 M1-T 只相差是否执行同一零初始化 gated E3FP 支路。

这延续 T5 的 span-corruption 生成目标，也保留 3D-MolT5/CAMT5 使用标准 T5 CE 主干的可比性；不同之处仅在于本项目把遮蔽单位定义为完整 logical motif identity。它不要求额外 head，也不把连续几何拟合强行并入 CE。[T5](https://www.jmlr.org/papers/v21/20-074.html)、[3D-MolT5 official code](https://github.com/QizhiPei/3D-MolT5/tree/82dbe088e424f19fa713dbd657f5235990bd324f)、[CAMT5 official code](https://github.com/Songhyeontae/CAMT5/tree/5875a0a6d73756b1204a7600af9c0b773d9e1ae3)

## 2. 冻结自变量与数据视图

M0-T/M1-T 共同使用：

- PF-1 run3 的 30,240/3,360 train/dev membership、顺序和 split；
- GraphPorts v1、34,666-token sample-bound union tokenizer 和同一 union-init；
- `microbatch=64, accumulation=2, drop_last=False`，nominal effective batch 128；该合同与 fresh F-Gate 相同；
- AdamWScale、gate absolute Adam group、100-step warmup、cosine、clip=1、BF16、1,000 updates；
- train corruption 随 epoch 确定变化，dev corruption 固定；但 mask probability 固定为 `1.0`，故每条记录全部 logical motif identity 被选中；
- 标准 T5 sentinel CE；无新词表、无新 head、无 teacher、无 MSE/辅助 loss。

M0-T 不执行 geometry path；M1-T 执行 F-Gate。connection skeleton 与 motif order 始终可见。因此该目标的准确名称是“topology-conditioned E3FP-to-motif-identity reconstruction”，不能称为 pure-3D reconstruction。E3FP 本身也携带 2D/局部身份信息，所以即使 T3MI 通过，仍须同 2D、不同 conformer 的独立探针。

## 3. 运行前 GPU 门

真实 run3 smoke 只使用 2 个成员和一个随后丢弃的 optimizer step，结果为：

| 检查 | 结果 |
|---|---:|
| logical motif identities | 19/19 全部 sentinel 化 |
| 原 identity tokens | 19 |
| M0-T/M1-T CE tensors | exact equal |
| zero-gate logits/loss | bitwise equal |
| first backward gate gradient | -2.448668，finite/nonzero |
| 同步 E3FP table gradient L1 | 0.0 |
| discarded step 后 gate logit | 9.1132e-5 |
| E3FP table | bitwise unchanged |
| peak allocated GPU memory | 5,308,870,656 bytes |
| persisted optimizer steps | 0 |

证据为 `tmp/pf2_t3mi_gpu_smoke_df426c8.json`。该 smoke 只证明数据视图、初始化和梯度边界，不进入模型效果比较。

先对 train 中 target 最宽的 32 条真实记录执行一次丢弃式 M1-T forward/backward/optimizer 资源探针：full-mask target 最大 92、encoder input 最大 79，峰值 allocated/reserved 分别为 9,739,185,664 / 9,978,249,216 bytes。该结果表明最初沿用的 `32×4` 过于保守。随后用最宽 64 条重复验证，target 仍最大 92，峰值 allocated/reserved 为 17,750,605,824 / 18,394,120,192 bytes，在 24 GB RTX 4090 上仍保留约 6.2 GB 余量。因此正式协议冻结为与 F-Gate 相同的 `64×2`；effective batch 始终为 128，变化只减少一半 microbatch 循环并提高 GPU 连续计算比例。证据为 `tmp/pf2_t3mi_memory_probe_df426c8.log` 与 `tmp/pf2_t3mi_memory_probe_b64_df426c8.log`。

## 4. 预注册裁决

只使用 step 1,000 完整 dev：

| Gate | 条件 | 目的 |
|---|---:|---|
| 3D efficacy | `NLL(M1-T) / NLL(M0-T) <= 0.98` | 几何至少带来 2% 的 T3MI NLL 改善 |
| accuracy safety | `Acc(M1-T) >= Acc(M0-T) - 0.01` | 防止平均 NLL 掩盖 token 准确率退化 |
| aligned sensitivity | shuffled-minus-aligned `ΔNLL >= 0.01` | 模型确实依赖当前分子的配对 E3FP |
| gate opening | `abs(tanh(alpha_final)) >= 0.001` | 几何支路实际打开 |
| M0 inactive | step 500/1000 gate 精确为 0 | 控制格没有意外执行 3D |

五项全部通过，自动决定为：

```text
retain_f_gate_plus_t3mi_and_proceed_to_same_2d_probe
```

任一项失败，自动决定为：

```text
stop_t3mi_and_revisit_geometry_state
```

失败不会自动授权 teacher/MSE。单 seed 只能做机制筛选，不能声称统计显著或架构优越。

## 5. 若通过，正式训练还差什么

1. **same-2D/不同构象探针：**区分 E3FP 的三维贡献与其内含的二维身份/拓扑贡献；
2. **PF-10：**在 group-complete 嵌套约 10% membership 上复核 winner 与最近因果对照；
3. **全量 support tokenizer：**PF-1 的 34,666 词表仅由 1% cohort 冻结，必须在完整 permitted train support 上重放 SELFIES syntax、port-aware motif census、macro K 与 fallback；
4. **数据源冻结：**正式声明 PCQM4Mv2 permitted 3,360,067 主线，或另行选择 legacy 3,119,714 admitted 主线，禁止混写；
5. **PF-FULL runner：**冻结总 updates/token budget、checkpoint、resume 和单卡/DDP 合同。

3D-MolT5 的公开默认配置使用 65,536 steps、batch 144、accumulation 2；CAMT5 的公开预训练配置使用 100,000 steps、batch 16、accumulation 2。它们只提供量级参考，不能直接替代本项目按最终 token/update exposure 冻结的预算。[3D-MolT5 config](https://github.com/QizhiPei/3D-MolT5/blob/82dbe088e424f19fa713dbd657f5235990bd324f/3d_molt5/configs/default.yaml)、[CAMT5 pretrain config](https://github.com/Songhyeontae/CAMT5/blob/5875a0a6d73756b1204a7600af9c0b773d9e1ae3/config/task/train/pretrain.yaml)

## 6. 时间与资源

- T3MI 单卡顺序 M0-T/M1-T：预计约 25–45 分钟；第二张卡只缩短墙钟，不增加证据；
- 若 T3MI 通过，same-2D probe 与 PF-10 数据准备可并行；32–64 vCPU 对物化更有价值，GPU 不应空等 CPU；
- 最早仍需约 24–48 小时才能满足 PF-FULL 启动条件，这一时间是准入窗口，不是保证；
- PF-FULL 一旦准入，按目前 1,000-update 实测吞吐粗略外推，单 4090 的 65k–100k update 训练约 8–14 小时；正式值须由 PF-10 token/update profiler 冻结；
- 4/8 卡若用于同一模型，必须先实现并验证 DDP；当前“一卡一格”只适用于配对机制实验。

## 7. 实现资产

- T3MI runner：`most_t5_next/p2/run_pf2_t3mi_v1.py`
- paired adjudicator：`most_t5_next/p2/merge_pf2_t3mi_v1.py`
- runtime smoke：`most_t5_next/p2/validate_pf2_t3mi_gpu_smoke_v1.py`
- 共享可变 mask 主干：`most_t5_next/p1/run_pf1_four_grid_v1.py`
- fusion 与 optimizer：`most_t5_next/p2/gated_reference_geometry_fusion_v1.py`、`most_t5_next/p2/run_pf2_gated_fusion_v1.py`
- nmb1 CPU 联合回归：P1+P2 `183/183 PASS`
