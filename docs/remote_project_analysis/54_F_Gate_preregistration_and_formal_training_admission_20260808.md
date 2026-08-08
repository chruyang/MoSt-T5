# F-Gate 预注册与正式训练准入边界

> 日期：2026-08-08
>
> 当前状态：F-Gate 实现、P1+P2 无卡回归与真实 RTX 4090 runtime smoke 已闭合；M0-G/M1-G 可启动
>
> 研究范围：GraphPorts v1、PF-1 run3 的 1% failure screen；不是 PF-10、PF-FULL 或下游优势结论

## 1. 为什么还不能直接启动数百万规模预训练

当前已经解决的是数据流、可逆 codec、训练稳定性和 1% 机制筛选，不是最终方法已经确定。已有证据依次表明：

1. 原始四表求和、直接残差注入在 PF-1 中很快失去 E3FP 敏感性；
2. 3D-MolT5 风格共享表、固定 `0.5 identity + 0.5 geometry` 的 PF-2A 严重损害 CE，M1 最终 NLL 为 M0 的 3.456 倍；
3. GraphPorts v2 虽缩短 37.49% encoder tokens，却使配对 M0 的最终 NLL 恶化 7.28%，因此 GraphPorts v1 已被正式保留；
4. 尚未独立回答的是：PF-2A 的失败主要来自 E3FP 无效，还是来自在训练开始时强制注入随机几何向量。

如果不回答第 4 点便启动 300 万量级预训练，模型可能再次用大量预算学会忽略几何支路。因此 F-Gate 是进入 PF-10 前最后一个必要的 1% 机制门，而不是继续无边界搜索架构。

## 2. 唯一自变量：零初始化门控残差

M0-G 与 M1-G 使用相同的：

- GraphPorts v1 paired release；
- 30,240/3,360 train/dev membership、顺序、mask 与 CE targets；
- union tokenizer、union-init、T5 初始化和 geometry embedding 初始化；
- AdamWScale、warmup/cosine、clip、BF16、1,000 updates；
- `microbatch=64, accumulation=2, drop_last=False` 的可变尾批合同。

两格都实例化相同参数结构；M0 不执行 geometry path，M1 的 carrier 注入为：

\[
g_i=\operatorname{mean}_{a\in i}\left(\operatorname{mean}_{s=1}^{4}E(f_{a,s})\right),
\qquad
h'_i=h_i+\tanh(\alpha)g_i,
\qquad \alpha_0=0.
\]

其中只有一个 shape `[1]` 的标量 gate；不加入 projection、LayerNorm、teacher、MSE 或辅助 loss。`alpha=0` 时 M1 在函数上与 M0 相同，首次反向只更新 gate，E3FP 表在 gate 打开前梯度严格为零。

这不是主观补丁。Flamingo 使用 zero-initialized `tanh` gates 将新增视觉 cross-attention/residual 安全接入冻结语言模型，并报告移除门控会损害稳定性；LLaMA-Adapter 同样用零初始化 attention gate 避免新增模态在训练初期扰动语言模型。[Flamingo](https://arxiv.org/abs/2204.14198)、[LLaMA-Adapter, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/c196239c5f9481e0db2755f31fe4585f-Abstract-Conference.html)。本项目只借用“安全注入”原则，不把它们当作分子 3D 有效性的证据。

代码还与固定上游实现逐行对照：OpenFlamingo commit [`655f693`](https://github.com/mlfoundations/open_flamingo/commit/655f693fbfa04cd6e9a987d960654624d48917cf) 和 LLaMA-Adapter commit [`521a09d`](https://github.com/OpenGVLab/LLaMA-Adapter/commit/521a09da84f70f6913d54b7421afa24010319e47) 均以 `tanh(gate)` 缩放新增支路。3D-MolT5 支持 E3FP embedding summation 的领域先例，但没有验证 motif pooling 后固定 0.5 注入一定稳定，因此不能机械照搬。[3D-MolT5, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/3d4dc72d715bd6415d356293079adf3d-Paper-Conference.pdf)

## 3. GPU 前的硬门

正式 M0-G/M1-G 前只运行一个 2-member、零 optimizer step 的真实 run3 smoke，并要求：

1. M0/M1 collator 的 CE tensors 完全相同；
2. 两个 wrapper 的完整初始化 state 逐 tensor 相同；
3. eval 模式下零 gate 的 M0/M1 logits 与 loss bitwise equal；
4. M1 首次 backward 的 gate gradient finite 且非零；
5. 同一 backward 中 E3FP table gradient L1 必须严格为 0；
6. 不写训练 checkpoint、不改变任何参数。

实现入口为 `most_t5_next/p2/validate_pf2_gated_fusion_gpu_smoke_v1.py`。该 smoke 只验证接口，不参与模型选择。

2026-08-08 在 nmb1 的真实 RTX 4090、PyTorch 2.1.0+cu118、BF16 上已执行并通过：

| 检查 | 结果 |
|---|---:|
| real run3 members | 2 |
| M0/M1 CE tensors | equal |
| 完整初始化 state | equal |
| zero-gate logits/loss | bitwise equal |
| 首次 gate gradient | 45.2913，finite/nonzero |
| 同次 E3FP table gradient L1 | 0.0 |
| optimizer steps | 0 |
| peak allocated GPU memory | 3,219,304,960 bytes |

证据为 `tmp/pf2_f_gate_gpu_smoke_v1_20260808.json`。因此 GPU runtime admission 已关闭；上述 loss=62.575 只是未训练模型的 smoke 数值，不进入科学比较。

## 4. F-Gate 的预注册裁决

fresh M0-G 与 M1-G 均完成 1,000 updates 后，以 step 1,000 完整 dev 为准：

| Gate | 通过条件 | 含义 |
|---|---:|---|
| CE NLL 非劣 | `NLL(M1) <= 1.02 × NLL(M0)` | 不重复 PF-2A 的语言恢复破坏 |
| masked-token accuracy 非劣 | `Acc(M1) >= Acc(M0) - 0.01` | 防止只靠 NLL 平均掩盖退化 |
| geometry sensitivity | shuffled-minus-aligned `ΔNLL >= 0.01` | 模型没有再次完全忽略 E3FP |
| M0 inactive gate | step 500/1000 gate 均精确为 0 | 控制格没有意外执行 3D 支路 |

自动决策只有三条：

```text
三门都通过
  -> 保留 F-Gate，进入 PF-10 前的 3D-sensitive probe

CE 两门通过、sensitivity 不通过
  -> 只进入一次预注册 T3MI 机制实验；仍不启动 PF-FULL

任一 CE 门失败
  -> 停止扩容，重新检查 geometry state / carrier interface
```

单 seed 只作机制筛选，不作统计显著性或架构优越性主张。teacher/MSE 不允许由本轮单独触发。

## 5. “正式训练”必须先明确是哪一个数据源

当前 1% run3 来自 PCQM4Mv2 clean/permitted 主线，不能在论文或配置中悄然称为旧的 3D-MoLM `pretrain=3,119,717`：

| profile | 当前可用口径 | 角色 |
|---|---:|---|
| PCQM4Mv2 paper-scope permitted | 3,360,067 | 当前候选主线；具有 DFT equilibrium geometry 与 3D-MolT5 直接先例 |
| legacy 3D-MoLM/3D-MoIT official pretrain | 3,119,717 | 原始 split 身份集合 |
| legacy geometry-admitted | 3,119,714 | 当前 codec 可接收集合；另有 3 条 H2 unsupported |

建议保持 PCQM4Mv2 为主线，并在 PF-10 做一次同 tokenizer、同 token/update budget 的 legacy source control；若最终改回 legacy，则必须把正式分母写成 3,119,714，或先为 3 条 H2 明确定义支持策略。PCQM4Mv2 的来源与已知 SDF/SMILES 映射边界以 OGB 官方说明为准：[OGB PCQM4Mv2](https://ogb.stanford.edu/docs/lsc/pcqm4mv2/)。

因此本项目后续必须区分：

- **F-Gate/PF-1：**1% 机制筛选；
- **PF-10：**约 10% 的架构与数据源确认；
- **PF-FULL：**冻结 winner、词表、membership、优化协议后才启动的正式 Phase-1 预训练。

## 6. 进入 PF-FULL 前仍需闭合的最小条件

1. **F-Gate：**真实 GPU smoke 与 M0-G/M1-G 自动裁决通过；
2. **3D-specificity：**至少一个保持 2D identity 的 conformer/geometry perturbation probe，避免把 E3FP 的 2D 拓扑部分误写成构象理解；
3. **PF-10：**winner 与最近因果对照在嵌套 10% membership 上确认，不允许 PF-1 排序直接外推；
4. **数据源：**冻结 PCQM mainline 或 legacy mainline，禁止混写 3,360,067 与 3,119,717；
5. **最终 tokenizer：**在完整 permitted training support 上重放 SELFIES syntax 与 port-aware motif census，冻结 K、macro/fallback 和 snapshot；PF-1 的 34,666-token 词表仍是 sample-bound；
6. **正式训练 runner：**冻结 total updates/token budget、checkpoint/resume、单卡或 DDP 合同。当前单条件 runner 可单卡运行；如果使用 4/8 卡加速 PF-FULL，必须先完成多卡数据并行与合并恢复门，不能把“一卡一格”误当作同一模型 DDP。

## 7. 从当前单卡到正式启动的现实时间

在代码与数据路径均已就绪的前提下：

| 时间窗 | 工作 | 资源 |
|---|---|---|
| 单卡恢复后 5–10 分钟 | real run3 F-Gate smoke | 1×4090 |
| 随后约 20–30 分钟 | 顺序 M0-G/M1-G 1% screen + merge | 1×4090；2 卡只缩短墙钟，不增加证据 |
| 接下来约 6–18 小时 | 冻结 PF-10 membership、完整-support tokenizer candidate、10% paired materialization | 优先 32–64 vCPU；GPU 可并行空闲 |
| 再约 2–8 小时 | PF-10 winner/nearest-control 与 3D-sensitive probe | 1–2×4090；协议确定后才考虑更多卡 |
| 最早约 24–48 小时 | 若上述门全过，发布 PF-FULL admission 并启动正式训练 | 单卡可启动；4/8 卡需先闭合 DDP runner |

如果 F-Gate 只达到“CE 安全但不使用 3D”，则增加一次 T3MI 机制门，正式启动至少顺延一轮；如果 F-Gate 仍破坏 CE，则不应靠扩大数据量掩盖接口问题。

这里的“24–48 小时”是**开始正式训练的最早条件时间**，不是承诺 24–48 小时内完成 PF-FULL。正式全量训练耗时还取决于最终 updates/token budget、单卡还是 DDP 以及完整 materialization 吞吐。

## 8. 当前执行资产

- F-Gate fusion：`most_t5_next/p2/gated_reference_geometry_fusion_v1.py`
- 单格 runner：`most_t5_next/p2/run_pf2_gated_fusion_v1.py`
- paired adjudicator：`most_t5_next/p2/merge_pf2_gated_fusion_v1.py`
- real GPU smoke：`most_t5_next/p2/validate_pf2_gated_fusion_gpu_smoke_v1.py`
- CPU regression：新增模块与既有 P1/P2 联合 `171/171 PASS`
- real GPU smoke：`tmp/pf2_f_gate_gpu_smoke_v1_20260808.json`，status=`pass`
- run3 paired release、union-init 与 base T5 均已在 nmb1 保留，无需重新下载或重建

当前 nmb1 已可见 1×RTX 4090（24,564 MiB），第 3 节 smoke 已通过。下一动作是先提交并固定本轮源码，再顺序运行 fresh M0-G/M1-G；不再插入新的架构候选。
