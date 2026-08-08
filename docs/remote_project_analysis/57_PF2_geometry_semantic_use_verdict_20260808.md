# PF-2 几何语义使用裁决：从隐式 CE 转向显式 motif 对齐

> **后续路线更新（2026-08-08）：** 本文第 1–3、5 节的实验事实与 MSE
> 边界仍有效；第 4、6 节“InfoNCE 优先、暂不做 same-2D conformer probe”的
> 执行建议已由文档 58 取代。新裁决优先采用可学习的置换不变 motif 几何编码、
> E3FP categorical state prediction 与 geometry→motif generation，并把
> same-2D multi-conformer sensitivity 作为扩容前门禁。

## 1. 已完成实验

### 1.1 T3MI：拓扑可见、全部 motif identity 遮蔽

同一 33,600-member cohort、64x2、1000 updates：

| 指标 | M0-T | M1-T |
|---|---:|---:|
| dev NLL | 1.794966 | 0.973769 |
| masked-token accuracy | 0.592885 | 0.759984 |
| final gate | 0 | -0.067486 |

表面上 M1-T 明显改善，但 same-atom-count shuffled E3FP 的
`shuffled - aligned ΔNLL=0.0000158`，未达到预注册0.01门槛。

### 1.2 最终权重无训练扰动

| 输入变体 | dev NLL | 相对 aligned |
|---|---:|---:|
| aligned | 0.9737703 | 0 |
| gate=0 | 0.9737701 | -0.0000002 |
| shell occupancy-only | 0.9737637 | -0.0000067 |
| molecule-internal atom-row rotation | 0.9738204 | +0.0000501 |
| random E3FP IDs | 0.9737583 | -0.0000120 |

E3FP carrier自身没有塌缩：aligned/shuffled carrier平均余弦仅0.079，平均
相对L2差异1.346；公共均值仅占motif carrier能量3.92%。因此模型忽略的是
E3FP内容，不是因为输入向量都变成相同常数。

### 1.3 PF-2C：冻结 M0-T，只训练 E3FP table + scalar gate

| 指标 | frozen M0-T | PF-2C adapter |
|---|---:|---:|
| dev NLL | 1.794966 | 1.795318 |
| accuracy | 0.592885 | 0.592545 |
| shuffled - aligned ΔNLL | — | -0.0000694 |
| final gate | 0 | -0.034396 |

冻结T5后，adapter不能改善基线，也没有配对敏感性。故T3MI中M1的优势是
几何分支改变全模型优化轨迹所产生的正则化效应，不能解释为3D语义增益。

## 2. E3FP reduction 信息检查

训练集217,784个motif中，当前归一化E3FP-ID袋的身份冲突仅0.011%；但
dev exact-signature覆盖仅5.13%。加入motif原子数、shell层级或完整atom-row
分组后仍约5.12%。这说明：

1. 两次均值没有造成大量离散碰撞；
2. E3FP组合高度样本/构象特异，1%数据中多数dev组合未原样出现；
3. 仅靠motif identity CE没有建立“该hidden必须对应该E3FP”的约束。

该统计不包含GraphPorts上下文，且exact unseen按错误计，因此不是模型精度
上限；其用途仅是排除“加一个attention pooling即可自动解决”的武断结论。

## 3. 与参考实现的关系

- 3D-MolT5仍以T5 CrossEntropyLoss为生成损失，并把E3FP送到原子SELFIES
  carrier；它没有证明motif均值状态会被CE自动使用。
- FineMolTex预训练显式提供InfoNCE：归一化两模态表示，以batch diagonal为
  正样本并用CrossEntropyLoss训练，同时再做mask任务。这直接支持“生成CE +
  显式跨模态对齐”，而不是把全部对齐责任交给生成CE。
- FineMolTex中的MSE用于连续视觉/表示回归路径，不能支撑对无序E3FP hash ID
  直接做数值MSE。

对应代码证据：

- `reference_repos/3D-MolT5_official_src_82dbe088/3d_molt5/utils/FPT5ForConditionalGeneration.py`
- `reference_repos/FineMolTex_official_src_c976faa/scripts/pretrain.py` 的
  `do_CL(..., SSL_loss='InfoNCE')` 与双向contrastive训练段。

## 4. 下一步最小方案

下一机制实验改为 motif-level paired alignment：

1. 主生成损失仍为标准T5 CE；
2. 从encoder的motif carrier hidden得到 `z_m`；
3. 从对应motif的E3FP atom states得到 `z_g`；
4. 两侧各用一个小投影并L2 normalize；
5. 用对称InfoNCE区分batch内matched与mismatched motif；
6. 最终仍用same-atom-count shuffle ΔNLL/对齐检索判断真实配对敏感性。

第一步只做冻结M0-T的alignment probe，确认E3FP与motif hidden在当前cohort上
能否建立可泛化配对；通过后才把对齐项与生成CE联合。这样不会立即引入大规模
联合训练或大量超参数。

## 5. 关于 MSE

当前证据不支持 `MSE(raw E3FP ID, hidden)`：hash ID没有顺序距离，ID 100与101
并不比100与300更接近。若后续使用MSE，目标必须是连续、归一化、stop-gradient
的3D teacher表示，并配独立projection head；其应作为InfoNCE之外的受控对比，
而不是当前主线补丁。

## 6. 当前裁决

- 不进入same-2D conformer probe；
- 不开始最终全量预训练；
- 不因T3MI表面NLL改善宣称3D增益；
- 不引入raw-ID MSE；
- 下一门为冻结骨干的motif-E3FP显式对齐probe。
