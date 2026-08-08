# Factorized state adapter P0 实现与 PF-10 放行门（2026-08-08）

## 1. 当前裁决

文档 65 冻结的架构已完成最小 P0 接口实现，但尚未获得效果证据。现行模型明确拆成三种训练视图，而不再把多个目标隐式混入同一次前向：

1. **Grammar view**：遮蔽 motif 身份，并同时遮蔽该 motif 所有已填充的 E3FP L0--L3，使用标准 T5 CE 恢复身份；
2. **State view**：保留身份，每个抽中的 motif 只遮蔽一个 atom row，仅以 L1/L2 categorical CE 恢复状态；
3. **Cross-view**：遮蔽 motif 身份，但保留对齐的 E3FP state，用于检验几何状态是否帮助身份恢复。

三种视图共享同一份 GraphPorts、atom--motif ownership 和 E3FP 行轴；目标函数在接口上互斥，避免 CE、MSE、teacher 和状态损失之间来源不清。当前没有引入 raw-ID MSE 或 EMA teacher。

## 2. 已实现的 P0 组件

### 2.1 Nested-shell-safe state masking

状态遮蔽器保留历史 `legacy_slot` 仅用于旧 G1 回放；正式接口只接受：

- `suffix`：从抽中的层开始遮蔽后续 shell；
- `atom_row`：遮蔽抽中原子的完整已填充 state row；
- `motif_block`：遮蔽抽中 motif 内所有原子的完整已填充 row。

正式默认采用 `motif_atom_row`：仅在含至少两个 eligible atoms 的 motif 中抽样，再均匀选择一个 atom row。这样每个 target 均保留至少一个真实的同 motif L1/L2 peer，并避免同 motif、同 attachment-role 的多个被遮蔽原子得到完全相同输入却承担不同目标。`atom_row`、`suffix` 和 `motif_block` 仅保留在底层 builder 中作历史回放或诊断；`motif_block` 不进入主训练接口，直到存在稳定的 per-atom 2D role/identity sidecar 或 set-matching 目标。

Grammar view 会遮蔽目标 motif 的全部已填充 L0--L3，防止通过更高层递归 shell 直接看到被遮蔽的 L1/L2。State adapter 本身只消费 L1/L2；L0 身份层和 L3 高阶层保留在数据产物中用于可追溯性与后续消融，不作为当前正式几何输入。

### 2.2 MotifGeometryAdapterV1

适配器采用两段式接口：

1. `encode`：由完整 motif identity span 构造 query，只对属于该 motif 的 atom-state memory 做受约束 cross-attention，并将结果通过零初始化 ReZero residual 写回唯一 motif carrier；
2. T5 encoder 处理融合后的序列；
3. `decode_state`：显式读取 post-T5 motif-carrier hidden state，与 owned atom memory、level embedding 联合预测 L1/L2 categorical state。

这避免了两个不优雅的替代方案：一是将 motif 内 E3FP 简单求和后直接加到 token；二是让 state head 只读 pre-T5 几何向量、实际上绕开语言模型。首版不加入 topology bias、connection attention 或额外 teacher，以保持单一可归因变量。

### 2.3 Production batch contract

新的 collator 从既有 production record 中保留并 padding：

- `atom_to_motif[B,A]`；
- `motif_mask[B,M]`；
- `motif_to_carrier[B,M]`；
- `identity_span_bounds[B,M,2]`；
- `e3fp_mask_token_id` 与状态域合同。

所有视图统一校验 tokenizer contract、snapshot 和词表边界。模型进一步校验 collator 与 adapter 的 E3FP domain 一致，避免 mask token 被误当作真实 state ID。

## 3. 当前实现证据

- P0 核心定向测试共 **50/50 PASS**，覆盖遮蔽语义、每 motif 单一 target-atom 可识别性、adapter ownership、post-T5 state dependency、三视图 wrapper、确定性共同初始化、production collator、Morgan-2D control、canonical-local matched donor 及原子轴反转不变性；PF-10 split-preservation 另有纯函数/profile 合同测试。
- 从 run3 paired release 读取真实 8 条记录，Grammar/State/Cross-view 三条数据流均通过；State forward 得到有限 loss，logits 为 `[8,18,2,4096]`；进一步排除无同 motif state peer 的目标后，实际 state targets 为 12、完整 state corruptions 为 22。
- 对 run3 全 33,600 条记录的只读 census 显示：单 motif identity span 最大 76，因此 adapter 默认容量由 64 调整为 128，PF-10 仍须重新做全量上限门；241,739 个含 L1/L2 target 的 motif 中，63,040 个含至少一个可见 peer，覆盖率 26.08%。正式 S-stage 只使用这部分可识别 motif，其余 motif 仍参加 Grammar/Cross-view，但不承担 state-imputation target。
- 该结果只证明接口和真实数据流闭合，不证明 F3D 优于 B2D/B0，也不构成正式训练放行。

## 4. B2D 对照的定义

B2D 使用 RDKit Morgan atom environments，固定：radius=3、fpSize=4096、includeChirality、useBondTypes、includeRedundantEnvironments。它从 persisted SELFIES 解码后的同一 model-atom row axis 产生四槽离散状态，不读取坐标。

其角色不是另一个候选主模型，而是检验 E3FP 增益是否超出 2D 身份、拓扑和哈希容量。正式 adapter 对 B2D/F3D 均只读对应的 L1/L2 槽；B0 则关闭 state adapter，作为自然 topology-only baseline。

## 5. PF-10 数据与训练门

PF-10 应从 final-v4 permitted members 3,360,067 中按 connectivity group 和固定 PCG64 seed 20260807 取 whole-group prefix，目标为 `floor(0.10 * 3,360,067) = 336,006`。扩展时必须保留 run3/PF-1 已有 connectivity group 的 train/dev 角色，只对新增 group 分配剩余 dev 名额；禁止重新抽签导致已有样本换 split。

预注册资源一致协议：

- microbatch 64，gradient accumulation 2；
- `drop_last=False`，允许短尾 microbatch 与 update 跨 epoch，禁止固定丢弃尾部成员；
- 10,000 optimizer updates，warmup 1,000；
- eval：0/2,500/5,000/7,500/10,000；
- checkpoint：5,000/10,000；
- 单 seed 只作架构失败屏，不宣称显著性。

## 6. 最小实验矩阵与裁决

| Cell | 输入状态 | 目的 |
|---|---|---|
| B0 | 无 state adapter | topology/identity-only 基线 |
| B2D | Morgan 2D state | 控制身份、拓扑与离散哈希容量 |
| F3D | aligned E3FP state | 主候选 |
| F3D-zero | 将 state memory 置零 | 检验是否只靠新增参数 |
| F3D-shuffle | 同签名 motif 间错配 state | 检验对应状态是否被使用 |

主裁决必须同时满足：F3D 相对 B2D/B0 改善；zero 后增益消失；matched shuffle 产生可重复的性能恶化。只看到 F3D CE 较低而 shuffle/zero 不敏感，仍判定为使用 identity-conditioned prior，不能声称利用了对应 3D 状态。

matched donor 的主签名冻结为 `(exact motif identity, atom count, port-degree multiset, incident bond-type multiset)`；邻域 identity 和 remote-port pattern 用作更严格敏感性层。必须报告覆盖率与未匹配类别，不能静默退化为全局随机错配。

### 6.1 两段训练预算

为避免把 state CE 与 grammar CE 主观相加，PF-10 采用两段式而非每步混合损失：

1. **S-stage（state imputation）**：B2D/F3D 各 2,500 updates，T5 参数冻结，只训练共同初始化的 motif-state adapter 与 state head；正式遮蔽为 `motif_atom_row`，每个抽中 motif 至多一个 target atom。B0 不参与此阶段。该阶段约等于一次 train-member exposure，用于确认状态通道可学习，而不是生成能力排名。
2. **G-stage（grammar bridge）**：B0/B2D/F3D 各 10,000 grammar CE updates；B0 从共同 union-init 的 raw T5 开始，B2D/F3D 从各自 S-stage adapter 状态开始，T5 与 adapter 联合训练。三格使用相同 grammar 成员顺序、mask、更新数与优化协议。

主容量归因是 B2D 对 F3D，因为两者参数、S/G 更新预算和初始化完全一致。B0 是自然 topology-only 参考，不宣称与带 adapter 的条件参数量完全匹配；F3D-zero 单独承担“新增 adapter 参数本身”的诊断。S-stage 与 G-stage 分开报告，不把不同目标的 loss 相加或跨任务直接比较。

## 7. 下一步与资源

在 GPU 前需完成：PF-10 membership、B2D overlay、F3D matched-donor overlay、128 条三视图/三状态源 smoke，以及单张 4090 的 3-update 显存与吞吐探针。

PF-10 数据物化适合 16--32 vCPU、80--120 GB 内存，预计 1--2 小时；当前 0.5 vCPU 无卡实例不适合正式物化。GPU gate 不需要 8 卡或 DDP；三张独立 4090 各跑 B0/B2D/F3D 最简洁，估计 2--3 小时，单卡顺序约 5--7 小时。

## 8. 尚未完成

- PF-10 正式 membership/overlay 尚未生成；
- 正式 GPU runner 尚未接入三视图新 wrapper；
- matched-shuffle 覆盖率尚未实测；
- geometry-sensitive endpoint 尚未给出阳性结果；
- 多卡 state-loss 全局 target-count 归一化需在未来 DDP full run 前实现，本轮独立单卡 cells 不受影响。

因此当前状态是：**P0 接口闭合，进入 PF-10 数据与因果对照准备；尚未放行全量预训练。**
