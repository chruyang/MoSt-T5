# MoSt-T5 候选实现隔离区

此目录预留给 P0 后得到证据支持的**单模块候选改动**，原始主干保持不变。

允许的首批候选范围：稳定 hybrid motif identity codec、token/logical-motif/atom 三域 binding、严格 Phase 2 atom mapping/TaskRouter、单张共享 E3FP table 的 level mean、molecule-global 与 motif-local 两种无参数归约、单 carrier 固定平均融合，以及标准 T5 `identity_recovery_mask`/CE。只有 motif-local 优于 molecule-global 后，才允许增加一个零初始化、无 bias、低秩的 `InterfaceStateResidual`；它只读取去重 attachment atoms 与 core atoms 的 E3FP mean contrast，不得读取 motif/token identity、任意 anchor ID、邻居身份、bond type、group count 或序号。若该候选为正，必须用保持真实分组大小和空组模式、但由固定 hash 选择 pseudo-attachment atoms 的 `C1-Rpseudo` 排除分组/容量解释。`state_prediction_mask` 与浅层 EMA E3FP-state target 只有在 CE-only 候选胜出后才能加入；不得预先成为首版依赖。所有路径只通过 `logical_motif_id` 对齐，不得重新依赖 token position。

禁止把整个 `model/`、`dataset/` 或 `tokenization/` 复制到这里。每个候选模块必须：

1. 明确导入未改变的原主干模块；
2. 对固定 golden case 通过 baseline parity，或在有意修复时给出化学真值与允许差异；
3. 产生新的数据/词表/checkpoint manifest，不覆盖旧 release。

整体原则见 [P0 非侵入式验证架构](../docs/remote_project_analysis/15_noninvasive_validation_architecture_and_compute_plan.md)。

2026-08-05 起的权威整体路线见 [Motif 身份 codec、motif-native 3D 聚合与条件式 EMA teacher](../docs/remote_project_analysis/35_unified_motif_identity_codec_ema_teacher_and_integrated_roadmap_20260805.md)。原始训练文件保持不变；旧 identity-query attention/online-detach MSE 只作为历史 baseline，新候选不得在其 merged-mask block 上继续叠加补丁。
