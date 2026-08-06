# MoSt-T5 候选实现隔离区

此目录预留给 P0 后得到证据支持的**单模块候选改动**，原始主干保持不变。

允许的首批科学候选范围：稳定 hybrid motif identity codec、token/logical-motif/atom 三域 binding、严格 Phase 2 atom mapping/TaskRouter、单张共享 E3FP table 的 level mean、单 carrier 固定平均融合，以及标准 T5 `identity_recovery_mask`/CE。当前必须形成 A0/A1/M0/M1（atom/motif × no-3D/3D）内部四格；molecule-global broadcast、`InterfaceStateResidual` 和 C3 teacher 均不再是首版依赖。`state_prediction_mask` 与浅层 EMA E3FP-state target 只有在 M1 CE-only 已相对 A1/M0 成立后才能加入。所有路径只通过 `logical_motif_id` 对齐，不得重新依赖 token position。

禁止把整个 `model/`、`dataset/` 或 `tokenization/` 复制到这里。每个候选模块必须：

1. 明确导入未改变的原主干模块；
2. 对固定 golden case 通过 baseline parity，或在有意修复时给出化学真值与允许差异；
3. 产生新的数据/词表/checkpoint manifest，不覆盖旧 release。

整体原则见 [P0 非侵入式验证架构](../docs/remote_project_analysis/15_noninvasive_validation_architecture_and_compute_plan.md)。

三域 codec 与 teacher 设计依据见[文档 35](../docs/remote_project_analysis/35_unified_motif_identity_codec_ema_teacher_and_integrated_roadmap_20260805.md)，当前科学候选、数据与执行总计划见[文档 41](../docs/remote_project_analysis/41_scientific_design_comparison_dataset_and_execution_plan_20260806.md)。原始训练文件保持不变；旧 identity-query attention/online-detach MSE 只作为历史 baseline，新候选不得在其 merged-mask block 上继续叠加补丁。
