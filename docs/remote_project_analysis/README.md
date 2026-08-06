# MoSt-T5 远端项目分析文档

> 建档日期：2026-07-17  
> 远端范围：`/root/autodl-tmp` 与 `/root/autodl-tmp/MoSt-T5`  
> 本地源码范围：当前工作区 `F:\the basis of pytorch\my_code`  
> 操作边界：早期盘点以只读为主；自用户授权执行 P0/R0/R1 后，新增文件只写入独立版本目录和持久化证据目录。未批量删除、移动或覆盖原始项目与数据。

## 文档目的

本目录用于持续回答四类问题：

1. 远端有哪些代码、数据、模型和实验结果，它们之间是什么关系？
2. MoSt-T5 的核心思路和完整数据流是什么？
3. 当前实验能否证明思路有效，哪些结论仍缺少证据？
4. 后续应优先修复什么、补做什么实验，才能低成本判断可行性？

## 文档索引

| 文档 | 内容 | 当前状态 |
|---|---|---|
| [01_remote_storage_inventory.md](01_remote_storage_inventory.md) | 远端目录、容量、数据集、模型与环境资产清单 | 已完成首轮盘点 |
| [02_codebase_catalog.md](02_codebase_catalog.md) | 代码目录、主要文件、入口脚本与职责边界 | 已完成架构级整理 |
| [03_architecture_training_pipeline.md](03_architecture_training_pipeline.md) | 模型结构、tokenizer、数据流、Phase 1/2 训练目标和下游任务 | 已完成首轮详细分析 |
| [04_feasibility_risks_improvements.md](04_feasibility_risks_improvements.md) | 可行性判断、已证实风险、推断风险和改进优先级 | 已完成首轮评审 |
| [05_evidence_and_open_questions.md](05_evidence_and_open_questions.md) | 已核验证据、版本差异、尚未回答的问题 | 持续更新 |
| [06_next_validation_plan.md](06_next_validation_plan.md) | 后续最小成本验证路线 | 可直接执行 |
| [07_overall_code_implementation_analysis.md](07_overall_code_implementation_analysis.md) | 从数据准备到下游评估的整体代码实现、调用链和设计判断 | 已完成首轮总览 |
| [08_literature_hierarchy_and_evidence_matrix.md](08_literature_hierarchy_and_evidence_matrix.md) | 按总体—局部—细节拆分代码操作，并标注文献证据等级、可行性边界和必要消融 | 已完成首轮文献审查 |
| [09_literature_catalog_and_gap_log.md](09_literature_catalog_and_gap_log.md) | 本地与补充论文目录、页级定位、下载异常和证据缺口 | 已完成首轮索引 |
| [10_core_questions_3d_loss_and_motif_partition.md](10_core_questions_3d_loss_and_motif_partition.md) | 专项审查 E3FP/3D Loss 的成立条件、mask 泄漏和当前 motif 划分的合理性 | 已完成专项分析 |
| [11_t5_ce_mse_compatibility_and_precedents.md](11_t5_ce_mse_compatibility_and_precedents.md) | 聚焦 T5 中 CE+MSE 的结构兼容性、梯度风险、分子领域先例和最小验证方案 | 已完成专项分析 |
| [12_elegant_mse_integration_design.md](12_elegant_mse_integration_design.md) | 将当前 MSE 重构为 EMA teacher 驱动的 3D masked latent prediction，并给出 mask、权重、DDP 和实验方案 | 历史候选；现行路线先跑简洁 E3FP+CE，teacher 仅条件启用 |
| [13_mainline_evidence_adjudication_and_top_tier_readiness.md](13_mainline_evidence_adjudication_and_top_tier_readiness.md) | 主干审稿裁定：核验 CE+MSE 直接先例，审计当前 target 语义，并给出顶刊证据门槛与接受条件 | 2026-07-30 更新 |
| [14_P0_data_integrity_lineage_and_tensor_flow_plan.md](14_P0_data_integrity_lineage_and_tensor_flow_plan.md) | P0 发布前验收：数据完整性、E3FP 血缘、原子/motif 映射、Phase 1/2 张量流、模型 smoke test 与启动配置 | 2026-07-30 新增 |
| [15_noninvasive_validation_architecture_and_compute_plan.md](15_noninvasive_validation_architecture_and_compute_plan.md) | P0 的主干旁路验证、候选模块隔离、3D-MolT5 工程对照与按 profile 决策的 GPU 资源阶梯 | 2026-07-30 新增 |
| [16_autodl_fs_migration_handoff.md](16_autodl_fs_migration_handoff.md) | AutoDL 数据盘到跨实例持久化文件存储的迁移范围、校验和交接 | 已执行 |
| [17_P0_execution_record_and_release_gates.md](17_P0_execution_record_and_release_gates.md) | P0 实际执行记录、数据/张量流门禁和放行边界 | 持续更新 |
| [18_phase1_restructuring_decision.md](18_phase1_restructuring_decision.md) | 第一阶段结构基础预训练、数据谱系与一次性固定词表的重构决策 | 已完成决策稿 |
| [19_R0_phase1_membership_audit.md](19_R0_phase1_membership_audit.md) | Phase-1 membership 与数据流准入审计 | R0 证据 |
| [20_R0_phase2_membership_and_singleton_policy.md](20_R0_phase2_membership_and_singleton_policy.md) | Phase-2 membership 与单构象策略审计 | R0 证据 |
| [21_R0_deterministic_tokenizer_contract.md](21_R0_deterministic_tokenizer_contract.md) | 可复现 tokenizer 合同与不扩词表边界 | 合同已形成，尚未最终绑定 |
| [22_R0_dataflow_and_release_decision.md](22_R0_dataflow_and_release_decision.md) | R0 实际数据流、release 结论与主干裁定 | 已完成 |
| [23_R1_pcqm4mv2_provenance_and_admission.md](23_R1_pcqm4mv2_provenance_and_admission.md) | PCQM4Mv2 来源、3D-MolT5 对照与准入起点 | 已完成 |
| [24_R1_pcqm_identity_and_normalization_gate.md](24_R1_pcqm_identity_and_normalization_gate.md) | 2D/3D 身份关联、氢规范化和拒绝策略 | 已执行门禁 |
| [25_R1_CPU_cross_region_full_release_execution.md](25_R1_CPU_cross_region_full_release_execution.md) | 96 vCPU 全量几何 release、跨区同步与双端独立审计 | release/transfer PASS，P1=false |
| [26_R1_tokenizer_binding_and_P1_candidate_spec.md](26_R1_tokenizer_binding_and_P1_candidate_spec.md) | motif digest 到固定 token 的绑定边界与 4090 最小候选门禁 | 规格完成，待执行 |
| [27_R1_membership_and_overlap_proof_gate.md](27_R1_membership_and_overlap_proof_gate.md) | P1/P2 membership、跨阶段 overlap proof 与 tokenizer 准入边界 | 门禁规格完成 |
| [28_R1_pause_checkpoint_20260805.md](28_R1_pause_checkpoint_20260805.md) | R1 暂停点、远端资产与可恢复执行状态 | 检查点已记录 |
| [29_R1_semantic_recompute_and_worker_IPC_boundary_20260805.md](29_R1_semantic_recompute_and_worker_IPC_boundary_20260805.md) | motif 语义复算与 CPU worker IPC/内存边界 | 已形成执行边界 |
| [30_R1_P1_P2_identity_overlap_facts_and_parallel_extraction_20260805.md](30_R1_P1_P2_identity_overlap_facts_and_parallel_extraction_20260805.md) | P1/P2 身份重合事实与并行提取验证 | 全量事实已记录 |
| [31_R1_P2_motif_census_and_projection_compatibility_20260805.md](31_R1_P2_motif_census_and_projection_compatibility_20260805.md) | P2 motif census、P1/P2 投影兼容性与并行基线验收 | compatibility=false，freeze=false |
| [32_downstream_dedup_minimal_ablations_and_anchor_decision_20260805.md](32_downstream_dedup_minimal_ablations_and_anchor_decision_20260805.md) | 3D-MolT5 下游、评估防泄漏、资源最小消融、CAMT5 对照与锚点因子化裁定 | 主干裁定完成，待执行 CPU 门禁 |
| [33_downstream_portfolio_anchor_factorization_and_vocab_policy_20260805.md](33_downstream_portfolio_anchor_factorization_and_vocab_policy_20260805.md) | 下游任务重排、3D-MolT5 去污染证据、锚点创新边界与 motif 词表频率—覆盖率政策 | 文献复核和远端 census 实验完成，待 molecule-level 门禁 |
| [34_lossless_fallback_motif_editing_3dmotif_and_downstream_freeze_20260805.md](34_lossless_fallback_motif_editing_3dmotif_and_downstream_freeze_20260805.md) | 无损回退边界、FineMolTex motif 编辑定位、3D motif 证据链与下游/词表冻结时序 | 主干裁决完成，P1 仍待 codec 与保护集门禁 |
| [35_unified_motif_identity_codec_ema_teacher_and_integrated_roadmap_20260805.md](35_unified_motif_identity_codec_ema_teacher_and_integrated_roadmap_20260805.md) | motif 身份无损 codec、motif-local E3FP 聚合与接口角色残差、条件式 EMA teacher、logical motif 三域接口及整合路线 | 三域 codec/teacher 设计依据；候选优先级见文档 41 |
| [36_P1_multifidelity_proxy_subset_and_training_gates_20260806.md](36_P1_multifidelity_proxy_subset_and_training_gates_20260806.md) | PF-CANARY/1%/10%/全量嵌套代理子集、固定 manifest、公平 token 预算与逐级晋级门禁 | PF 层级定义保留；候选矩阵见文档 41 |
| [37_overall_readiness_assessment_and_execution_roadmap_20260806.md](37_overall_readiness_assessment_and_execution_roadmap_20260806.md) | 从科学命题、数据/代码成熟度、因果实验、下游角色和单 4090 资源边界统一评估项目，并给出 R1 收口至论文证据的关键路径 | 2026-08-06 成熟度快照；当前总路线见文档 41 |
| [38_compute_resource_switch_decision_20260806.md](38_compute_resource_switch_decision_20260806.md) | 现有 16 vCPU、条件式 96 vCPU、单卡 canary/PF-1 和 4/8 卡 PF-10/FULL 的切换门槛、I/O 与成本边界 | 当前资源租用裁决 |
| [39_execution_start_and_remote_runtime_checkpoint_20260806.md](39_execution_start_and_remote_runtime_checkpoint_20260806.md) | 单卡实例启动后的 cgroup/GPU/存储/环境/持久化资产核验、下游源形态和三条并行执行支线 | 2026-08-06 执行起点 |
| [40_P1_clean_membership_topology_and_ce_canary_checkpoint_20260806.md](40_P1_clean_membership_topology_and_ce_canary_checkpoint_20260806.md) | 当前三任务 clean-v0、32+256 拓扑回放、标准 T5 CE 的 4090 前后向与保存重载，以及进入真实 batch canary 的证据边界 | topology/CE canary PASS |
| [41_scientific_design_comparison_dataset_and_execution_plan_20260806.md](41_scientific_design_comparison_dataset_and_execution_plan_20260806.md) | 以 atom/motif × no-3D/3D 四格收束架构比较，裁定 P1/P2 目标、motif/anchor/vocab、下游组合、数据准备与三天资源计划 | 当前科学执行总计划 |
| [42_R1_downstream_protocol_and_four_grid_research_checkpoint_20260807.md](42_R1_downstream_protocol_and_four_grid_research_checkpoint_20260807.md) | 冻结 3D-MolT5 优先的下游来源、QM9/KPGT/HIV split 工具、level-aware 四格接口、科研 estimand 与 CPU/GPU 门禁 | 高 CPU 放行；GPU PF-CANARY 暂未放行 |
| [43_R1_official_downstream_materialization_and_protected_scope_checkpoint_20260807.md](43_R1_official_downstream_materialization_and_protected_scope_checkpoint_20260807.md) | 正式下游成员、PCQM identity、保护并集与 paper-scope 差集的物化过程和边界 | 历史执行检查点；最终身份口径见文档 44 |
| [44_P0_pre_gpu_data_code_audit_and_handoff_20260807.md](44_P0_pre_gpu_data_code_audit_and_handoff_20260807.md) | 统一化学入口、QM9/HIV v2、final-v4 保护集、motif parseability、E3FP duplicate-shell 与模型 I/O 的上卡前审计 | 当前 CPU→GPU 交接依据 |

## 当前结论摘要

MoSt-T5 的研究方向具有明确合理性：它试图把分子 motif 序列、原子级多层 E3FP 三维特征和自然语言统一到 T5 编解码框架中，并拟通过局部 atom-to-motif 融合、三维掩码重建和多任务训练学习跨模态表示。

早期 checkpoint 的最高优先级问题曾是 legacy tokenizer 通过 `list(set)` 注册 motif，导致 ID 不稳定；R0 已形成确定性 tokenizer 合同，因此它是已定位并有替代路径的历史问题，不再是当前首要阻断项。当前优先级依次是：按显式 duplicate inheritance 重算 E3FP、实现可逆 graph+ports motif codec、从同一 SDF Mol 生产真实 A/M records，以及闭合 Trainer 与 checkpoint/resume。完成这些输入合同前，不能把旧 checkpoint 当作“整体思路已有效”的证据，也不启动 10% 或全量训练。

2026-08-05 历史更新：R0 已形成确定性 tokenizer 合同；R1 已把 3,378,606 条 PCQM4Mv2 train-3D 记录制成 136 个不可变分片，并在 CPU 源端和 4090 区域持久化副本分别通过 v3 独立审计。该结果只放行 production release 与跨区传输，不等于 tokenizer 已绑定或 P1 可训练；当时计划的 CE+MSE 门禁已被后续四格路线取代，现行 GPU-G0 先做 CE-only，不引入 MSE/teacher。

2026-08-06 更新：当时三任务的 `provisional clean-v0` 已派生；冻结 32+256 样本的 topology replay 为 288/288 PASS；标准 T5 CE 已在单张 RTX 4090 上完成 synthetic/contract canary 的前向、反向、一次 AdamW step 和 schema-v2 保存重载验证。这不是尚未放行的 production A0/A1/M0/M1 canary；下一门仍是接入真实 production records。

2026-08-06 科学路线更新：架构选择收束为 A0/A1/M0/M1（atom/motif × no-3D/3D）四格；C1-G、interface residual、legacy MSE 和多种 fusion 不再进入当前主比较。QM9 split、Motif Editing 协议、MoleculeNet 四任务版本及 P2 重线性化先于 full P1 冻结；GPU 只在 production 四格 canary 就绪后启用。

2026-08-07 CPU 审计更新：QM9/HIV 已统一到与 PCQM 相同的显式氢身份投影，QM9 改为 connectivity-group split；最终 paper-scope 保护差集保留 3,360,067 / 3,365,577 个 PCQM members。paper-scope-v2 的全量 motif 审计足以否定删除 anchor 的 pure identity，但 final-v4 的词频与 K 尚待重物化；10% PCQM E3FP 审计已裁决 duplicate shell 使用显式 inheritance，但 payload 尚未重算。下一步是实现 graph+ports codec、同源 A/M producer，并先在 128 条 paired records 上生成 inherited E3FP；完成前不启动 10% 或全量 GPU 训练。

## 维护约定

- 远端实际训练资产是运行证据的权威来源。
- 本地源码便于逐行阅读，但关键文件与远端并非全部一致；涉及历史训练时必须标注使用的是“远端版本”还是“本地版本”。
- 每个判断尽量标注为：`已实测`、`代码直接证据`、`合理推断` 或 `待验证`。
- 不把聚合日志中的最好数字直接当作最终模型指标；必须先确认实验边界、数据划分和 checkpoint 对应关系。
- 不在没有逐项确认的情况下清理 checkpoint、回收站或未跟踪文件。
