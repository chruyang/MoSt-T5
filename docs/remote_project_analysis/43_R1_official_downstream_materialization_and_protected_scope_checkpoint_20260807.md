# R1 正式下游数据物化与保护范围检查点（2026-08-07）

> **历史快照，已由文档 44 取代。** 本文保留旧 QM9/HIV v1 和当时保护差集的执行证据；不得把其中成员数、路径或“当前状态”用于新的训练。现行结果为 QM9/HIV v2 与 paper-scope-final-v4：3,365,577 个 PCQM members 中排除 5,510、保留 3,360,067；详见文档 44。

状态：**历史执行检查点；最终论文范围的保护集合后来已完成。**

本文接续文档 40–42 的执行状态。旧文中的架构假设、四格实验设计与任务优先级仍然有效；其中关于数据“待物化”、旧 QM9 行数、HIV 7 条无效以及 0.5 vCPU 调度的描述，不再代表当前事实。

## 1. 本轮科研裁决

1. 正式化学身份协议统一使用 **RDKit 2024.03.5**。该版本与 PCQM4Mv2 production release 及 KPGT 处理环境一致，并能完整解析本项目采用的 QM9 与 HIV 总体。
2. RDKit 2025.09.1 下 QM9 有 1,116 条、HIV 有 7 条不能按同一规则解析；这组结果仅用于说明版本漂移，不进入正式训练或评估成员。
3. 数据准入依据是来源与版本、字段和标签语义、完整总体计数、切分成员、化学身份规则及泄漏不变量。文件大小和摘要可以保留作来源观察或接口标识，但不是额外的科研结论。
4. 预训练去污染只对下游 **validation/test 的 canonical non-isomeric connectivity** 做硬排除。立体身份只报告，文本身份不参与当前分子成员差集。
5. 当前 22 个评估集合只覆盖 QM9 与四项 MoleculeNet 支撑任务；它不是最终论文范围的保护集合，不能据此宣布 full-P1 admission。

## 2. 已物化的数据与切分

| 数据 | 来源与用途 | 正式成员/切分 | 关键不变量 | 远端产物 |
|---|---|---:|---|---|
| PCQM4Mv2 P1 identity | 已完成的 136-shard geometry production-v2 release | 源 membership 3,378,606；admitted 3,365,577；reject 13,029 | shard 无缺口；admitted/reject 闭合；每个 admitted payload 已解码；RDKit 2024.03.5 | `/root/autodl-tmp/most-t5-r1/derived/p1-pcqm-production-v2-identity-v1` |
| QM9 clean view | 3D-MolT5 处理后的 `QizhiPei/e3fp-mol-instructions-qm9@bfe55090...` | groups 110,000 / 10,000 / 8,836；rows 298,518 / 27,147 / 23,995 | 42 条模型可见精确重复移除；349,660 retained rows 全覆盖；同分子及不同 E3FP 状态不跨 split | `/root/autodl-tmp/most-t5-r1/derived/qm9-idgroup-110k10k-rest-s42-v1` |
| BACE | KPGT Figshare official archive，三套 scaffold replica | 每套 1,210 / 151 / 152，总体 1,513 | 每套完整覆盖；train/validation/test 不交叉；chiral 与 achiral Murcko 交叉均为 0；标签可评估 | `/root/autodl-tmp/most-t5-r1/derived/kpgt-membership-v2-interface-aligned` |
| BBBP | 同上 | 每套 1,631 / 203 / 205，总体 2,039 | 同上 | 同上 |
| ClinTox | 同上 | 每套 1,182 / 147 / 149，总体 1,478 | 同上；两个标签均可评估 | 同上 |
| HIV | DeepChem `HIV.csv` authoritative population；项目复现 DeepChem 2.8.0 non-chiral Murcko greedy 8:1:1 语义 | 32,901 / 4,113 / 4,113，总体 41,127；invalid 0 | 全成员恰好分配一次；scaffold 不跨 split；每个 split 两类齐全且 AUROC 可计算 | `/root/autodl-tmp/most-t5-r1/derived/hiv-moleculenet-deepchem-murcko-8-1-1-derived-v1-rdkit202403` |

HIV 的类别计数为：train 31,669/1,232，validation 4,032/81，test 3,983/130（negative/positive）。该 HIV membership 是透明、可重放的项目 derived split，不冒称 3D-MolT5、KPGT 或 DeepChem 曾发布完全相同的成员文件。

## 3. 统一 identity-collection 接口

当前可直接供预训练差集使用的标准 eval manifests 共 **22 个**：

- KPGT：3 tasks × 3 scaffold replicas × validation/test = 18；
- QM9：validation/test = 2；
- HIV：validation/test = 2。

由于旧远端代码归档把同一 JSON 规范的 LF/CRLF 原始字节摘要当作 identity-spec ID，本轮没有在差集算法中加入特殊映射补丁，而是以 PCQM release 使用的原始规范文件重新生成轻量 identity manifests。分子身份行、split 与标签均未改变；只统一了接口 ID。对齐后的路径为：

- `/root/autodl-tmp/most-t5-r1/derived/kpgt-membership-v2-interface-aligned`
- `/root/autodl-tmp/most-t5-r1/derived/qm9-hiv-identity-collections-v2-interface-aligned`

对齐前后 KPGT 的 37 个 JSONL 和 QM9/HIV 的 4 个 molecule-row JSONL 均逐文件相同；变化仅发生在带接口 ID 的 manifests/summary。

22 个 manifests 包含 30,083 个 eval member occurrences。按 canonical non-isomeric connectivity 去重后：

| 来源 | eval occurrences | unique connectivity |
|---|---:|---:|
| KPGT 三任务三套 split | 3,021 | 2,254 |
| QM9 | 18,836 | 18,834 |
| HIV | 8,226 | 8,226 |
| 三者并集 | — | **29,273** |

跨来源交集很小但非零：KPGT∩QM9 为 2，KPGT∩HIV 为 38，QM9∩HIV 为 1。因此差集必须对保护并集做一次集合语义排除，不能把各任务命中数简单相加。

## 4. PCQM 身份提取结果

完整提取以 shard 为并行边界，32 workers 在 64 vCPU 实例上完成，最终得到 3,365,577 条 PCQM identity rows。13,029 条 reject 的原因构成为：

- `PCQM_STEREO_2D3D_DIVERGENCE`: 12,978；
- `PCQM_SDF_CSV_CONNECTIVITY_MISMATCH`: 33；
- `HYDROGEN_PROJECTION_RESIDUAL_H`: 18。

这里的 3,365,577 是已经具有 production-v2 几何 payload 的 P1 结构成员，不等于最初讨论的 3,899,644 条多模态总体，也不应与 3,119,717 条阶段一文本/多模态 pretrain 计划直接混用。后续 producer 必须通过 member ID 显式连接相应数据视图。

## 5. 当前保护差集结果

数据流为：

```text
PCQM production-v2 identity（3,365,577）
  + 22 个 QM9/MoleculeNet validation/test identity collections
  -> connectivity union difference
  -> permitted_member_ids + excluded_member_ledger
```

执行路径：`/root/autodl-tmp/most-t5-r1/derived/p1-clean-membership-qm9-moleculenet-current-v1`。

结果为：

| 指标 | 计数 |
|---|---:|
| PCQM pretrain members | 3,365,577 |
| permitted members | **3,362,482** |
| excluded members | **3,095** |
| excluded unique connectivity | 2,055 |
| 同时命中多个 protected collections 的 excluded members | 184 |

按任务来源观察 PCQM 命中：QM9 为 2,725 members / 1,948 unique connectivity，KPGT 为 279 / 48，HIV 为 93 / 60。来源组合中，2,725 条仅命中 QM9，277 条仅命中 KPGT，91 条仅命中 HIV，2 条同时命中 KPGT 与 HIV。这里的分来源数字用于解释组成；正式排除仍以 29,273 个 connectivity 的总并集一次性完成。

该结果只称为历史的 `QM9+MoleculeNet current executable clean membership`。它仅可用于 membership 接口联调，不能单独构成 PF-CANARY 放行依据；当前 GPU 门禁见文档 44 的 CPU-G0/G1。

## 6. 仍未冻结的核心论文任务

在 final paper-scope protected union 前还需要冻结：

1. PubChem molecule captioning 的 validation/test molecule 与 text-pair identities；
2. ChEBI-20 text-to-molecule / molecule-to-text 的 validation/test identities；
3. FineMolTex 风格 Controlled Motif Editing 的 compatibility test 与 connectivity-disjoint development set；
4. 对需要同时报告 published-protocol 与 clean-protocol 的任务，保留两套结果视图，不用去污染后的数字替换论文原协议数字。

Zero-Shot Retrieval 仍为次级表征评估，不因当前保护集合而自动升级为核心预训练门禁。MoleculeNet 四任务是迁移能力与 3D-MolT5 Table 7 对照，不承担 motif-local 3D 核心因果论证。

## 7. 代码与测试证据

- 本轮直接相关的 QM9/KPGT/HIV builders、identity adapter、clean membership、overlap proof 与 PCQM parallel extractor：Linux/remote 合并定向联测 48/48 通过；
- clean membership 新接口及缓存优化：本地定向联测通过；
- 本地 overlap 全量发现 103 tests，其中 99 pass、2 skip；另 2 项仅因当前 Windows sandbox 禁止创建 multiprocessing pipe 而无法执行，同两项已在 Linux remote 通过。

远端全量发现还会触发旧 downstream registry/config 的 LF/CRLF 字节绑定失败；这些不是本轮数据或算法失败，也不作为新的科研门禁。后续只在实际迁移相关旧模块时逐步改为语义版本/字段检查，不为追求测试总数批量重写历史合同。

`derive_clean_pretrain_membership_v1.py` 现在只要求 manifest 路径。程序仍校验 collection schema、角色、引用行闭合和 connectivity-spec 一致性；artifact digest 由程序观察并写入结果，不再要求调用者人工重复提供。

## 8. 下一阶段门禁

CPU 路线继续按以下顺序推进：

1. 冻结 Caption、ChEBI-20、Controlled Motif Editing 的核心 eval members；
2. 形成 final paper-scope protected union 与最终 clean membership；
3. 在最终 clean membership 上完成 motif census、词表覆盖/OOV 与 producer 输入输出检验；
4. 之后才进入 1×4090 的 A0/A1/M0/M1 PF-CANARY；胜出架构再进行全量预训练，避免把所有候选架构都做完整实验。

本文当时曾建议在 22 集合差集上运行短 PF-CANARY；该建议现已撤销。新的 production canary 必须先满足文档 44 的 graph+ports codec、同源 A/M producer 与 inherited-E3FP 输入合同。
