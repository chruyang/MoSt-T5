# R1 下游协议与四格训练接口科研检查点（2026-08-07）

状态：**允许进入高 CPU 数据物化；暂不允许启动 GPU PF-CANARY**。

本检查点执行文档 41 的科学路线，并吸收独立审稿后的收紧项。它是“科研协议与训练接口 checkpoint”，不是“生产训练已就绪”声明。

## 1. 本轮裁决

下游任务统一采用以下来源优先级：

```text
3D-MolT5 实际发布且可追溯的任务数据
  → 任务/benchmark 官方或领域权威发布物
  → 明确命名、记录版本与边界的 fallback
```

数据来源与切分权威性分开记录。沿用 3D-MolT5 的分子、E3FP 或指令 artifact，不等于必须沿用已证实有缺陷或无法恢复的 row-level split。正式比较要求 MoSt-T5 与 3D-MolT5 checkpoint 在同一冻结成员、split、微调预算和 evaluator 上重评；3D-MolT5 论文数字单列为 published background。

MoleculeNet 只保留四个支撑任务：BACE、BBBP、HIV、ClinTox。它们用于测试迁移能力和方便与 3D-MolT5 Table 7 对照，不承担 motif-local 3D 的核心因果证明。

## 2. 已冻结的数据与 split 方案

| 任务 | 主来源 | 正式 split | 当前状态 |
|---|---|---|---|
| QM9 HOMO/LUMO/gap | 3D-MolT5 HF revision `bfe5509...` 的 E3FP/SELFIES/instruction artifact | `qm9-3dmolt5-idgroup-110k10k-rest-s42-v1`；strict canonical-isomeric group 切分，non-isomeric connectivity 只用于去污染 | 源文件名、bytes、SHA-256 与预期 census 已写入生产门禁；待高 CPU 物化 |
| BACE/BBBP/ClinTox | [KPGT 官方仓库](https://github.com/lihan97/KPGT)指向的 Figshare 发布包 | 官方 `scaffold-0/1/2` membership replicas，8:1:1 | 当前三套远端文件仅为 internally-valid candidate；nmb1 对 Figshare 返回 HTTP 403，待可访问地区取得官方归档并核验 |
| HIV | DeepChem 2.8.0 官方 `HIV.csv` | `HIV-MoleculeNet/DeepChem-Murcko-8:1:1-derived-v1` | 源已冻结：2,193,844 bytes、41,127 rows、SHA-256 `9ffa7fe57dc86c342627ee1d5255e937e2ab812393c73c4d16c697022f6e1d22`；待物化 membership |
| PubChem caption | 3D-MolT5 发布/现有 LMDB 证据 | reported-protocol 与 connectivity-clean 两个视图 | 待 reconciliation 与 clean membership |
| ChEBI-20 | 3D-MolT5 HF revision `9949fae...` | reported-protocol 与 connectivity-clean 两个视图 | 待 canonical-connectivity proof 与 codec adapter |
| Controlled Motif Editing | MoleculeSTM 发布的 200 molecules + FineMolTex 12 prompts | 200 molecules 为 sealed compatibility test；从其余 ZINC250K 冻结 disjoint dev | test 源已冻结；dev membership 与 runner 待完成 |

KPGT 的论文协议是 scaffold 8:1:1，并报告三次独立运行；`scaffold-0/1/2` 是三份发布 membership replicas，不能在没有代码证据时反推为某三个数值 split seeds。最终同时区分：

- split-replica variation；
- 同一 membership 内 training-seed variation。

KPGT 发布的 11-task 集不含 HIV。3D-MolT5 官方仓库完整 dataset 表及作者公开 HF dataset inventory 也没有可追溯 HIV members/split，所以 HIV 不再保持“来源待选择”，而是固定回退至 DeepChem 权威成员；新派生 membership 不冒称 3D-MolT5、KPGT 或 DeepChem released exact split。

## 3. 已完成的可执行实现

### 3.1 下游成员与去污染工具

- `build_qm9_identity_split.py`
  - 禁止读取与 validation 字节相同的 released test；
  - 只删除语义、SELFIES 与完整 E3FP 均相同的模型可见精确重复；
  - production 模式同时强制源文件 hashes 与完整 census：349,702 input rows、42 removed、349,660 retained rows、128,836 groups，以及固定的 row/group split counts；
  - census 不符时在创建输出目录前失败。
- `build_kpgt_scaffold_manifests.py`
  - 先实测 official archive SHA-256；
  - 直接流式枚举 ZIP/TAR，拒绝 traversal、链接、非普通或重复/歧义成员；
  - 12 个 BACE/BBBP/ClinTox CSV/NPY 文件必须与 archive 内唯一成员逐字节一致，之后才允许读取；
  - 输出 9 份 replica member manifests、27 份 split collection manifests 和三组 valid/test connectivity 保护并集。
- `build_hiv_murcko_split.py`
  - 强制 DeepChem revision、source bytes、SHA-256、MD5/ETag、header、SMILES 与 label；
  - 复刻 DeepChem 2.8.0 非手性 Bemis-Murcko grouping/order/greedy 8:1:1 语义，不使用 RNG；
  - 检查完整覆盖、零 scaffold 泄漏及每个 split 的 AUROC 可计算性；
  - 输出 exact project membership 与 valid/test connectivity rows。

统一数据流为：

```text
source bytes + revision + hash
  → task-specific member builder
  → validation/test identity-collection manifests
  → current paper-scope protected connectivity union
  → P1/P2 permitted membership
```

只从预训练 membership 中排除当前论文任务的 validation/test canonical connectivity。下游 train overlap 保留并披露；不做全局 scaffold 排除，也不让 deferred retrieval/PubChemQC 阻塞当前 P1。

### 3.2 A0/A1/M0/M1 训练接口

- Atom/SELFIES 与 logical-motif 两个 production record/CE bridge 已形成独立数据边界；
- A1/M1 显式校验完全相同的 post-hydrogen-projection E3FP atom rows、atom mask 与 source-atom mapping；
- 四格使用同一 T5、union vocabulary 形状、LM head 和 wrapper schema；A0/M0 走标准 `input_ids + CE`，A1/M1 走同一 geometry sidecar；
- 几何链已修正为四个有序 E3FP shell 的 level-specific embeddings，相加生成 atom state，再按 carrier 做 atom mean 并加到 T5 input embedding；A1/M1 唯一几何差异是 carrier mapping；
- 未加入 MSE、EMA teacher、gate、concat、C1/C2/C3。

参数报告必须同时给 nominal total 与 gradient-active parameters。A0/M0 虽持有相同 geometry state schema，但不激活该路径，不能仅凭名义参数量相等宣称完全排除 active-path 容量影响。

## 4. 四格能够和不能够回答的问题

`A1-A0` 与 `M1-M0` 是两种 representation package 内部的 E3FP 条件效应。差分中的差分衡量 E3FP 在 atom package 与 motif package 下的相对效应。

A/M 同时改变 tokenization、connection 表示和 corruption unit；同一个 `mask_probability=0.15` 也不代表 atom 与 motif 条件具有相同 mask realization。因此当前四格不能把 `M1-A1` 写成“仅由 atom→motif mean pooling 造成”的纯因果效应。执行时需报告实际 masked atoms/motifs、masked identity tokens、supervised target tokens、molecule exposures 与 GPU hours。

PF-10 的 2 paired seeds 只用于方向筛选、候选排序与交互方向估计。正式论文主张以 PF-FULL 实际训练的 winner 与最近机制对照为边界；未全量覆盖的交互后续再补，不用一股脑训练所有架构。

## 5. 当前测试证据

- QM9 builder：9/9 PASS；
- KPGT builder：11/11 PASS；与现有 derive/proof 联测 23/23 PASS；
- HIV builder：4/4 PASS；
- downstream registry/source contracts：38/38 PASS；
- P1 本地无 Torch 路径：47 tests PASS，18 个 Torch tests 按环境跳过；
- level-aware geometry 的远端 PyTorch 联测：29/29 PASS。

这些结果证明合同、边界和核心 tensor path 可执行，不证明生产数据 producer、训练收敛或科学假设已成立。

## 6. 高 CPU 阶段的准入与验收

下一步需要至少 `16 vCPU / 120 GiB`，GPU 可暂时不启用。建议继续使用 nmb1 同地区实例，因为 PCQM/QM9/legacy downstream 资产已经在该地区的 `autodl-fs`；只需用可访问 Figshare 的其他地区实例短暂下载 KPGT 官方约 27.74 MB 归档并传回 hash-bound 文件。

高 CPU 顺序：

1. 物化 QM9 split，验收所有固定 counts 与 RDKit version；
2. 获取 KPGT official archive，运行 archive-to-root binding，物化三任务全部 replicas；
3. 下载/绑定 DeepChem HIV.csv，物化 deterministic derived membership；
4. 冻结 Caption、ChEBI、Editing dev 的 valid/test identity manifests；
5. 构建当前 paper-scope protected union，重新派生 P1/P2 membership；
6. 完成 motif census、K 值、一次性 union tokenizer snapshot；
7. 完成 topology/geometry → atom/SELFIES record 与 logical-motif record 的批量 producer，证明 SELFIES carrier order 与重原子 E3FP row order；
8. 完成统一 trainer/launcher/checkpoint 日志并跑 CPU batch contract。

高 CPU 放行标准：所有 builder 结果可重复、source/manifest hashes 固定、protected union 完整、两个 production record family 在同一成员顺序可重放。

## 7. GPU 门禁与资源结论

当前远端无卡实例 cgroup 是 `0.5 vCPU / 2 GiB`，只适合轻量下载/header/hash 检查；它不适合上面的物化工作。当前不需要切 8×4090，也不应租 GPU 等待 CPU 处理。

只有满足以下条件才启用 `1×4090`：

1. union tokenizer snapshot 冻结；
2. A/M production batch producer 完成；
3. 同一真实 batch 的 A1/M1 atom-parity gate 通过；
4. 四格 level-aware wrapper 完成前向、反向、保存/恢复合同；
5. PF-CANARY 配置固定并能记录 actual masks、tokens、active parameters、吞吐和显存。

随后才按 `1×4090 PF-CANARY → 4–8×4090 PF-1 → 8×4090 PF-10` 升级。当前最合理的动作是切回高 CPU 配置，而不是直接开启 8 卡。
