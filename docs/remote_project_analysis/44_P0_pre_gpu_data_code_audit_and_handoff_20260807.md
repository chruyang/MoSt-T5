# P0 上卡前数据、化学入口与模型 I/O 审计交接（2026-08-07）

> 历史状态说明：本文记录 GPU canary 前的阶段性审计。graph+ports codec、同源 A/M producer、paired-128、union-init 与真实 RTX 4090 四格前后向已于同日完成；最终状态与实测数字见文档 45。

状态：**历史上卡前检查点；已由文档 45 的 P0 收口结果取代。**

本文接续文档 41–43，记录本轮本地与 westb 64-vCPU 实例上的实际代码、数据和测试事实。目标不是增加工程门禁，而是保证后续四格实验只改变“atom/motif 粒度”和“是否融合 3D”两个声明变量。

## 1. 主干裁决

当前 PCQM4Mv2 结构 release 的主干是可保留的：同一个 SDF RDKit Mol 经冻结的显式氢投影后，同时生成坐标、E3FP、motif partition 和 atom mapping；官方 CSV SMILES 只用于身份核验，不参与重建几何。

本轮已解决一个会真实改变数据的身份问题：QM9/HIV 原实现没有执行其声称共享的 PCQM 显式氢投影。修复不是元数据调整；QM9 有 3,719 条 retained records 的 strict/connectivity identity 发生变化，并且重新按 connectivity 分组后有 91,119 条记录改变 split。

但目前仍不应直接启动 10% 或全量 GPU 训练，原因是三项模型输入尚未闭合：

1. 删除 anchor 得到的 `pure motif` 不是可靠化学子图；
2. A0/A1 缺少真实 SDF-atom → SELFIES token/carrier producer；
3. M0/M1 的 slot-aware graph+ports codec 尚在最终复核，正式 record producer 尚未完成。

E3FP duplicate shell 已不再是语义阻塞：128 条跨范围样本已从 SDF 重放，并证明 raw payload 128/128 与 production-v2 一致；显式 inherited 结果作为独立 overlay 发布，未修改原 release。

此外，geometry batch → Trainer、union vocab resize/init 和 wrapper checkpoint/resume 仍需三个薄层。它们应在同一批 128 条 paired A/M records 上完成后，才租用 1×4090 做真实 forward/backward。

## 2. 当前应采用的数据流

```text
PCQM SDF record
  -> ForwardSDMolSupplier(sanitize=True, removeHs=False)
  -> source atom tag
  -> locked RemoveHs projection + Sanitize + AssignStereo
  -> one geometry_mol
       -> coordinates[A,3]
       -> E3FP[A,4]
       -> motif atom groups[M]
       -> model_to_source_atom_index[A]
       -> atom_to_motif_map[A]
  -> identity/reject ledger
  -> topology augmentation
       -> motif-local ports + cross-motif connections
  -> A producer or M producer
  -> one frozen union tokenizer
  -> A0/A1/M0/M1 collator
  -> standard T5 CE; A1/M1 additionally receive the same geometry side tensors
```

必须保持：

- `coordinates[i]`、`e3fp[i]` 和 `model_to_source_atom_index[i]` 指向同一个投影后原子；
- 每个原子恰好属于一个 motif；
- A1 与 M1 使用同一 E3FP 行和同一几何融合参数，只允许 carrier mapping 不同；
- A0/A1 内 CE batch 相同，M0/M1 内 CE batch相同；
- 3D E3FP 是有损状态描述，无损保证只属于 2D motif graph+ports codec。

## 3. `Chem.MolFromSmiles()` 的真实角色

`Chem.MolFromSmiles()` 只解析并 sanitise 2D 分子图；它不产生可信 3D 坐标，也不证明其 atom order 与 SDF、SELFIES 或保存的 E3FP 行一致。

在本项目中应区分三类入口：

| 数据/阶段 | `MolFromSmiles()` 的允许角色 | 禁止行为 |
|---|---|---|
| PCQM P1 结构预训练 | 解析官方 companion SMILES，生成 strict/connectivity identity，与 SDF identity 比对 | 不得从该 Mol 重新 embed、生成 E3FP、给 SDF 坐标贴索引或替换 SDF atom order |
| QM9/HIV/KPGT/Editing 身份与 split | 解析 2D SMILES，随后调用统一显式氢投影 helper | 不得各 builder 自己形成不同 canonicalization universe |
| 仅有 SMILES、但模型需要 3D 的下游任务 | 必须另行冻结 conformer 来源、生成算法、随机种子、失败 ledger 和 atom mapping | 不得把成功 `MolFromSmiles()` 等同于已经有 3D state |

因此，旧代码中“SMILES 新建 Mol/构象 → E3FP”与“SDF 坐标只按 atom count 贴给另一个 Mol”的路径不得进入新主线。

## 4. 已完成的身份修复

### 4.1 统一 helper

新增 `most_t5_next/r1/overlap/shared_identity_normalization_v1.py`，固定执行：

```text
Chem.Mol(input_mol)
-> RemoveHsParameters，仅 override removeDefiningBondStereo=True
-> Chem.RemoveHs(..., sanitize=True)
-> Chem.SanitizeMol
-> Chem.AssignStereochemistry(cleanIt=True, force=True)
-> canonical strict-isomeric / non-isomeric connectivity SMILES
```

QM9、HIV、KPGT、Controlled Editing 四个 builder 均委托该 helper。helper 与冻结的 `pcqm_identity_smoke.canonical_forms()` 在显式立体氢、氘、R/S、E/Z、盐、带电分子和不同 SMILES atom order fixture 上一致。

### 4.2 QM9 connectivity-level v2

冻结源仍为 `QizhiPei/e3fp-mol-instructions-qm9@bfe55090be9ebf1c9cbbe6687a5796711ac0edd8` 的 train+validation；发布的 test 因与 validation 重复而不参与派生。

远端产物：

```text
/root/autodl-tmp/most-t5-r1/derived/
qm9-connectivity-group-110k10k-rest-s42-v2
```

全量结果：

| 指标 | 结果 |
|---|---:|
| 输入 rows | 349,702 |
| retained rows | 349,660 |
| model-visible exact duplicates removed | 42 |
| connectivity groups | 128,783 |
| train groups / rows | 110,000 / 298,728 |
| validation groups / rows | 10,000 / 27,139 |
| test groups / rows | 8,783 / 23,793 |
| train/validation/test connectivity intersections | 0 / 0 / 0 |
| distinct stereo states：train/validation/test | 110,044 / 10,006 / 8,786 |

与旧 v1 比较：

- 3,719 条 identity 改变，3,719/3,719 的旧 strict surface 均显式含 `[H]`；
- 91,119 条 assigned split 改变；
- 旧 split 中已实测的 11 个跨 split connectivity 已消失；
- adapter 按 connectivity group 验证 split 闭合，但在 eval collection 中保留不同 stereo state；相同 stereo state 的多条 instruction 不重复扩大 molecule membership。

### 4.3 HIV v2

远端最终产物：

```text
/root/autodl-tmp/most-t5-r1/derived/hiv-murcko-derived-final-v2
```

身份与 Murcko scaffold 均从同一个 post-projection Mol 计算。41,127 条全部可解析；成员数仍为 32,901 / 4,113 / 4,113。与旧 RDKit-2024.03 v1 比较仅 1 条训练分子的 identity/scaffold serialization 改变，0 条成员改变 split，因此 validation/test protection 内容不变。

HIV source/split/member schema 与 protocol 均升级为 v2。adapter 现在逐条检查 dataset ID、protocol ID、三 split 行数以及全局 member ID 唯一性；旧 v1 member rows 不能再与 v2 split 拼接后重新贴 spec。

### 4.4 identity adapter 闭合

最终 adapter 远端产物：

```text
/root/autodl-tmp/most-t5-r1/derived/
qm9-hiv-identity-collections-final-v2
```

它独立验证：

- QM9 summary dataset/protocol/identity spec；
- 所有 train/validation/test rows 与 group counts；
- `group_id == qm9-canonical-connectivity-smiles-sha256:<connectivity_digest>`；
- connectivity group 不跨 split；
- HIV split/member dataset、schema、protocol、row count 和 member ID 闭合。

五类反例测试均会拒绝：QM9 同 connectivity 使用不同 group ID、QM9 group 跨 split、QM9/HIV identity-spec 不一致、HIV 旧/缺失 row protocol、HIV member ID 跨 split 重复。

## 5. motif 词表的全量事实

旧 tokenizer 将 exact fragment 中的 `<N*>` 用正则删除，得到所谓 pure motif。paper-scope-v2 全量 clean-P1 census 的 RDKit parseability 结果为：

| 投影 | RDKit-invalid unique | occurrence-weighted invalid |
|---|---:|---:|
| 441,452 exact lexemes 删除 anchor | 278,695（63.1314%） | 6,001,259 / 24,153,133（24.8467%） |
| 聚合后的 214,378 pure core | 108,506（50.6143%） | 同上 |
| exact lexeme 将 anchor 保留为合法 `[*]` port | 0 / 441,452 | 0 / 24,153,133 |
| 229,359 slot templates 将 `<*>` 表示为 `[*]` | 0 / 229,359 | 0 / 229,359 |

这里的 0 invalid 只证明这些字符串可被 RDKit 解析；它不证明 port canonicalization、attachment 对应关系或 graph+ports round-trip 已经实现。

高频失败 surface 包括 `C()` 1,232,689 次、`C()=O` 1,004,310 次、`C()()` 379,796 次、`N()` 365,031 次、`C()=N` 301,725 次。失败均来自 branch anchor 被删除后留下空分支。

裁决：

- 旧 pure-token 统计只能作为历史字符串投影事实；
- 正式 identity 必须是 chemical core graph + canonical motif-local ports；
- 分子级 edge ID 不进入 motif identity，但连接 span 中同一 edge ID 必须恰好出现两次且 bond type 一致；
- macro 和 rare fallback 必须 graph+ports round-trip 到同一 identity，不能使用普通 `<unk>`；
- 修复后重跑词表规模、覆盖率、fallback 长度和完整序列长度；旧 214k pure vocab 数字不能用于选 K。

上述 parseability 比率绑定的是当时排除 5,386 个成员的 paper-scope-v2，足以否定“删除 anchor 后仍是合法化学 core”的设计。最终 paper-scope-final-v4 已排除 5,510 个成员，并已完成独立 clean census：permitted 3,360,067、clean motif occurrences 24,152,754、clean exact unique motifs 441,442、clean slot unique motifs 229,337；相对全局去除 27,474 个 motif occurrences。后续正式 K、覆盖率和长度统计必须使用这份 final-v4 census，而不再使用旧 214k pure vocab。

## 6. PCQM 真实 sidecar 的输入输出检查

在 128 个不同 shard 上各取一条确定性记录，共 128 records / 1,792 model atoms / 939 motifs：

| 指标 | min | p50 | mean | p95 | max |
|---|---:|---:|---:|---:|---:|
| model atom count A | 5 | 14 | 14.000 | 17 | 18 |
| source atom count（含显式 H） | 9 | 29.5 | 29.070 | 40.65 | 48 |
| projected explicit H | 0 | 15 | 15.070 | 24.65 | 31 |
| motif count M | 2 | 7 | 7.336 | 11.65 | 14 |
| motif atom count | 1 | 1 | 1.908 | 6 | 13 |

以下违例均为 0：

- `coordinates.shape != [A,3]` 或非有限值；
- `e3fp.shape != [A,4]`；
- source atom map 非唯一、非递增或越界；
- motif groups 空、交叠或未覆盖全部 model atoms；
- identity 非 strict match；
- motif geometry invalid。

E3FP level 0/1/2 无缺失，level 3 有 350/1,792（19.53125%）为 `-1`。因此几何融合必须把 `-1` 当缺失 shell，而不能查成普通 embedding。

这 128 条只能证明 geometry pretokenizer sidecar 的 atom/motif/shape 合同；它不能证明 A/M token producer，因为 sidecar 尚无 SELFIES attribution、slot/connection span、token IDs 或 carrier map。

## 7. 模型 I/O 与测试事实

westb 环境为 Python 3.8.20、PyTorch 2.1.0+cpu、RDKit 2024.03.5、LMDB 1.7.5、NumPy 1.24.4；cgroup 实际为 64 vCPU / 120 GiB。环境没有 `transformers`、`selfies` 或 T5 snapshot，本轮没有临时安装它们。

实际测试：

| 测试范围 | 结果 |
|---|---:|
| `most_t5_next/p1/tests` | 65/65 pass |
| linearizer + topology + vNext + tokenizer 定向 | 35 pass / 2 skip（仅缺 transformers） |
| 完整 tokenizer | 19 pass / 2 skip |
| 完整 adapter | 46 pass；另 2 项仅因 Windows 上传形成 CRLF，内存 LF 规范化后与冻结合同逐字节一致 |
| 最终 identity/QM9/HIV/KPGT/Editing/adapter/clean-view/clean-census 本地定向 | 62/62 pass |

最终 62 项在本机 Python 3.12.1 / RDKit 2025.09.1 下用于证明代码分支、异常和合同测试闭合；它不替代上表数据在远端冻结 RDKit 2024.03.5 下的正式物化与全量统计。

目前可以保留的模型侧实现：

- whole atom/motif identity CE masking；
- A0/A1 与 M0/M1 各自 CE parity；
- T5 labels 与 padding；
- A1/M1 共用的 level-specific E3FP table 和 atom-to-carrier scatter mean；
- A1/M1 geometry row parity；
- 四格 wrapper 的 CE-only/no-extra-loss 边界。

尚缺：

1. geometry/topology → slot-aware tokenizer → `ProductionMotifRecord`；
2. 同一 SDF Mol → canonical atom order/SELFIES attribution → `ProductionAtomSelfiesRecord`；
3. `P1ConditionBatch.geometry` → Trainer tensor kwargs；
4. union vocab resize 和新 token initialization；
5. wrapper + tokenizer + condition + optimizer/scheduler checkpoint/resume；
6. paired real-record corruption budget audit。

## 8. E3FP duplicate-shell 裁决

vendored E3FP 会把重复 substructure shell 保留在 `all_shells`，标记 `is_duplicate=True` 并指向 `shell.duplicate`，但不把其 raw identifier 加入最终 fingerprint。当前 producer 直接使用 raw identifier；3D-MolT5 参考实现则按 folded-bit membership 和相同 substructure 搜索已接受 shell。

需要比较：

在覆盖全部 136 shards 的 stride-10 系统抽样中，处理了 336,608 个 admitted molecules、4,755,781 个 atoms：

| 指标 | 结果 |
|---|---:|
| 含 duplicate shell 的 molecule | 335,548 / 336,608（99.6851%） |
| raw→inherit 后至少一个 token 改变的 molecule | 334,781 / 336,608（99.4572%） |
| 受 duplicate 影响的 atom | 3,559,069 / 4,755,781（74.8367%） |
| raw→inherit 后 token 改变的 atom | 3,534,785 / 4,755,781（74.3261%） |
| duplicate slots / populated slots | 4,343,532 / 17,926,336（24.2299%） |
| inherit 与 3D-MolT5 heuristic 不同的 duplicate slots | 137,828 / 4,343,532（3.1732%） |

另对前 1,000 个来源记录中的 998 个 admitted payload 做了逐格复算，当前 stored raw matrix 为 998/998 一致，说明差异来自语义选择而不是审计脚本读错数据。25k 精细样本中显式 duplicate pointer 缺失为 0、inherit bit 不在 final fingerprint 为 0；inherit 与 3D-MolT5 heuristic 的差异主要来自 4096-bit folding collision 与多个 duplicate candidate 的歧义。

裁决：正式主方案使用 `shell.duplicate.identifier` 的显式 inheritance。当前 raw 只可称为 E3FP-derived intermediate shell hash，并保留为一次小样本高影响消融；把 duplicate 置 `-1` 会影响 99.6851% molecule/74.8367% atom，只作诊断，不进入四格主比较。由于现有 matrix 没有保存 duplicate pointer，必须从 SDF 重算 payload，不能在模型端无损补映射；也不应宣称该实现与 3D-MolT5 heuristic byte-equivalent。

2026-08-07 已在 nmb1 的 RDKit 2024.03.5 / E3FP 1.2.5 环境完成 128 条确定性等距样本的真实重算与独立 LMDB overlay：128/128 raw parity 通过；共 6,818 个 populated shell slots，其中 1,711 个 duplicate slots，显式 inheritance 使 1,688 个 folded tokens 改变；128 条全部编码后又从 LMDB 完整解码复核。该 artifact 明确为 `sample_scope_only=true`、`training_admission=false`，因此只放行 paired canary，不替代 PF-10 或全量 inherited payload。

## 9. 数据特殊情况及报告策略

### 9.1 MoleculeNet/KPGT

- HIV：3,087 多组分、1,329 含常见有机集合之外元素/金属、1,062 净电荷非零，最大 222 atoms / 6 fragments；
- BBBP：105 多组分、37 显式 H；
- ClinTox：14 多组分、2 显式 H；
- KPGT 官方行级数据有同分子冲突标签，但重复都在同一 scaffold split。

为与 KPGT/3D-MolT5 对比，主结果保留官方 row-weighted AUROC，不擅自去重或改标签；论文披露 duplicate-label noise，可增加 molecule-group sensitivity。

### 9.2 QM9 task scope

349,660 retained rows混合 gap 116,441、HOMO 116,610、LUMO 116,609。下游 loader 必须冻结“只做 gap”或“明确做三任务”，不能把三类 instruction 混成一个 gap 指标。相同 exact input 的 3 个目标冲突均为 0.0001 舍入级差异；聚合/保留规则仍需写入任务协议。

### 9.3 Caption / ChEBI

connectivity-clean view 保证 test 不变、三 split connectivity 不交叉，但仍有相同文本跨 split：ChEBI train-test/train-val/val-test 为 19/12/4，Caption 为 20/11/3。论文主表保留 reported split 以便比较，同时报告 connectivity-clean sensitivity；ChEBI 可再做 normalized-text+connectivity 双重 disjoint sensitivity。

### 9.4 Controlled Editing

dev400 中 106 个分子净电荷非零，sealed test200 全为中性，存在 charge shift。后续至少按 charge 分层报告成功率；若重新设计 dev sampling，应新建协议，不覆盖当前 random seed-42 membership。

## 10. 与参考代码的合理边界

- 3D-MolT5 支撑“SDF/SELFIES attribution + per-atom E3FP + T5”路线；其 canonical atom reordering 可作为 A producer 参考，但本项目应显式映射回 SDF source atom，而不是假定 token order。
- CAMT5 支撑 ring/non-single-bond motif 与分子语言建模；官方实现会在切分前保存 R/S 与 BondStereo，并在重连后恢复，而不是永久丢弃 stereo。它仍不能证明该 motif partition 对 3D 聚合最优。
- FineMolTex 支撑 motif 粒度在 text-guided editing 中有价值；其 BRICS/merge 路线不能替代对本项目 motif rule 的结构分层实验。
- 当前项目的 level-specific E3FP tables、level sum 和 carrier scatter mean 是自己的候选实现，不能写成复现 3D-MolT5 的 shared-table/level-mean/0.5 融合。

参考：

- [3D-MolT5 official repository](https://github.com/QizhiPei/3D-MolT5)
- [CAMT5 official repository](https://github.com/Songhyeontae/CAMT5)
- [FineMolTex](https://arxiv.org/abs/2409.14106)
- [E3FP paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6075869/)
- [RDKit RemoveHsParameters](https://www.rdkit.org/docs/cppapi/structRDKit_1_1MolOps_1_1RemoveHsParameters.html)

## 11. 下一轮 GPU 前的最短执行序列

### CPU-G0：先冻结输入语义

1. **已完成**：在同一批 128 条上复算 raw 并与 stored raw matrix 做逐格 parity，从同一次 fingerprinter shells 派生 inherited payload，并完成 artifact 写入/解码 replay；
2. **最终复核中**：实现 graph+motif-local-ports identity、logical↔canonical motif permutation、connection grammar 与 forced-fallback round-trip；
3. 重新做 slot-aware vocab coverage 和 full sequence length；
4. 冻结 A/M 共同 stereo 语义；
5. 冻结 QM9 gap 或三任务口径。

GPU-G0 只要求这 128 条 paired records 的 inherited payload。进入 PF-10/10% 或全量训练前，再分别物化相应范围的 inherited payload；不为 one-batch canary 提前做不必要的全量 E3FP 重算。

### CPU-G1：生产同一批 paired records

1. 从同一批 128 条 SDF records 生成 A 与 M；
2. 证明 record order、source atom map、coordinates 和 E3FP rows 完全相同；
3. 证明 A/M carrier、span、connection、fallback round-trip；
4. 报告 token length、截断、fallback、atom/motif mask coverage 和 target length；
5. 超长结构禁止在 identity/connection span 中途静默截断。

### GPU-G0：1×4090 canary

只在 CPU-G0/G1 通过后租用 1×4090：四格各做真实 forward/backward、一次 optimizer step、save/reload/resume，并报告 logits/loss、显存、吞吐、geometry/token norm ratio。该阶段不引入 MSE/teacher。

### GPU-G1：架构选择

canary 通过后再进入 PF-1 或 stage-1/stage-2 各 10% 的 A0/A1/M0/M1 比较。只有四格结果显示 motif 与 3D 有稳定增益后，才把胜出结构扩大至完整预训练；8×4090 应用于此后的并行训练，而不是用于修 producer。

## 12. 本轮证据与迁移边界

已经完成迁移：

- QM9 connectivity v2；
- HIV final v2；
- QM9/HIV final identity collections；
- final-v4 paper-scope clean membership；
- final-v4 clean slot motif census；
- QM9/HIV identity projection diff ledger；
- motif parseability 与 E3FP duplicate-shell census。

final-v4 与已审阅 v3 的 permitted/excluded JSONL 成员内容逐字节一致；最终目录已打入本地轻量迁移包 `dataset/p0-pre-gpu-final-delta-20260807.tar.gz`（171,361,556 bytes，238 entries），并解包到 nmb1 的 `/root/autodl-tmp/most-t5-r1-final-derived`。该包不包含 41GB PCQM payload。

最终 membership 目录：

```text
/root/autodl-tmp/most-t5-r1/derived/
p1-clean-membership-paper-scope-final-v4
```

其闭合计数为：pretrain 3,365,577、excluded 5,510、permitted 3,360,067、excluded unique connectivity 2,789、同时命中多个 protected collections 的 excluded members 540。现有本地 `dataset/p1-cpu-checkpoint-20260807.tar.gz` 只包含 paper-scope-v2（excluded 5,386），仍不得重命名为 final；应使用上述 final-delta 包。

PCQM 41GB payload 已在现有数据位置，不重复打入轻量迁移包。旧 v1/v2/v3 目录作为过程快照保留，不覆盖、不删除，也不得作为 final 路径被训练脚本默认发现。

本地不提交的详细审计报告：

- `tmp/chemistry_pipeline_audit_20260807.md`；
- `tmp/model_io_remote_cpu_validation_20260807.md`；
- `tmp/dataset_edgecase_audit_summary_20260807.md`；
- `tmp/clean_motif_rdkit_parseability_remote.json`；
- 其余结构化特殊样例 JSON 位于 `tmp/`。

上述报告与结构化 JSON 已另存为本地轻量证据包
`dataset/p0-pre-gpu-audit-evidence-20260807.tar.gz`（约 64 KB，14 个条目）；
该 evidence 包不包含原始数据、checkpoint 或源码；final-v4 数据使用独立的 final-delta 包。

最终放行状态（已更新）：**上述阻断项均已在文档 45 闭合；本文保留为上卡前审计记录。**
