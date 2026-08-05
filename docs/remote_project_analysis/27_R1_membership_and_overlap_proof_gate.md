# R1 P1/P2 Membership 与 Downstream Overlap Proof Gate

状态：版本化 source extractor、proof consumer 与 hermetic tests 已完成；真实全量 identity extraction / overlap proof 暂停且尚未执行，当前不能写成“下游零泄漏”。  
日期：2026-08-05  
边界：只新增 `most_t5_next/r1/overlap/` 与新 contract，不修改 production、auditor、原始 LMDB 或旧训练代码。

## 1. 当前结论

PCQM P1 production release 已经具备生成分子身份集合的必要字段，但 P2 和正式 downstream 仍未形成与它使用**同一化学归一化规格**的不可变 identity manifests。因此现在可以冻结并测试证明方法，不能诚实地产生最终的零交集结论。

本轮已经实现：

- `p1_p2_downstream_overlap_proof_contract_v1.json`：定义成员、connectivity、stereo、conformer、文本和 molecule-text pair 的独立证据层；
- `prove_membership_identity_overlap_v1.py`：标准库 + SQLite 的独立 set proof consumer；
- `extract_identity_collection_v1.py`：配置驱动的 legacy single-file LMDB、JSON array/JSONL、CSV、Parquet hash-only identity extractor，不猜 CID、SMILES、split、task 或 metadata key；
- P2 raw 301,658 与 motif-ready 301,655 两份已锁 source path/bytes/SHA-256 的候选 config；
- 4090 Python 3.8 环境的 15 个 fixture tests 全部通过，包含真实微型 LMDB `__len__` 排除和 Parquet 批读取。新实现只增加独立目录和 contract，没有改动既有 production/auditor 文件。

该 gate 即使通过，也固定输出：

```text
p1_training_admission = false
p2_training_admission = false
```

它只证明 request 中明确列出的 collection、task family、split、identity spec 和比较维度。

## 2. 为什么不能直接比较 CID 与 PCQM ordinal

不同来源的 record ID 只在自身 namespace 中有意义：

| 来源 | 成员 ID | 能证明什么 | 不能证明什么 |
|---|---|---|---|
| PCQM4Mv2 P1 | `ogb_pcqm4mv2_train_row_index:<ordinal>` | PCQM release membership 闭合 | 与 PubChem CID 是否为同一分子 |
| PubChem P2 | PubChem CID | PubChem split membership 闭合 | 与 PCQM ordinal 是否为同一分子 |
| QM9 / ChEBI / MoleculeNet | 各自局部 ID | 各自 split membership | 跨数据集化学等价 |

跨来源比较必须重新投影到共享的化学身份域。v1 gate 将以下维度分开，不允许互相替代：

1. `connectivity_identity`：忽略立体的共享规范身份，是主泄漏键；同一 connectivity 的不同立体异构体仍计为 molecule-level overlap。
2. `stereo_identity`：保留立体信息，用于区分严格同一 stereoisomer 与仅 connectivity 相同。
3. `conformer_identity`：只在两侧使用同一、明确处理原子顺序和刚体变换的 conformer spec 时可比。当前 P1 的坐标字节哈希或 `source_mol_identity_sha256` 不能直接当成跨数据集 conformer identity。
4. `text_exact` 与 `text_normalized`：分别检测精确文本和按冻结规格规范化后的文本复用。
5. `connectivity_text_pair` 与 `stereo_text_pair`：检测同一分子—文本单元是否跨 split/task 重复；pair hash 不包含来源 row ID 或 task 标签，避免这些标签掩盖重复。

“同一分子有多个构象”不会产生多个独立分子：gate 按 connectivity/stereo 聚合 molecule overlap，同时单独报告 exact-conformer overlap 和“分子相同但构象不同”的受影响成员数。

## 3. Collection、split 与 task 证明结构

每个 collection 只代表一个明确角色和 split：

- `p1_structure_train`；
- `p2_permitted_train_membership`（只表达最终允许的 P2 分子集合，不混入 task text）；
- `p2_alignment_train`；
- `p2_geometry_replay_train`；
- `downstream_train`、`downstream_validation`、`downstream_test`。

每个 downstream task family 必须分别声明 train/validation/test collection。正式 request 至少证明：

```text
P1 train × 每个 downstream validation/test
P2 permitted train membership × 每个 downstream validation/test
P2 alignment task collection × 对应 downstream 文本任务 validation/test
P1 train × P2 permitted train membership（按预注册 policy）
每个 downstream task 内 train × valid、train × test、valid × test
```

如果 P2 caption、text2mol 使用不同的路由文本单元，应建立不同 task-family collection；不能把二者合成一个无法解释的“P2 text”集合。C4 denoise 没有 molecule membership，不应伪装成 molecule-text collection；它需要另一个纯文本污染审计。

对于 evaluation 与 downstream split 隔离，connectivity overlap 必须为零。若两侧都有文本，另要求 molecule-text pair overlap 为零，并至少报告 normalized-text overlap。纯文本相同并不自动等于分子泄漏，所以它单独报告，不与 molecule-pair 计数混合。

## 4. 当前可见身份字段和远端证据

### 4.1 PCQM P1

持久化 release：

`/root/autodl-fs/most-t5-r1/runs/cpu-p0-20260805/incoming/pcqm-geometry-production-v2-20260805T0430Z`

- 3,378,606 个 source membership；3,365,577 admitted；13,029 rejected；
- membership JSONL 提供 ordinal、disposition 和 admitted record hash，但不提供跨数据集化学身份；
- admitted safe payload 提供 `canonical_connectivity_sha256`、严格 isomeric hash、identity spec hash 和 RDKit version；
- `coordinates_sha256` 只绑定当前原子顺序下的坐标字节，不是现成的跨数据集 conformer key；
- rejected records 不属于当前 P1 geometry-admitted train 集，因此 overlap manifest 应以 admitted membership 为主，同时继续绑定完整 reject ledger。

这意味着 P1 identity extractor 可以只读扫描已审计 safe LMDB，无需重扫 SDF、重算 motif 或 E3FP。

### 4.2 PubChem P2 与其内部 holdout

当前 canonical 副本已持久化到：

`/root/autodl-fs/most-t5-r1/sources/p2-pubchem-evidence-r0-v1`

共 21 files / 约 8.7 GB；源端与目标端 checksum diff 为 0。关键资产包括：

- `pubchem/pretrain/3d-pubchem.lmdb`：301,658 个 CID，字段含 `cid`、`smiles`、坐标和描述；
- `pretrain/phase2_pubchem_ready.lmdb`：301,655 个 geometry/motif-ready CID，字段含 `smiles/raw_smiles/motif_seq`；
- `pretrain/2d_computed_properties.json`：1,199,066 task rows；
- `pretrain/2d_descriptive_properties.json`：1,508,290 task rows；
- `train/valid/test/3d-pubchem.lmdb`：12,000 / 1,000 / 2,000 个 CID，原始 CID split 两两零交集。

本轮再次只读确认：`phase2_pubchem_final.lmdb` 的 LMDB stat 为 301,656，是 301,655 个 payload 加一个 `__len__` 元键；ready LMDB 没有该元键。排除 `__len__` 后 final 与 ready 的 payload CID set 完全相同，均只缺 3 个 singleton CID。因此 extractor 必须列出并排除 metadata keys，禁止把 `txn.stat()['entries']` 直接当作成员数。

上述事实只证明 CID split，不证明它们与 PCQM、QM9、ChEBI 或 MoleculeNet 在 connectivity/stereo 上无交集。P2 SMILES 仍需用与 P1 相同的冻结 identity normalization 生成身份哈希。

已冻结但尚未执行全量的两个 extractor config：

- raw pretrain：bytes `1310617600`，SHA-256 `6f775c388b0e4397286235d088e33c14910b92deacd122c5af4e9a1b5dd11662`；
- motif-ready：bytes `1909297152`，SHA-256 `465d89f4aafb36043a5964441feffceb3e3e6493fe2ffee9d53190ec7587d5e5`。

两者均显式 permit（但不要求出现）`__len__` metadata key；任何未声明的 `__*` key fail closed。legacy pickle 只在 acknowledgement SHA 与已核验源 SHA 完全相同时允许离线解码。

### 4.3 其他 downstream

Legacy/downstream canonical 副本已持久化到：

`/root/autodl-fs/most-t5-r1/sources/legacy-and-downstream-evidence-r0-v1`

共 130 files / 约 12 GB；8 类逐组 checksum diff 均为 0。该 closure 包含 PubChemQC、QM9、ChEBI、MoleculeNet/ADMET、C4 等现存证据。QM9 与 ChEBI 的原始 Parquet 包括：

- `/root/autodl-tmp/e3fp-mol-instructions-qm9/data/{train,validation,test}*.parquet`；
- `/root/autodl-tmp/e3fp-chebi-molgen/data/{train,validation,test}*.parquet`。

Parquet metadata 显示：QM9 为 347,774 / 1,928 / 1,928 个 instruction rows，字段是 `instruction/output/element/selfies/smiles`，没有独立 molecule ID，因此成员必须由 source row address 与共享 SMILES identity 分层表示，不能把 instruction row 数当唯一分子数。ChEBI 为 26,407 / 3,301 / 3,300 rows，字段是 `cid/smiles/element/selfies/instruction/input`；CID 仍只作 source member ID，跨数据集比较使用共享 connectivity/stereo key。

持久化 closure 中已经找到多套 MoleculeNet CSV 和 `scaffold-{0,1,2}.npy`，但具体采用哪个 seed/split 以及每个数组的 train/valid/test 语义仍需按实际 loader 配置冻结；文件存在不等于 task matrix 已裁决。因此仍不能称为“all downstream exclusion”。

## 5. 跨区持久化结果与排除项

### 5.1 P2 / PubChem 内部 overlap 与 tokenizer census 的核心集合

以下 P2 核心清单已经持久化；canonical closure 不依赖 `.lmdb-lock`、`.DS_Store` 或 notebook checkpoint：

1. `pretrain/3d-pubchem.lmdb`；
2. `train/3d-pubchem.lmdb`；
3. `valid/3d-pubchem.lmdb`；
4. `test/3d-pubchem.lmdb`；
5. `pretrain/2d_computed_properties.json`；
6. `pretrain/2d_descriptive_properties.json`；
7. train/valid/test 下对应的两类 `2d_*.json`；
8. `pretrain/phase2_pubchem_ready.lmdb`。

为保留 301,658→301,655 的处理谱系，也已保存：

- `pretrain/phase2_pubchem_final.lmdb`；
- `/root/autodl-tmp/3D-MolT5/3d_tokenization/3d-pubchem-all-e3fp.lmdb`。

`3d-pubchem-all.lmdb` 与四个原始 split LMDB 在 membership 角色上重复；空间允许时可作为 union closure 的独立复核源保存。

### 5.2 Downstream

QM9、ChEBI、PubChemQC 与现存 MoleculeNet/ADMET 证据已进入上述 12 GB closure。正式评测仍须先冻结使用哪些真实版本和 split；不能由代码中可能自动下载的浮动版本替代。

所有复制应在源端和目标端各自生成 canonical manifest：相对 POSIX 路径、bytes、SHA-256；目标端逐文件复核。不得把锁文件、cache 或临时文件混入 canonical closure。

### 5.3 明确排除的 932 GiB 稀疏占位 LMDB

`/root/autodl-tmp/3D-MoIT/3d-mol-dataset/pubchemqc/pubchemqc_e3fp.lmdb` 未复制。只读实测：logical bytes `1,000,000,000,000`，allocated bytes `8192`，LMDB `entries=0`、`depth=0`、`last_txnid=0`，无 first key。它是空稀疏 placeholder，应记录为 `EMPTY_SPARSE_LMDB_PLACEHOLDER_ZERO_ENTRIES`，不能作为 P1/E3FP source lock。

Tokenizer 所需 base snapshot 也已独立持久化：

`/root/autodl-fs/most-t5-r1/base-models/google-t5-v1_1-base/b5fc...`

## 6. 真实 proof 前仍需裁决的四个硬阻断

1. **P1/P2 policy**：主文档推荐 `P1 ∩ P2 = 0`，但尚未正式预注册。gate 不替用户决定，可选择 `disjoint_required`、`explicitly_declared` 或 `replay_permitted`。若主线选 disjoint 且测得非零，必须生成排除后的新 P1 或 P2 membership manifest，不能只在报告中解释。
2. **P2 permitted membership**：选择 301,655 个 geometry-ready 分子，还是保留 3 个 singleton 仅走 2D/text 路由的 301,658 集合。两者会改变 P2 membership 与 tokenizer discovery scope。
3. **最终 downstream task/split 清单**：QM9、ChEBI、MoleculeNet 各哪些任务，采用官方 split、scaffold split 还是固定 seed split；每项的真实 source revision 尚未统一冻结。exact molecule overlap proof 也不能替代 MoleculeNet 的 scaffold-isolation proof。
4. **文本单元与 normalization**：必须按真实 Dataset/Collator 路由决定 hash 的单位是 encoder input、decoder target，还是 canonical input-target object，并冻结 Unicode/空白/模板处理。不能为了获得零交集临时更换规范。

## 7. 下一执行顺序

1. 当前按用户要求暂停，不启动 P2 全量 identity extraction；
2. 恢复后先冻结 P1/P2 overlap policy、P2 301655/301658 policy 和正式 downstream task matrix；
3. 运行已完成的 legacy PubChem/downstream extractor；PCQM safe-payload extractor仍需补齐；
4. extractor 在不同进程运行两次，要求 canonical JSONL、key-set digest 和 manifest SHA-256 一致；
5. 使用本轮 gate 在 4090 端磁盘 SQLite 模式运行全量 proof；
6. 只有 overlap proof、tokenizer binding 和 GPU candidate gate 均完成，才进入独立 P1 admission decision。

## 8. 文件与验证

- Contract：`most_t5_next/r1/contracts/p1_p2_downstream_overlap_proof_contract_v1.json`
- Extraction contract：`most_t5_next/r1/contracts/identity_collection_extraction_contract_v1.json`
- Gate：`most_t5_next/r1/overlap/prove_membership_identity_overlap_v1.py`
- Extractor：`most_t5_next/r1/overlap/extract_identity_collection_v1.py`
- P2 configs：`most_t5_next/r1/overlap/configs/`；
- Tests：`most_t5_next/r1/overlap/tests/`；
- 4090 Python 3.8 fixture tests：15/15 通过；
- Python 3.8 grammar parse：通过。

远端 checkpoint bundle：

`/root/autodl-fs/most-t5-r1/cpu-bundles/overlap-identity-v1-20260805`

- bundle manifest SHA-256：`41cd4730f39fbccd5c36d2b8b7f61f54498497e496c22bc5564ca8bcef4ca26e`；
- 12-file tree SHA-256：`210eed1169d50a07764cfc4dc79b49454d2568ef35022d09b0216b594862ff59`；
- `full_data_extraction_executed=false`。

## 9. 2026-08-05 geometry replay coverage 修正

后续主干审查发现：contract 与 collection validator 已允许 `p2_geometry_replay_train`，但 proof consumer 的 `PRETRAIN_ROLES` 原先只包含 P1 structure、P2 permitted membership 与 P2 alignment。其结果是：即使 request 已声明 geometry replay collection，`require_each_pretrain_vs_each_downstream_eval=true` 也不会自动要求 replay 与每个 downstream validation/test 比较。

已做最小修正：

- 将 `p2_geometry_replay_train` 加入 `PRETRAIN_ROLES`；
- 新增回归测试：request 含 replay collection 但缺 replay-vs-validation/test 比较时 coverage 必须失败；两项比较补齐后才允许通过；
- 本地主干 overlap 测试更新为 20/20 PASS，另有 2 项因本地缺少可选 pyarrow/python-lmdb 而跳过；
- 独立范围审查结论为 PASS：`replay_permitted` 只可能是 P1/P2 之间的科学政策，不能放宽训练数据相对于 downstream evaluation 的 connectivity 隔离。

该修正仍不能发现“真实训练启用了 replay，但 proof request 完全漏报 replay collection”。正式 request 必须由冻结的训练成员清单生成；启用 replay 时，`required_collection_roles` 必须显式包含 `p2_geometry_replay_train`。
