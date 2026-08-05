# R1 Tokenizer Binding 与 P1 Candidate Gate 最小规格

**日期：** 2026-08-05  
**文档性质：** 实施规格与准入边界审查，不是训练放行报告  
**当前结论：** production v2 可以在不重扫 SDF 的前提下完成 tokenizer binding；但是 pretokenizer release、tokenizer release 与 128 条 GPU candidate gate 均不能单独等同于 P1 training admission。

## 1. 目的与边界

本文规定从 R1 PCQM4Mv2 production v2 的全局 motif census 出发，构建确定性 motif 词表、冻结 T5 tokenizer、完成逐记录 token 绑定，并运行 128 条 Dataset → Collator → CE+MSE → forward/backward → save/reload 候选门禁所需的最小实现。

本文不执行以下裁决：

- 不把正在构建或已完成的 geometry-only pretokenizer release 改称 P1 数据准入；
- 不替代完整 membership、reject ledger、下游身份排除和最终 data-release manifest 门禁；
- 不依据 128 条样本判断 CE+MSE 的科学有效性或下游收益；
- 不修改历史 `MotifTokenizer`、旧 Dataset、旧模型或旧 launcher；
- 不在本文中主观决定词表 cutoff、OOV、MSE 权重等研究超参数。

相关现有契约与审计依据：

- `most_t5_next/r1/contracts/p1_pcqm_geometry_production_release_contract.json`
- `most_t5_next/r1/contracts/p1_pcqm_geometry_record_schema.json`
- `most_t5_next/r1/contracts/data_release_manifest_contract.json`
- `p0_validation/r0_stable_tokenizer_contract.py`
- `p0_validation/r0_stable_tokenizer_determinism_gate.py`
- `p0_validation/R0_STABLE_TOKENIZER_CONTRACT.md`
- `docs/remote_project_analysis/21_R0_deterministic_tokenizer_contract.md`
- `docs/remote_project_analysis/22_R0_dataflow_and_release_decision.md`

## 2. 对 production v1 旧结论的明确纠正

早期 production v1 仅在 worker 内临时取得 `fragment_sequence`，随后只用于 motif census，而没有为每条记录保存可恢复 motif 身份的稳定引用。因此，当时“若要从已有 geometry record 生成 tokenizer-bound record，必须重新读取源分子并再次运行 linearizer”的判断对 v1 成立。

production v2 已改变这一条件：

- 每条 admitted record 按 motif 顺序保存 `topology.motif_lexeme_sha256`；
- 每个 digest 是 exact motif fragment UTF-8 字节的 SHA-256，不进行 normalization；
- 全局 census 保存 `motif_lexeme_sha256 -> exact motif_fragment -> count`；
- digest collision、digest/fragment 不一致及 motif 数量不一致均要求 fail closed。

因此，**对 production v2，tokenizer binding 不需要重扫 SDF，不需要重新运行 RDKit、linearizer 或 E3FP**。绑定过程只需安全解码 pretokenizer record、读取全局 census 字典，并执行一个冻结、可哈希的 lexeme projection/binding adapter。

该结论仅适用于完整且通过自身契约校验的 v2 产物。production v1、旧 pickle LMDB 或缺少 ordered digest sequence 的记录不能自动升级或混用。

## 3. 总体数据流

```text
complete production-v2 pretokenizer release
  ├─ sharded safe LMDB records
  │    ├─ ordered motif_lexeme_sha256
  │    ├─ motif_atom_indices
  │    ├─ motif_geometry_valid
  │    └─ E3FP / atom provenance
  └─ global motif census
       └─ digest -> exact lexeme -> count

          ↓ frozen lexeme projection spec

phase-specific pure-motif census
          ↓ frozen selection/cutoff/OOV policy
ordered motif vocabulary
          ↓ frozen local T5 base snapshot + stable builder
tokenizer release manifest + saved tokenizer snapshot
          ↓ digest binding table / manifest-bound adapter
tokenizer-bound logical Dataset records
          ↓ deterministic Collator corruption
explicit CE and geometry-MSE batch fields
          ↓
128-record GPU candidate gate
```

最后一项通过时只能记录 `passed_non_release` 或含义等价的状态。只有 data-release manifest 明确完成所有源锁、成员、排除、tokenizer、模型和 checkpoint 门禁后，才允许单独作出 P1 admission 决策。

## 4. Full census 到 ordered motif vocabulary

### 4.1 输入前提

最终词表只能消费完整 release manifest 所绑定的 global census。partial shard 或 128/10k benchmark census 可以用于开发 candidate，但不得成为最终词表来源。

对每个 census row 必须验证：

1. 字段集合严格闭合；
2. `count` 为正整数，累计时使用至少 64-bit 整数；
3. `sha256(motif_fragment.encode("utf-8")) == motif_lexeme_sha256`；
4. 同一 digest 不得对应不同 exact fragment；
5. exact fragment 不得包含 NUL、换行或制表符等会破坏冻结词表格式的字符；
6. census 文件 hash 必须与完整 production manifest 的 artifact binding 一致。

### 4.2 Exact lexeme 不是最终 motif token

production v2 的 exact lexeme 保留分子内局部 anchor，例如 `<0*>`。它是内容寻址与逐记录恢复的依据，但不能直接作为普通 motif vocabulary 的 token，否则同一化学 motif 会因 anchor 编号或位置产生大量语义重复词项，也会改变当前“anchor token + pure motif token”的结构表达。

需要新增并冻结 `motif_lexeme_projection_spec/v1`，其最小行为为：

1. 以原始字符串中出现的顺序提取所有 `<非负十进制整数*>` anchor；
2. 从 exact lexeme 中删除这些 anchor 子串，但不做其他字符串清洗；
3. 不调用 RDKit、不重新解析 SMILES、不改变 Unicode、不做化学 canonicalization；
4. 将删除 anchor 后的 core 外包一层方括号，得到 pure motif token；
5. 保存 projection harness、正则/语法、测试向量和完整脚本 SHA-256。

例如，若 core 本身为 `[NH]`，则外包后可以得到历史约定兼容的 `[[NH]]`。该结果看似存在双层方括号，但不得以人工“美化”为由再次规范化。

不同 exact lexeme digest 投影到同一个 pure motif token 是正常的多对一聚合，而不是 SHA-256 collision。census 应同时保留：

- exact lexeme unique count；
- pure motif unique count；
- 每个 pure motif 的 P1 count、P2 count 和 total count；
- 每个 pure motif 对应的 exact lexeme 数量；
- projection failure 或 special-token collision 数量。

### 4.3 确定性排序与 tie-break

在 selection policy 已决定后，普通 motif token 使用以下全序：

```text
1. selection_score 降序；
2. pure motif token 的原始 UTF-8 bytes 字典序升序；
3. sha256(pure motif token UTF-8) 十六进制升序。
```

第三项通常不会被触发，但用于使排序规则成为明确全序。实现中不得通过 `set -> list(set)`、无序 dict 遍历、locale-sensitive sort 或平台默认编码决定输出顺序。

建议 ordered vocabulary 使用 UTF-8、LF 换行的不可变 TSV，并至少保存：

```text
motif_token    p1_count    p2_count    total_count    motif_token_sha256
```

稳定 tokenizer harness 当前读取 TSV 第一列；其余列用于来源审计。发布物同时记录原始文件 SHA-256、canonical JSON 内容 SHA-256、非空行数、唯一 token 数和重复检查结果。

### 4.4 Special-token overlap

`[.]`、anchor、任务控制符和 T5 sentinel 属于 special-token domain，不得作为普通 motif 再次加入词表。

- `[.]` 只表示断开组分分隔符；
- `<n*>` 只表示分子内局部拓扑 anchor；
- 其余 pure motif 若意外等于 special token 或与 base vocabulary 发生未声明重叠，应 fail closed；
- 只有在单独 allowlist、行为测试和 manifest 中显式记录时才允许例外。

### 4.5 Cutoff、cap 与 OOV

选择规则至少参数化以下字段：

- `discovery_scope`：P1-only 或 P1+P2 permitted-train union；
- `min_count`；
- `max_motif_tokens`，允许 `null` 表示不设上限；
- `selection_score` 及 P1/P2 phase weighting；
- 边界同频 token 的 tie-break；
- `oov_token`；
- P1、P2 各自的 unique/occurrence coverage 与 OOV rate。

在最终裁决前，应从 census 一次性生成候选统计表，例如：

- `min_count = 1, 2, 5, 10`；
- `topK = 20k, 50k, all`；
- 各方案的 vocabulary size、P1/P2 occurrence coverage、P1/P2 OOV rate；
- embedding/LM-head 参数增量和 4090 显存估算。

如果 P2 禁止扩容是主线约束，推荐最终 discovery scope 使用 **P1 admitted membership 与 P2 alignment-train permitted membership 的并集**。如果 P2 census 尚未冻结，则最终 tokenizer 仍被阻断；另一种可选路线是由用户明确接受“P1-only vocabulary，P2 新 motif 走 OOV”，但该路线不能被默认推断。

## 5. Base T5 snapshot 与 tokenizer release

### 5.1 Base snapshot 锁

选择 T5 家族和模型尺寸属于研究/资源决策；一旦确定，则 exact revision 与本地快照锁定属于工程工作。

最终 base tokenizer 必须满足：

- 对应明确模型标识与不可变 commit/revision，而非浮动 `main`；
- 保存本地 snapshot 每个 regular file 的字节数和 SHA-256；
- 保存按相对 POSIX 路径排序的 canonical tree SHA-256；
- 只通过本地目录加载；
- `local_files_only=True`、`use_fast=False`、`trust_remote_code=False`；
- base tokenizer 与后续初始化模型权重来自同一 revision；
- 加载期间禁止网络 fallback。

### 5.2 Special token 固定顺序

若用户没有决定新增 OOV token 或扩大 anchor 预留，R0 harness 当前顺序可作为候选基线：

1. base snapshot 已有 additional special tokens，保持原顺序与 AddedToken metadata；
2. 数字 `0` 到 `9`；
3. `<bom>`、`<eom>`、`[MMM]:`、`[Caption]:`、`[Text2Mol]:`、`[Denoise]:`；
4. `[.]`；
5. `[RESERVED_0]` 到 `[RESERVED_99]`；
6. `<0*>` 到 `<99*>`。

T5 的 `<extra_id_0>` 到 `<extra_id_99>` 必须已存在于 base snapshot，不得重复添加。release manifest 必须保存完整 sentinel token-to-ID map；Collator 必须使用该显式映射，不能假定 `extra_id_0 - counter` 总是成立。

production census/binding gate 还必须统计最大 anchor ID。如果任何允许成员需要 `<100*>` 或更大的 anchor，而当前 special spec 仅预留 100 个，则必须先版本化扩容或制定显式 reject policy，不能静默映射为 `<unk>`。

### 5.3 R0 harness 的可复用边界

`r0_stable_tokenizer_contract.py` 已正确实现以下能力：

- 冻结 base snapshot 离线加载；
- 固定 special-token 注册顺序；
- 按 frozen motif 文件逐行注册；
- 拒绝重复 motif；
- 保存完整 `id_to_token`、`token_to_id` 和 AddedToken metadata hash；
- 跨 `PYTHONHASHSEED` 确定性比较。

但它当前明确标记为 `legacy_vocab_validation_only_not_final_vocabulary_release`，且只生成 sidecar manifest，不保存最终 tokenizer snapshot。因此不能将它的输出直接改名为正式 tokenizer release。

需要单独的 production wrapper 或版本升级，执行：

```text
load frozen base
  -> register frozen specials
  -> register frozen ordered motifs
  -> save_pretrained(new immutable directory)
  -> offline reload saved directory
  -> compare complete semantic mapping and AddedToken metadata
  -> run at least three distinct PYTHONHASHSEED processes
  -> emit tokenizer release manifest
```

### 5.4 Tokenizer release manifest 最小字段

manifest 至少绑定：

- schema version、release ID、candidate/frozen 状态；
- base tokenizer identifier、exact revision、tree hash 与逐文件 hash；
- P1/P2 permitted membership manifest hashes；
- P1/P2 census hashes及其完整性状态；
- lexeme projection spec hash；
- cutoff/cap/phase-weight/OOV policy hash；
- ordered motif vocab 文件 hash、行数和 coverage/OOV 统计；
- tokenizer builder harness hash；
- Python、Transformers、Tokenizers、SentencePiece 等运行时版本；
- special token order、token-to-ID map 和 AddedToken metadata hashes；
- sentinel map 与 anchor map hashes；
- 完整 `id_to_token` payload/hash、`token_to_id` hash、`vocab_size`；
- 最终 `save_pretrained` 输出目录 tree hash 与逐文件 hash；
- 三个或更多不同 `PYTHONHASHSEED` 的 determinism gate 报告及共同 hash；
- `p1_p2_exact_same_mapping=true`；
- `p2_vocab_extension_forbidden=true`；
- 不包含训练准入含义的明确声明。

输出路径必须是新目录，不得覆盖已有 release。

## 6. Digest 到 token ID 的无 SDF 绑定

### 6.1 Digest binding table

从全局 census 与 tokenizer manifest 构建只读 binding table：

```text
motif_lexeme_sha256
  -> exact motif lexeme
  -> ordered anchor tokens
  -> pure motif token
  -> selected token ID 或 frozen OOV ID
```

binding table 必须绑定：

- production full-release root/hash；
- global census hash；
- lexeme projection spec hash；
- ordered motif vocab hash；
- tokenizer contract/root hash；
- OOV policy hash；
- binding harness hash。

运行时不得调用 `MotifTokenizer`、HF 字符串 tokenize、RDKit、linearizer 或 E3FP。motif token ID 只能通过 manifest 中冻结的 token-to-ID map 直接取得。

### 6.2 每条记录的完整性检查

对每条 admitted record 必须验证：

```text
len(motif_lexeme_sha256)
  == motif_count
  == len(motif_atom_indices)
  == len(motif_geometry_valid)
```

每个 digest 必须在 census/binding table 中存在。任何缺失、额外 digest、token collision、anchor 不合法、atom partition 不完整或上游 record hash 不匹配都应立即失败，禁止自动修复或跳到下一条样本。

### 6.3 组分分隔符恢复

production v2 每条记录没有保存 `component_fragment_ranges`。为了不重扫 SDF，可以从 exact lexeme 的 anchor ID 构图恢复：

1. 每个 anchor ID 必须在同一分子中恰好出现两次；
2. 两次必须属于不同 motif ordinal；
3. 两个 motif 之间由该 anchor 连边；
4. 无 anchor 的 motif 是单 motif 连通分量；
5. 图的每个连通分量必须在既有 motif ordinal 顺序中形成连续区间；
6. 在连续分量区间之间插入 `[.]`。

这一推导必须进入 binder spec 并由 128 reference audit 与 linearizer 原始 `fragment_string` 做等价验证。若任何 anchor 只出现一次、超过两次、出现在同一 motif 两次，或连通分量不连续，则必须 fail closed。

### 6.4 Unmasked P1 sequence 与 atom 映射

P1 unmasked token 序列按以下方式拼接：

```text
<bom>
  + [该 motif lexeme 中按出现顺序列出的 anchors]
  + [该 motif 的 pure motif token ID]
  + [必要时的 component separator [.]]
  + ...
<eom>
```

`motif_ordinal_to_unmasked_token_index[j]` 指向第 `j` 个 motif 的 pure motif token，而不是其 anchor。对 `motif_atom_indices[j]` 中每个原子，`atom_to_unmasked_token_index` 均赋值为该 pure motif token 位置。由此可以仅使用 pretokenizer record 完成 atom → motif → token 映射。

推荐首先实现 manifest-bound lazy adapter：训练 Dataset 读取 immutable pretokenizer shard、binding table 与 tokenizer manifest，按需构造 token IDs。这样无需重新物化 337 万条 tokenized records。若后续吞吐测试证明需要缓存，缓存只能是可由上述不可变输入重建的加速层，并拥有独立 manifest/hash。

## 7. Tokenizer-bound Dataset 与 Collator 字段

### 7.1 Dataset 的最小逻辑字段

来源与绑定：

- `member_id`；
- upstream `record_content_sha256`；
- pretokenizer release root/hash；
- tokenizer contract/release root/hash；
- `id_to_token_sha256`；
- projection/binder spec hashes；
- corruption policy hash（若 Dataset/Collator release 已冻结）。

数组与序列：

- `full_input_ids`；
- `unmasked_input_ids`；
- `full_to_unmasked_token_index`；
- `motif_ordinal_to_unmasked_token_index`；
- `token_geometry_valid_mask`；
- `atom_to_unmasked_token_index`；
- `unmasked_atom_attention_mask`；
- 原始、未掩码 `e3fp`；
- 上游 `motif_atom_indices` 和 `motif_geometry_valid` 的 hash binding。

对于纯 P1 motif 序列，如果 `full_input_ids` 与 `unmasked_input_ids` 完全相同，应在 schema 中明确声明二者相等，且 `full_to_unmasked_token_index` 为 identity mapping；不要留下依靠实现猜测的歧义。

### 7.2 Collator 的最小输出字段

- `input_ids`；
- `attention_mask`；
- `labels`；
- `e3fp_input_ids`；
- `unmasked_e3fp_ids`；
- `atom_attention_mask`；
- `unmasked_atom_attention_mask`；
- `atom_to_token_index`；
- `joint_mask_positions`；
- `geo_only_mask_positions`；
- `geometry_input_mask`；
- `geometry_target_mask`。

必须满足：

```text
joint_mask_positions AND geo_only_mask_positions == false
geometry_input_mask == joint_mask_positions OR geo_only_mask_positions
geometry_target_mask == geo_only_mask_positions AND token_geometry_valid_mask
```

其他不变量：

- 只有 `joint_mask_positions` 生成 CE reconstruction labels；
- `geo_only_mask_positions` 保留 2D motif identity，只屏蔽其 3D 输入；
- joint 位置不能在当前 v1 语义下静默加入 MSE target；
- `<bom>`、`<eom>`、anchor、`[.]`、task token、sentinel 和 pad 永不成为 geometry target；
- geometry input mask 必须屏蔽映射到该 motif token 的全部相关原子；
- batch 中无有效 geometry target 时跳过 MSE，不创建零向量伪 target；
- 禁止 legacy `mask_positions` 字段及其合并语义；
- sentinel 使用 manifest 显式 token-ID 列表，并检查数量足够；不得算术递减猜 ID。

### 7.3 持久记录与动态 mask 的契约冲突

当前 `p1_pcqm_geometry_record_schema.json` 的 future tokenizer-bound requirements 将四类 mask 列为 record required fields。若照字面物化，会让每个分子永久只有一次 corruption，不利于预训练中的多轮随机掩码，也混淆不可变数据与 epoch-time transformation。

正式主线应版本化拆分：

- tokenizer-bound storage record 只保存 unmasked sequence、映射、geometry validity 和 tokenizer binding；
- Collator 按冻结的 corruption policy，从 `(global_seed, epoch, member_id)` 或等价无状态 key 确定性生成四类 mask；
- batch 输出接受四类 mask 契约校验；
- 128 candidate gate 可以使用固定 mask fixture 复现测试，但不得据此固化全量训练 mask。

该 schema 调整是正式 P1 前的硬阻断之一；不能通过对旧字段作宽松解释绕过。

## 8. 128 条 tokenizer-bound GPU candidate gate

### 8.1 可以验证的内容

128 candidate 可以验证工程链路：

1. safe payload decode，无 pickle 或 legacy fallback；
2. 上游 source/record/release hash binding；
3. digest → exact lexeme → anchors/pure token → token ID；
4. component separator 恢复与 anchor pair 不变量；
5. motif/atom/token 映射和 geometry validity；
6. 同 seed、同 epoch、同 member 在不同 worker/PYTHONHASHSEED 下 mask 一致；
7. CE labels 可以精确恢复 joint-masked original token；
8. geometry input mask 确实屏蔽对应原子；
9. MSE 只消费 `geometry_target_mask`；
10. zero-target batch 正确跳过 MSE；
11. forward loss、logits、所有参与梯度均为有限值；
12. `total_loss == CE + lambda * geometry_loss`；
13. `lambda=0` 与 `lambda>0` 的梯度路由符合预期；
14. padding 与 batch permutation 不改变单样本有效位置结果；
15. 1–2 次 optimizer step 可完成；
16. 保存并严格重载后 tokenizer/config/hash 一致，eval logits 与 loss 在声明容差内一致；
17. 4090 上记录 peak memory 与单步耗时。

自然的前 128 条未必覆盖边界情况。必须补充确定性合成负例或从 pretokenizer release 中定向选择样本，覆盖：

- digest 缺失或 digest/fragment 不符；
- digest collision；
- disconnected components；
- anchor 只出现一次、超过两次或超过预留上限；
- selected-vocab OOV；
- zero geometry target；
- sentinel 数量耗尽；
- 超长序列；
- atom partition/motif count 不一致；
- 非有限 E3FP/hidden/loss；
- tokenizer hash 或 checkpoint config 漂移。

### 8.2 不能验证的内容

128 candidate 不能证明：

- full release 对 3,378,606 个 source ordinal 的完整覆盖；
- 全量 reject ledger 与 downstream identity exclusion 正确；
- 最终 P1/P2 vocabulary coverage 与 OOV rate；
- 全量 Dataset 吞吐、epoch 时间和多机行为；
- CE+MSE 长期稳定性、representation collapse 风险或最优 lambda；
- MSE 对下游任务有统计显著收益；
- 正式 P1 training admission。

gate 报告应使用 `candidate_status=passed_non_release` 或语义等价字段，并显式记录 `p1_training_admission=false`。

## 9. 旧代码复用边界

### 9.1 不得原样复用

- `dataset/dataset.py::GSMATDataset`：使用 pickle、在线 SMILES 重分词/E3FP fallback、错误后换样、E3FP 宽度静默 pad/truncate，且不了解 v2 hash/schema；
- `dataset/dataset.py::GSMATPretrainingCollator`：使用 legacy merged `mask_positions`，随机性与 seed/epoch/member 未绑定；
- `tokenization/motif_tokenizer.py`：`set -> list(set)` 导致 token ID 不确定，且在线重分词失败时退化为 `<unk>`；
- `model/modeling.py` 当前 MSE block：使用 masked input IDs 构造 target、把 joint mask 纳入 MSE、target 缺失时回退、`nan_to_num`/非有限 loss 静默处理；
- `train1.py` 当前 launcher：`ignore_mismatched_sizes=True`、动态词表扩容、硬编码大 lambda、空 whitelist、未绑定保存 tokenizer。

### 9.2 可有条件复用

- T5 主干结构；
- `GSMATEmbeddings`、`GeoSemanticFusion`、`MoStT5Encoder` 的结构思想；
- 分项 loss 日志方式；
- production v2 safe payload codec；
- R0 stable tokenizer harness 的确定性构造内核。

上述结构性代码仍需由新 candidate 路径增加 schema/hash/range/finite/strict-load 门禁，不能因为“类名相同”直接纳入新主线。

## 10. 三类决策状态

### 10.1 可直接工程冻结

以下项目不需要额外科学假设，可以直接形成版本化契约：

- SHA-256、canonical JSON、UTF-8/LF 编码和不可覆盖输出规则；
- census digest/fragment/count 的严格校验；
- exact lexeme projection 的纯字符串算法；
- pure motif 聚合规则；
- 排序的频次降序与 UTF-8/tokendigest tie-break；
- local-only base snapshot、exact revision、tree/per-file hash；
- 已批准 special token 的固定注册顺序与 AddedToken metadata；
- 显式 sentinel/anchor token-to-ID map；
- digest binding table 与逐记录 cardinality/mapping 校验；
- 无 SDF、RDKit、linearizer、E3FP 重跑的绑定边界；
- joint/geo-only/geometry-input/geometry-target mask 恒等式；
- special/pad 不参与 geometry target；
- zero-target 跳过 MSE；
- strict save/reload、禁止 `ignore_mismatched_sizes`；
- 128 candidate 的 pass/fail 项与 `passed_non_release` 状态。

### 10.2 必须由实验或用户裁决

这些参数会改变模型容量、训练分布或科学结论，不能由工程实现者静默决定：

- vocabulary discovery 使用 P1-only 还是 P1+P2 permitted-train union；
- `min_count`、`max_motif_tokens`；
- P1/P2 frequency weighting 与 selection score；
- 使用 base `<unk>` 还是专用 `<motif_unk>`；
- T5 family/size；
- maximum sequence length 与超长分子的排除/截断/分桶策略；
- anchor 预留是否从 100 扩展及扩展规模；
- joint mask ratio、geo-only mask ratio及 span/单 token masking；
- MSE target 是 online stop-gradient、EMA teacher 还是其他设计；
- MSE target/prediction normalization；
- `lambda_3d`、warmup、调度与损失平衡方案；
- full-vocab 与 cutoff-vocab 的消融设计和最终选择。

### 10.3 正式 P1 前的硬阻断

以下任一未关闭，都不能宣称 P1 已准入：

1. production v2 full release 尚未 complete，或 full manifest 未通过独立校验；
2. P1/P2 permitted membership scope 与 downstream identity exclusion 尚未冻结；
3. 若选择 P1+P2 union，P2 train motif census 尚未完成；
4. motif lexeme projection/binding spec 尚未版本化；
5. cutoff/cap/OOV/source-scope 等科学参数尚未裁决；
6. 真实本地 T5 base snapshot 与 exact revision 尚未冻结；
7. production tokenizer builder/save/reload wrapper 尚未完成；
8. tokenizer release manifest 与三进程 determinism gate 尚未通过；
9. tokenizer-bound storage schema 与 dynamic Collator mask schema 尚未拆分/冻结；
10. 128 candidate Dataset → Collator → CE+MSE → GPU backward → strict save/reload gate 尚未通过；
11. 完整 data-release manifest 未把 source、membership、reject、tokenizer、geometry policy 和 checkpoint prerequisites 统一绑定；
12. 独立 admission decision 尚未明确记录 `p1_training_admission=true`。

## 11. 建议的最小实施顺序

1. 等待并验证 production v2 full release complete；partial 期间可用 v2 128 benchmark 开发 binder。
2. 冻结 `motif_lexeme_projection_spec/v1`，先对 128 做 digest、anchor、component 和 pure-token reference audit。
3. 从 global census 生成 phase-specific pure-motif census 与多组 cutoff/cap coverage 报告。
4. 由用户/实验计划裁决 discovery scope、cutoff/cap、OOV、anchor reserve 和 max length。
5. 冻结 exact T5 snapshot，实施 production tokenizer wrapper，保存并重载最终 tokenizer。
6. 跨至少三个 `PYTHONHASHSEED` 运行 determinism gate，生成 tokenizer release manifest。
7. 实施 manifest-bound lazy Dataset 与显式四-mask Collator；不修改旧主干。
8. 在 4090 上运行 128 candidate gate 与负例门禁，状态只记录为 `passed_non_release`。
9. 关闭全量 membership、reject、overlap 与 data-release manifest 门禁后，再作独立 P1 admission 裁决。

## 12. 最终审查结论

production v2 的 ordered motif digest 与全局 exact-lexeme census 已经提供了正确的内容寻址桥梁。只要新增冻结的 lexeme projection/binder，就可以直接把 immutable pretokenizer shards 绑定到 frozen tokenizer，无需再次扫描 9.7 GB SDF，也无需重新计算 motif 或 E3FP。这是当前最重要的时间优化。

但该优化只消除了重复数据处理，不会消除必要的语义裁决。词表来源、cutoff/OOV、序列长度、mask 比例与 MSE 设计仍需可追溯的实验或用户决定。实施中必须保持三层状态分离：

```text
geometry-only pretokenizer release complete
    != tokenizer release complete
    != 128 GPU candidate gate passed
    != P1 training admitted
```

任何阶段都不得通过重命名 manifest、复用旧 checkpoint shape 或启用宽松加载来跨越下一阶段门禁。
