# Anchored motif fallback 单后缀语法冻结（2026-08-10）

## 裁决

Stage 3 正式生成语法冻结为：

```text
macro motif:    anchors* + one_macro_token
fallback motif: anchors* + chemical_tokens+ + <MOST:FALLBACK_END>
```

`fallback_single_suffix` 是唯一进入 tokenizer snapshot、训练缓存和生成目标的
boundary mode。`implicit_sidecar` 只保留为 encoder 长度下界；全 motif 前缀、fallback
单前缀和双边界均不进入训练候选。

本次只冻结语法，不提前冻结 macro 来源。以下两种 macro policy 仍是 Stage 3 候选：

- `pretrain_train_only`：面向通用 checkpoint；
- `balanced_pretrain_plus_registered_downstream_train`：只有在 ChEBI-20 train 确实进入
  Phase I/II exposure 时才可采用，并必须标记为 task-aware specialist。

## 不变量

1. macro token 自定界，其 carrier 就是 macro token，不增加边界。
2. fallback carrier 是短语末尾的 `<MOST:FALLBACK_END>`。
3. anchor token 位于 identity span 之外，继续作为 attachment endpoint carrier。
4. motif identity corruption 的最小单位是完整 identity phrase：macro 为一个 token；
   fallback 为 `chemical_tokens+ + suffix`。禁止只遮 fallback 的一部分。
5. suffix 是 ordinary token，参与 CE 和 corruption；不得注册为 Hugging Face special token。
6. 生成解码时缺失 suffix、空 chemical span、未知 chemical token 或 chemical span 内出现
   anchor/macro 均为非法序列，不回退到 `<unk>`。
7. Phase I 第一次 optimizer update 后禁止继续 `add_tokens` 或改变 token ID。

## 全量证据

PF1 的 33,600 条记录、241,799 个 motif occurrence 已完成 standalone replay：

- macro coverage（pretrain-only 512 macro）：95.7316%；
- fallback occurrence：10,321（4.2684%）；
- suffix surface 长度 `p50/p95/p99/max = 25/38/43/56`；
- 33,600/33,600 精确解码，0 条超过 512；
- 相比“每 motif 一个边界”，边界 token 数下降 95.73%；
- prefix-only 与 suffix-only 逐记录等长，但 suffix 保留旧版 phrase-end carrier 语义。

balanced specialist 的最大长度同样为 56，0 条超过 512。因此无需再为
prefix-vs-suffix 消耗 GPU；后续实验只裁决 macro 来源/预算及训练任务。

## 冻结产物

- model-surface contract：
  `most_t5_next/r1/tokenizer/anchored_motif_model_surface_v1.py`
- tokenizer plan schema：`most-t5-next/anchored-tokenizer-plan/v2`
- 当前计划目录：
  `tmp/anchored_tokenizer_plans_v4_frozen_fallback_suffix_b0693bb`
- plan manifest SHA-256：
  `3ff533f6448955c0931dbe4940b2019d6914c70b4d0b911e511f7b79e1ac4588`
- 两个 plan 的 final vocab size 均为 33,041，ordinary additions 均为 941；
- 完整回放报告：
  `tmp/anchored_model_surface_analysis_v6_frozen_suffix_b0693bb.json`
- 完整回放报告 SHA-256：
  `cf8a650e03227cc7be8d420d87f28c0f231a9ccf940ac59a4a7a60d7b97a6f95`
- 已验证共享 tokenizer snapshot：
  `tmp/anchored_tokenizer_snapshot_v2_remote_frozen_suffix_b0693bb`
- snapshot manifest SHA-256：
  `a3edbf55389abbd11962ca9639225b86e5933d0618fcb070748493d279b4d374`
- snapshot tree SHA-256：
  `1cc2e8c6ca3987d15d3558648e535f1e7157f4192cb4e2511ba2ad8ac47bc6a0`
- 固定运行时：Transformers 4.45.2、SentencePiece 0.2.0、slow T5 tokenizer；
- 离线重载确认 `len=33041`、`<MOST:FALLBACK_END>=32100`、
  `<extra_id_0>=32099`，自然文本 `C.O` 仍为 `[205,5,667]`。

tokenizer plan manifest 必须同时绑定 model-surface contract、registry hashes、plan semantic
hash 和 macro policy。旧 v1 plan bundle、implicit plan 或 prefix plan 不得由新版 candidate
tokenizer builder 接受。

## 下一门

边界语法已经关闭，不再作为实验变量。Stage 3 剩余工作是裁决 macro policy/预算，然后只
构建一个共享 token-surface snapshot；若不同 policy 的 token 字符串集合相同，应共享同一
snapshot，并分别绑定 motif identity→macro token 的语义 registry，避免重复 tokenizer 和模型
初始化。这里“共享 snapshot”只表示 token 字符串和 ID 序列相同，不表示两个 registry 的
macro 语义相同：当前两个 512-macro registry 只有 413 个 identity 交集，且只有 6 个 rank
恰好对应同一 identity。任一 checkpoint lineage 必须在 embedding 初始化前选择并绑定一个
registry，之后禁止切换映射。

构建期间还修复了一个纯性能问题：旧实现对每个 base token 都重新调用一次
`get_vocab()`，形成二次复杂度；现改为一次读取后线性比较。相同远端构建从超过 8 分钟
未完成降至约 5 秒完成，验证内容和结果未改变。

## Macro 方法裁决

general mainline 已冻结的是“train-only 高频 whole-motif macro + 无损 chemical lexer fallback”
的方法，而不是最终 macro 数量。512 仅为 PF1 candidate budget；正式 Phase-I 前须在完整
pretraining train corpus 上比较至少 512/2,048/4,096/8,192，并联合报告 occurrence/type
coverage、seen/unseen motif、序列长度、softmax/参数成本、吞吐和下游语义任务。ranking 仅来自
完整 Phase-I pretraining train corpus，按 occurrence 降序、UTF-8 identity 稳定破同分。
ChEBI-20 等已注册 downstream train 只用于覆盖率审计，不改变 general registry；balanced
policy 仅保留为显式 specialist 分支，不能在同一 checkpoint lineage 中途切换。

PF1 样本证据中，512 macro 对 pretrain train/dev 的 occurrence coverage 分别为
95.77%/95.39%，对 ChEBI-20 train 仍覆盖 94.41%；未选 motif 全部走无损 lexer，不产生
`<unk>`。相对 256，512 将 pretrain mean identity tokens/motif 从 1.771 降到 1.633；相对
1024，512 少一半 macro rows，而 1024 只把该均值进一步降到 1.501。T5-v1.1-base 的
input embedding 与 lm_head 不绑定，因此 512 macro 实际新增 786,432 个 vocab 参数，而不是
旧报告只计 input embedding 得到的 393,216。

机器可读政策位于
`most_t5_next/p1/support/anchored_vocabulary_policy_v1.json`。这里冻结的是方法；预算和具体
identity registry 都必须从完整 Phase-I corpus 裁决/重算。最终 tokenizer 仍需先完成所有
已注册 training corpus 的 anchor-domain census；当前 512-macro、33,041-vocab snapshot 是
已验证 candidate，不具有 training admission。
