# R0.3 — 可复现 tokenizer 合约审计

**状态：** 离线确定性 harness 已通过；真实项目 tokenizer release 仍被阻断。

**远端 gate 报告：**
`/root/autodl-fs/most-t5-p0/reports/r0-stable-tokenizer-contract-smoke-20260731b/gate/r0_stable_tokenizer_determinism_gate.json`

本项只新增 `p0_validation/` 下的 sidecar 工具；没有修改原始
`MotifTokenizer`、训练代码、LMDB 或 checkpoint，也没有下载任何模型或数据到
本机。

## 发现的问题

现有 `MotifTokenizer` 将 motif 放入 Python `set` 后再调用
`add_tokens(list(set))`。`set` 的迭代顺序受 `PYTHONHASHSEED` 影响，因此相同
motif 对应的 token ID 不能作为稳定的语义合约。旧 P1 checkpoint 又没有保存
完整 tokenizer 映射；即使 embedding shape 相同，也不能证明其行与 motif 的
语义相同。

## 新的 sidecar 合约

`r0_stable_tokenizer_contract.py` 不导入 live `MotifTokenizer`，而是：

1. 只接受已存在的本地 tokenizer snapshot 目录，强制
   `local_files_only=True`、`trust_remote_code=False`；不能因传入 Hub 标识符而
   发生网络下载。
2. 以固定顺序注册数字、任务、结构、reserved 和 anchor special tokens。
3. 显式要求 T5 的 100 个 `<extra_id_*>` sentinel 已存在于 base snapshot。
4. 按冻结 motif 文件的逐行顺序加入普通 motif，从不通过 `set` 输出顺序。
5. 保存完整 `id_to_token` 映射及其 hash，同时记录 `AddedToken` 的
   normalized/lstrip/rstrip 等元数据，避免“ID 相同但编码行为不同”。

独立的 `r0_stable_tokenizer_determinism_gate.py` 会启动新的离线 Python 进程，
以不同 `PYTHONHASHSEED` 比较完整映射与相关 hash。

## 已完成的 smoke 证据

使用远端构建的合成、完全离线 T5 tokenizer snapshot，三个独立进程
`PYTHONHASHSEED=0, 1, 271828` 均通过：

| 项目 | 结果 |
| --- | --- |
| 词表大小 | 326 |
| 三进程 `id_to_token` hash | `c874dc81087c5b6bbaf1321235e7d5585900a066e475e6d68c7027bbc2ed25dd` |
| 所有被比较字段 | 完全一致 |
| 网络模式 | Transformers/HF/Datasets 全部 offline |

这证明的是**新 harness 的确定性**，不是项目真实词表已经可发布。

## 真实 tokenizer release 的阻断条件

- 未找到可冻结的真实 T5 tokenizer snapshot，故不能对真实 20k/P2 词表运行
  gate；不得以联网 Hub 名称替代该 snapshot。
- P1 与 P2 的最终训练 membership、下游 identity-exclusion 报告尚未冻结。
- motif 来源必须成为有序、可哈希的产物，明确 canonicalization、频率 cutoff、
  “频率降序 + 词典序 tie-break”、OOV 策略。
- 显式 H、E3FP、atom-to-motif mapping 策略尚未冻结。
- 合约通过后还需要模型集成 gate：tokenizer 长度、embedding resize、
  `config.vocab_size` 和 strict checkpoint load 必须同时验证。

在这些条件满足前，不能语义复用旧 P1 checkpoint，也不能在 P2 再扩容词表。
