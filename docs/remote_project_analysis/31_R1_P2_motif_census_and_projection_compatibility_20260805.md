# R1 P2 motif census 与 P1/P2 投影兼容性审计（2026-08-05）

状态：P2 candidate motif census 全量双跑 PASS；P1/P2 direct projection-domain compatibility = false。`P1_ADMISSION=false`、`P2_ADMISSION=false`、`TOKENIZER_FREEZE_PERMITTED=false`。

## 1. 真实数据源与信任边界

实际读取文件：

`/root/autodl-tmp/3D-MoIT/3d-mol-dataset/pubchem/pretrain/phase2_pubchem_ready.lmdb`

该文件与 `autodl-fs` 规范副本逐字节一致：

- bytes：1,909,297,152
- SHA-256：`465d89f4aafb36043a5964441feffceb3e3e6493fe2ffee9d53190ec7587d5e5`
- payload records：301,655
- metadata keys：空集

为了使用快速盘而不改变证据对象，运行前重新读取并计算了 tmp 原件的完整 SHA；它与两次既有 P2 identity extraction 对 `autodl-fs` 副本记录的完整 SHA 相同。

正式配置绑定：

- source copy manifest：`a8a9a082bee04a77c153548e460692c14e627748e0b7daa25deb9f2930803cd9`
- pickle trust basis：`2f6890ae865d7a871f1cc6771783c5d91bc164a63e54ba45b11767662c2f66a7`
- P2 source lock：`ce9fe30657f3cd1e7ac55229fbc45bbdabddf124a936a8d781bb2eb41189602c`
- retrospective legacy motif spec：`f8bc45d09c4989af44faeb6ad6dc239f5a966da11746f7aef107789e7ffc50c4`

两次 identity extraction 只证明完整文件、301,655 条记录、CID/key、身份所需字段和空 metadata；它们不被写成已经证明 census 十字段闭集。十字段闭集由本次 census 在解码过程中逐条检查并进入 closed reject ledger。

旧 LMDB 使用 pickle value。运行严格限定于上述完整 SHA，首次 `pickle.loads` 前先校验整文件，读事务结束后再次计算整文件 SHA；源文件只读且不需要网络。该授权不外推到任何其他 pickle 文件。

## 2. 全量 census 与 fresh-process 复跑

baseline：

`/root/autodl-tmp/most-t5-r1/p2-motif-census-v1-baseline-20260805T113300Z`

持久化 baseline：

`/root/autodl-fs/most-t5-r1/reports/cpu-p0-20260805/p2-motif-census-v1-destination-20260805T114500Z/baseline`

fresh-process rerun（`PYTHONHASHSEED=17`）：

`/root/autodl-tmp/most-t5-r1/p2-motif-census-v1-rerun-seed17-20260805T113600Z`

rerun verification：

`/root/autodl-tmp/most-t5-r1/p2-motif-census-v1-rerun-report-seed17-20260805T113600Z.json`

复跑 receipt/report 已持久化到：

`/root/autodl-fs/most-t5-r1/reports/cpu-p0-20260805/p2-motif-census-v1-destination-20260805T114500Z/rerun_evidence`

结果：

| 指标 | 数值 |
|---|---:|
| source payload | 301,655 |
| admitted | 301,655 |
| rejected | 0 |
| exact motif occurrences | 4,560,114 |
| unique exact motifs | 114,736 |
| unique pure motifs | 99,442 |

六个确定性产物在两次独立进程中全部逐字节一致：

- `membership.jsonl` SHA-256：`9065b811f95f6e8694ca43468f87e1c09b34497cca5089bd6492f293e52069f9`
- `reject_ledger.jsonl` SHA-256：`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- `record_projection.jsonl` SHA-256：`4d9a524a70cc6ecc98e12a1ebd71cef1448ac20d689a72163d799a75c718f7b5`
- `motif_census.jsonl` SHA-256：`8321cd9d0864456ab0e1b54d5af48218e125f6fb81d2159dd68cacfe1cba10f6`
- `pure_motif_census.jsonl` SHA-256：`09b784ddcc3759ddceb17d07821346decddbfef2566277e05545e6b19ffed6bf`
- `anchor_summary.json` SHA-256：`9534566fa3c43c20045369c530b1bc72fc1af3bf5b73bee6164867530c11d074`

两次 logical derivation SHA-256 均为 `f31bed80d9f433e28746df6aefc955f81d46e15bf45a600bd43873237511abaa`。baseline receipt 文件 SHA-256 为 `6415e13edef6bddeeaf3ef4e58d63b5a0ff092a9653e03d0742d9750b3df8f8f`；rerun report 文件 SHA-256 为 `d9715a04d0b73423183b941e16264be37c6d96995bc0b30a41e25baba60bdaf1`。

## 3. P1/P2 projection-domain compatibility

报告：

`/root/autodl-fs/most-t5-r1/reports/cpu-p0-20260805/p2-motif-census-v1-destination-20260805T114500Z/compatibility_audit_v1_2.json`

报告文件 SHA-256：`5fea4af5b35b9d8ef7834404555e58f5555d61db8f5fc1f14e18f8debfbb1e48`

经收紧并独立接收的 audit script SHA-256 为 `953ec81e0ed4488564375e39e626bdd4a5d426377f65493e98c8e294a9007259`，compatibility contract SHA-256 为 `5457a0f669b9e9392eef82d8dc4d1057c54c390181dfc2cf9e92346beb798908`；远端 Linux/Python 3.8 测试为 8/8 PASS。

结论：`direct_projection_domain_compatible=false`。报告给出五项原因：

1. P1/P2 component-boundary semantics 尚未证明相同；
2. P1 与 P2 的 linearization semantic mapping 尚未证明；实现 SHA 是否相同只作为 provenance fact，本身不被当成科学不兼容理由；
3. P2 component-local anchor multiplicity 不满足 P1 molecule-global pair rule；
4. P2 motif sequence 的原 producer 未知；
5. P2 保留 `@`、`/`、`\\` 立体标记，而当前 P1 census 域中这些标记计数为零。

词面事实：

| 指标 | P1 | P2 | shared unique |
|---|---:|---:|---:|
| unique exact lexeme | 441,769 | 114,736 | 2,448 |
| unique pure motif | 214,554 | 99,442 | 3,982 |
| exact occurrence shared coverage | 15.6318% | 86.2893% | — |
| pure occurrence shared coverage | 93.3111% | 87.4272% | — |

高 pure-occurrence coverage 说明删除旧 anchor 后，常见化学片段在频次层面存在较强重合；它支持进一步做统一表示实验，但不能证明 anchor、component、stereo 或 producer semantics 相同。尤其不能用该覆盖率直接批准 P1/P2 vocabulary union。

## 4. 决策与下一步

当前不应直接构建 P1+P2 联合 tokenizer。可接受的下一步是二选一并做消融：

1. `p1_only`：正式词表只由 P1 molecule-native 表示发现，P2 通过 OOV/通用 SMILES 子词路径进入；
2. `p2_relinearized`：从 P2 原始 molecule/fragment 事实重新生成与 P1 同一 producer、同一 global anchor、同一 stereo policy 的序列，再重跑 census 和 compatibility。

顶刊级实验至少比较 `P1-only`、`P1 + legacy-P2（仅作为失败/风险对照）`、`P1 + relinearized-P2`，同时报告词表大小、OOV、token length、P1 几何任务、P2 对齐任务和下游泛化。legacy-P2 不能在 compatibility=false 时被作为正式主线训练输入。

上述报告均为 candidate evidence；它们不批准 tokenizer freeze、training launcher 或 P1/P2 admission。

## 5. P1 并行提速的最终基线验收

保留的原串行 extractor 在约 1 小时 45 分后自然完成；8-worker 版本完整运行约 11 分 24 秒，实测墙钟加速约 9 倍。二者的执行合同与 extractor SHA 不同，因此不要求 receipt/manifest 文件自身相同；要求相同的 core deterministic boundary 已全部一致：

| 指标 | serial | parallel |
|---|---:|---:|
| rows bytes | 1,348,489,549 | 1,348,489,549 |
| row count | 3,365,577 | 3,365,577 |
| rows SHA-256 | `44f98d41de48a7b81b3315b79cb80ab8fbb63a4d43792c45edd5569e7cbe47c4` | 相同 |
| key-LF SHA-256 | `159b1688effe34d75be6613368cc9a0e08bbea1c398baccba392a1541de1b001` | 相同 |

serial：

`/root/autodl-fs/most-t5-r1/reports/cpu-p0-20260805/pcqm-identity-collection-v1-destination-20260805T095747Z`

parallel：

`/root/autodl-fs/most-t5-r1/reports/cpu-p0-20260805/pcqm-identity-collection-parallel-v1-destination-20260805T105845Z`

因此 parallel extractor 已证明为语义保持的性能实现，可替代串行实现执行后续同类全量身份抽取；串行产物继续保留作为独立基线，不做删除。
