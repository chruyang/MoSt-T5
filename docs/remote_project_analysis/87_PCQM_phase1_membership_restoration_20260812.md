# PCQM Phase-I 下游排除回加与统一成员闭合（2026-08-12）

状态：`PASS`。旧 final-v4 与 production-v2 均保持不可变；5,510 条历史下游保护排除已全部恢复到当前 Phase-I 成员策略。

## 1. 为什么要恢复 5,510 条

历史 `paper-scope-final-v4` 使用如下规则：只要 PCQM 记录的 canonical non-stereo connectivity 出现在任一已登记下游 validation/test 集合，就从预训练成员中排除。该规则得到：

- production-v2 admitted：3,365,577；
- final-v4 permitted：3,360,067；
- downstream-overlap excluded：5,510；
- excluded unique connectivity：2,789；
- 同时命中多个保护集合的成员：540。

该差集是下游评估保护策略，不是化学、坐标、E3FP 或 wire payload 拒绝。当前冻结的训练边界已经改为：

- **Phase I** 学习 fragSMILES/motif 语法、结构与几何表征，使用所有化学准入成员，不再执行历史“所有下游 validation/test 并集”排除；
- **Phase II** 是带文本的表征阶段，继续通过独立派生视图执行 ChEBI-20 **test-only** canonical non-stereo connectivity 排除；
- 下游模型选择和最终评估仍只能读取各任务冻结的 train/validation/test，不能把 Phase-I 回加描述成“下游无泄漏训练集”。

因此恢复 5,510 条不会改变分子准入事实，只撤销了一个不再适用于 Phase-I 的历史成员过滤器。

## 2. 恢复方法

实现：

`most_t5_next/r1/overlap/restore_phase1_downstream_exclusions_v1.py`

输入均为已冻结产物：

1. final-v4 `excluded_member_ledger.jsonl`；
2. immutable production-v2 的 136 个 shard；
3. stereo-free + SDF-authoritative E3FP supplement。

工具不重算化学、不复制 33 GB payload，也不通过新规则猜测成员。它逐条执行：

```text
historical excluded member_id
-> source ordinal / production shard
-> shard membership disposition=admit
-> payload_index key/content/wire hash
-> LMDB raw bytes existence + byte count + SHA-256
-> explicit restoration ledger
```

输出目录：

`/root/autodl-tmp/most-t5-r1/derived/phase1-unified-pcqm-membership-v2`

其中：

- `restored_downstream_members.jsonl`：5,510 条逐成员恢复证据；
- `source_segments.jsonl`：两个不可变数据段（production-v2 与 stereo supplement）；
- `manifest.json`：计数、来源 SHA-256、策略和闭包不变量。

恢复账本保留每条历史命中的任务、split、collection ID，同时记录 production shard、storage key、content SHA、wire bytes 和 wire SHA。因此历史 final-v4 的排除事实没有被抹除；它只是不再决定当前 Phase-I 成员资格。

## 3. 全量结果

远端 CPU 全量执行约 20 秒，结果：

| 项目 | 数量 |
|---|---:|
| 历史 final-v4 permitted | 3,360,067 |
| 本次恢复 downstream-overlap | 5,510 |
| strict production-v2 admitted | 3,365,577 |
| stereo recovery supplement | 12,978 |
| 当前统一 Phase-I members | **3,378,555** |
| PCQM4Mv2 train-3D source | 3,378,606 |
| 仍未解决 | **51** |

闭包：

```text
3,360,067 + 5,510 = 3,365,577
3,365,577 + 12,978 = 3,378,555
3,378,606 - 3,378,555 = 51
```

独立只读复核：

- 恢复 ID 与历史 excluded ledger：5,510/5,510 完全相等；
- 恢复集合与 stereo supplement：交集 0；
- production LMDB payload：5,510/5,510 存在，byte count 与 wire SHA 全部匹配；
- 恢复账本 SHA-256：`4ac5f811e84dcbcdce4bdb79a7651ada82905df1aad05466331941396b6495b8`；
- unified manifest SHA-256：`eb2c780b0f8b709c455f69daa718de8538a0d32a55bc4b1e519f33fab1a3d76a`；该最终 manifest 另显式绑定恢复脚本 SHA-256。

剩余 51 条不是下游保护过滤：

- `PCQM_SDF_CSV_CONNECTIVITY_MISMATCH`：33；
- `HYDROGEN_PROJECTION_RESIDUAL_H`：18。

这 51 条继续隔离；不得用 fallback 静默放行。

## 4. 后续消费边界

正式 fragSMILES/geometry tensor cache 应读取 `source_segments.jsonl`，按以下固定顺序物化：

1. production-v2 中全部 admitted records（包含本次恢复的 5,510 条）；
2. stereo recovery supplement 的 12,978 条。

无需再读取 final-v4 permitted 文件拼装 Phase-I，也不得把 5,510 条重复追加到 production-v2——它们本来就在 production-v2 内。恢复账本的作用是撤销旧 membership filter 并提供审计证据，不是第三个 payload 数据段。

Phase-II 则从其独立 PubChem/text 源重新派生成员，并继续执行已冻结的 ChEBI-20 test-only connectivity exclusion；该规则与本次 Phase-I 恢复不冲突。
