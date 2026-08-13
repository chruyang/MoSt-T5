# Anchored 3D-MotifT5：原子状态 reducer 筛选结果与正式架构裁决

日期：2026-08-11

## 1. 结论先行

本轮在完全相同的标准 QM9 molecule-disjoint split、五个性质、T5 初始化、
训练预算和 carrier/endpoint 接口下，依次比较了五种 atom-state reducer。

结果否定了“越接近 3D-MolT5 的原子 shell 均值就越适合 motif 载体”的假设：

- 固定四槽均值、单标量 L0/high 权重和单线性投影均明显退化；
- 不带 level embedding 的最小 MLP 可以恢复 B2D，但 F3D 仍劣于 B2D；
- 只恢复 level embedding 后，B2D 达到本轮最佳，但 F3D 仍劣于 B2D；
- 因而 level embedding 对区分 E3FP shell 有价值，但它本身不足以组织出稳定的 3D 增益；
- 正式主线停止继续堆叠 V10/V11。历史 Anchored V4
  `l0_l123_mean + level/presence/role + MLP` 暂时保留为唯一通过标准 QM9
  方向门和 component ablation 的 atom encoder。

这不是把 V4 中每个部件都声明为必要或创新；它只说明目前没有更简单的候选在同一门上取代它。

## 2. 冻结实验合同

所有新格共同使用：

- QM9 train/dev/test：`7971 / 992 / 995`；
- targets：`mu, alpha, r2, u0, u0_atom`；
- microbatch `256`，gradient accumulation `1`；
- `30` epochs，`960` optimizer updates；
- AdamW，LR `3e-4`，warmup `100`，随后 cosine；
- 完整 encoder finetuning；
- train seed `20260810`，adapter seed `20260809`；
- B2D 与 F3D 参数量和数据顺序配对；
- 同时报告 `both / carrier-only / endpoint-only / zero`。

该筛选比较的是架构方向，不是最终 QM9 SOTA；数值均为五个任务 standardized MAE 的宏平均，越低越好。

## 3. 主结果

| atom reducer | B2D test | F3D test | F3D 相对 B2D | 裁决 |
|---|---:|---:|---:|---|
| 历史 Anchored V4 | 0.38271 | **0.35781** | **-6.51%** | 保留的已通过基线 |
| V5 fixed-four mean | 0.59317 | 0.62884 | +6.01% | 淘汰 |
| V6 adaptive scalar L0/high | 0.44777 | 0.52029 | +16.20% | 淘汰 |
| V7 one linear L0/high | 0.58118 | 0.53304 | -8.28% | 方向正确但绝对性能严重退化，淘汰 |
| V8 minimal phi, no level | 0.37545 | 0.42164 | +12.30% | B2D 恢复，F3D 失败，淘汰 |
| V9 level-aware phi | **0.32567** | 0.36751 | +12.84% | B2D 最好但 3D 门失败，淘汰 |

V9 的关键 component 结果为：

| cell | both | carrier-only | endpoint-only | zero |
|---|---:|---:|---:|---:|
| B2D | **0.32567** | 0.84795 | 0.64473 | 0.91865 |
| F3D | **0.36751** | 0.79271 | 0.57917 | 0.95684 |

两格都证明 carrier 与 endpoint 的联合路径可被模型消费；但 F3D 没有优于同接口、
同参数量的坐标无关 B2D，因此不能把 V9 的收益解释为三维表征增益。

## 4. 对 L0、L1--L3 的解释

本轮支持以下更严谨的表述：

1. L0 是 atom identity/local 2D context，不是纯 3D state；
2. L1--L3 是继承身份并经空间排序增强的 environment state，也不是纯几何坐标；
3. 固定把 L0--L3 当同质槽平均，对 motif 级模型并不充分；
4. level embedding 明显改善 B2D，说明 shell level 的语义不能轻易抹平；
5. F3D 仍未改善，说明还需要 V4 中的某些组合因素（presence、role、非线性容量或其交互），
   但本轮不能识别其中哪一个单独必要。

因此正式论文不把 V4 atom encoder 单独包装成核心创新，也不把其内部部件逐一赋予未经验证的化学含义。
核心创新仍限定为 stereo-free semantic motif phrase、atom-to-motif carrier 与
attachment-specific ordered-anchor endpoint；V4 是实现该接口的当前经验证编码器。

## 5. 为什么停止继续做 reducer 变体

本轮已经形成一条有信息量的最小递增链：

```text
fixed mean
 -> scalar weight
 -> one linear projection
 -> minimal nonlinear phi
 -> add level embedding
```

继续单独恢复 role、presence 或扩大 MLP 会逐步回到 V4，却不会回答预训练准入所需的新问题。
正式预训练前更高价值的工作是冻结一次性词表、全量 anchored cache、Phase I/II 任务混合与
可恢复 runner，而不是继续在 8k QM9 子集上搜索编码器。

## 6. 正式预训练准入边界

当前冻结为：

- atom encoder：历史 Anchored V4；
- geometry provider：F3D E3FP；
- model surface：stereo-free pure motif + ordered anchors；
- geometry injection：motif carrier + exact attachment endpoint；
- GraphPorts：仅作离线无损审计/重建层，不作模型主语言；
- reference-aligned V5--V9：保留为研究控制，不进入正式 checkpoint。

启动正式预训练前仍必须完成：

1. 在正式预训练语料上一次性冻结 macro/chemical-lexer/tokenizer；
2. 从权威 release 派生无 JSON 热路径的 anchored tensor cache；
3. 冻结 Phase I/II 数据成员、任务 mixer、优化预算和 dev 门；
4. 发布与 tokenizer/cache/union-init 严格绑定、可从 checkpoint 精确恢复的 runner。

不阻塞启动、可在训练期间执行：正式语料的 Lmax=3 覆盖率审计、更多 reducer 解释性消融、
retrosynthesis 可选任务和额外随机种子复验。

## 7. 证据位置

远端权威结果：

- `/root/autodl-tmp/qm9-reference-reducer-v1-20260811`
- `/root/autodl-tmp/qm9-linear-reducer-v1-20260811`
- `/root/autodl-tmp/qm9-minimal-phi-v1-20260811`
- `/root/autodl-tmp/qm9-level-aware-phi-v1-20260811`

历史 V4 结果见文档 77。V5--V9 的实现和测试位于 `most_t5_next/p2/`，它们不改变
anchored data surface、tokenizer 或 tensor cache。
