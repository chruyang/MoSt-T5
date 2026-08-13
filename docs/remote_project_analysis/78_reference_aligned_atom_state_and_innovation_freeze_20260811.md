# Anchored 3D-MotifT5：原子状态参考边界与创新焦点冻结

日期：2026-08-11

## 1. 本次裁决

Anchored V4 的 `l0_l123_mean` 已证明当前 carrier+endpoint 路径可以工作，但其 atom
encoder 同时包含 level embedding、L0/高层双槽、presence bits、core/attachment role
embedding 与两层 MLP。它继续作为**已完成的研究基线**，不直接等同于正式预训练架构。

正式候选先收敛为一个 reference-aligned atom-state path：

1. 每个原子保留固定四个有序 E3FP 槽 `L0--L3`；
2. 缺失 shell 仍为 `-1`，映射到固定零 embedding；
3. 四个槽使用一个共享 E3FP embedding 表并做固定四槽算术平均；
4. 不加入 level embedding、presence bits、attachment-role embedding 或 atom MLP；
5. `atom_is_attachment` 只用于 endpoint 路由与合同验证；
6. 后续 motif owned pooling、carrier 注入、anchor endpoint 精确寻址及 component ablation
   保持不变，以便把差异严格限制在 atom-state encoder。

公式为：

\[
u_i=\frac{1}{4}\sum_{l=0}^{3}E(\widetilde{fp}_{i,l}),
\qquad
E(\mathrm{missing})=\mathbf 0.
\]

这里不是 masked mean。即使 L3 缺失，分母仍为 4；这与 3D-MolT5 官方
`molecule_fp_embed_tokens(molecule_fp_ids + 1).mean(dim=-2)` 的数值语义一致。

## 2. `role` 的冻结含义

当前 `atom_is_attachment` 只表示：该原子是否为一个跨 motif SINGLE bond 的端点，也即
是否被某个 ordered anchor occurrence 寻址。它不是元素、杂化、手性、官能团或内禀3D
属性；同一分子采用不同 motif partition 时，该标记可以改变。

因此正式候选中：

- endpoint token 必须唯一寻址一个 active attachment atom；
- 多 anchor motif 继续通过 `endpoint_token_to_atom` 保留逐 occurrence 地址；
- Boolean role 保留在权威 sidecar 和 tensor cache 中用于验证；
- role 不再转成 learned vector，也不进入 carrier 或 endpoint 的数值状态。

这减少了由 partition-derived 2D 标记进入几何通道的捷径，也避免与显式 anchor 语法和
endpoint address 重复编码。

## 3. 为什么不是机械照搬

原子 shell embedding 与缺失值语义采用 3D-MolT5 参考，是为了把低层表征设为已验证、
可解释的控制变量。项目创新从该 atom state 之后开始：

```text
fixed-four-shell atom E3FP state
              |
              +--> motif-owned aggregation --> motif carrier
              |
              +--> ordered-anchor address ----> endpoint token
```

与相邻工作的边界为：

- 3D-MolT5：原子 E3FP 与 SELFIES 原子 token 对齐，没有 atom-to-motif/anchor 双路接口；
- FineMolTex：有 atom-to-motif pooling 和 motif-text 对齐，没有 E3FP/T5 anchor endpoint；
- t-SMILES/CAMT5：有 fragment/anchor 语言，没有逐 anchor 的 atom E3FP 地址；
- FACET/Frag2Seq：已有 fragment-aware 3D 建模，因此本项目不得主张“首次结合 fragment、
  3D 与语言模型”；它们也没有本文的 stereo-free semantic motif phrase、离散 E3FP
  carrier 与 ordered-anchor endpoint 的生成式 T5 接口。

正式创新表述冻结为：

> We introduce an anchored 3D-motif interface for a generative molecular T5.
> A stereo-free local motif phrase carries lexical identity, ordered anchors
> carry inter-motif topology, and one atom-centred discrete E3FP state is reused
> both to form a motif-level carrier state and to condition the exact attachment
> endpoint addressed by each anchor occurrence.

这属于**表示与接口组织创新**，不把共享 E3FP embedding、固定 shell mean、motif
fragmentation 或通用线性投影单独声称为创新。

## 4. V4 结果如何保留

文档 77 的标准 QM9 结果继续有效：在 V4 复杂 atom encoder 下，F3D test standardized
macro MAE 为 `0.35781`，相对参数匹配 B2D 改善 `6.51%`；both 优于 carrier-only、
endpoint-only 与 zero。这证明：

- anchored carrier+endpoint 数据与梯度路径可用；
- L3 在当前 V4 组织方式下不可直接删除；
- F3D 在至少一组真实3D敏感属性上有初步增益。

它不再冻结“正式 atom encoder 必须是 `l0_l123_mean + role + MLP`”。文档 77 第83行的
“冻结 `l0_l123_mean`”由本文件修订为“冻结为历史最佳 V4 基线”。

## 5. 下一步最小实验

不重跑此前全部阶段，只在同一标准 QM9 数据、split、target、T5初始化和训练预算下运行：

1. B2D-reference：坐标无关 atom-state provider + fixed-four mean path；
2. F3D-reference：E3FP + fixed-four mean path。

当前 V4 B2D/F3D 数值直接作为已完成对照，不再重复。正式候选的准入条件是：

- F3D-reference 相对 B2D-reference 保持方向一致的 aggregate 改善；
- no-geometry 数值等价、缺失 shell 固定零、role 翻转不改变 atom memory；
- endpoint 仍只能寻址 attachment atom；
- both/carrier-only/endpoint-only/zero 四种输入诊断继续可执行。

若 reference candidate 保持主要收益，则删除正式主线中的 V4 atom MLP/role/presence/level
分支；若明显退化，只按顺序增加一个变量：先比较固定 `L0 + mean(L1:L3)` 双流，再考虑
level embedding。role embedding 与无依据的多层 atom MLP 不作为优先回加项。

## 6. L0 身份语义与 `Lmax=3` 覆盖率待验证项

L0 与 L1--L3 不应继续被写成四个完全同质的“空间层”。按当前锁定的 E3FP
实现，L0 是由原子初始 invariants 哈希得到的中心原子身份/局部二维上下文，包含原子序数、
度、氢数、形式电荷、质量差和环标记等。本文后续允许简称其为 **atom identity/context**，
但不得把它解释为完整、可逆的原子身份标签。

L1--L3 则递归消费上一层 identifier、邻接/键关系与立体空间排序。它们应表述为
**identity-conditioned, spatially enriched environment states**，而不是 pure spatial states：
即使移除显式 L0 输入，高层 identifier 仍继承 L0 的二维身份信息。因此：

- `motif-only -> +L0 -> +L1:L3` 必须分层报告；
- L0 带来的改善只归因于原子身份/组成上下文补充；
- F3D 的三维主张仍必须来自同接口 B2D、aligned/zero/扰动和3D敏感下游对照；
- 正式 reference fixed-four mean 保留为控制，若需要提高细致度，只优先增加一个从
  fixed-four 数值等价点初始化的全局 L0/high-shell 权重，不恢复 role/presence/多层 MLP。

正式预训练训练集上的 `Lmax=3` 覆盖率另列为低优先级 CPU 审计。它不阻塞 tokenizer、
cache、union-init 或正式预训练启动，安排在正式预训练运行期间利用空闲 CPU 执行。审计至少
报告：

1. 逐 atom 的最高可用 level 为 0/1/2/3 的计数与比例；
2. 逐 molecule 的 `any-L3`、`all-atoms-L3`、L3 atom fraction 与 atom-count 分层；
3. 逐 motif 的 L3 覆盖率，并按 motif size、ring/非ring、core/attachment 地址分层；
4. train/dev 的覆盖漂移，以及与 PF-1、QM9 已有统计的差异；
5. L3-present 与 L3-absent 子群的下游指标，以及“L3 内容置零”和“仅 availability
   可见”的分离诊断。

E3FP 因子结构收敛或重复 substructure 抑制而没有 L3 时，这是指纹生成语义，不自动视为
坏样本，也不得为了追求覆盖率而补造 identifier。审计直接读取正式 tensor cache 中已有的
`[atom,4]` 行，不重新计算构象、不改变训练成员、不占用 GPU；输出独立 manifest 并绑定正式
训练 membership/cache hash。

## 7. 当前放行边界

本文件放行 CPU 实现、单元测试和未来两格短程 GPU 比较；不因架构公式更简洁而自动放行
full-scale预训练。数据、词表、anchored surface和训练cache无需重建；两种atom encoder
消费同一 `[atom,4]` E3FP与地址sidecar。

## 8. 直接参考

- 3D-MolT5 official code, fixed-shell embedding mean:
  https://github.com/QizhiPei/3D-MolT5/blob/82dbe088e424f19fa713dbd657f5235990bd324f/3d_molt5/utils/FPT5EncoderStack.py
- 3D-MolT5 atom-shell construction:
  https://github.com/QizhiPei/3D-MolT5/blob/82dbe088e424f19fa713dbd657f5235990bd324f/3d_tokenization/3d_tokenize.py
- FineMolTex atom-to-fragment pooling:
  https://doi.org/10.5281/zenodo.15501037
- t-SMILES: https://doi.org/10.1038/s41467-024-49388-6
- FACET: https://openreview.net/forum?id=013f4015fb03c0772407553acc0690618b0a974e
- Frag2Seq: https://openreview.net/forum?id=mMhZS7qt0U
