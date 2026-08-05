# 11 T5 中 CE + MSE 的适配性、风险与相关先例

> 审查日期：2026-07-17  
> 结论范围：只回答“CE 与 MSE 能否在 T5 中共同训练、会不会损害模型、分子语言模型是否存在相近实现”。当前 E3FP target 本身是否合理另见文档 10。

## 1. 结论

1. **CE + MSE 与 T5 在结构上兼容。** T5 并不限制只能有 CE；encoder hidden state 可以连接任意可微辅助 head。
2. **它不会必然破坏模型，也不会天然提升模型。** 是否损害生成能力由辅助目标质量、梯度方向、梯度尺度、作用参数和训练阶段决定。
3. **已有直接和相邻先例，但没有找到与当前方案完全一致的分子 T5 实现。** 3DMolFormer 直接在分子 3D 序列上使用 token CE+坐标 MSE；DynaBERT/PKD 直接在 Transformer 中联合离散监督与 hidden-state MSE；但没有一篇同时采用当前 E3FP、motif pooling、T5 和 same-network target 的组合。
4. **当前 `lambda_3d=500` 不能仅凭 loss 数值判断。** 应比较 CE 与加权 MSE 在共享参数上的梯度范数和夹角。
5. **最可靠的判断方法是 CE-only 对照加梯度诊断。** 若 CE+MSE 未超过 motif+E3FP+CE，则 MSE 没有保留价值。

## 2. 为什么结构上兼容

标准 T5 训练目标为 token-level cross-entropy：

\[
L_{CE}=-\sum_t \log p(y_t\mid y_{<t},x).
\]

当前几何 head 从 encoder hidden state 预测一个连续 target：

\[
L_{3D}=\frac{1}{N}\sum_i\|\hat z_i-z_i\|_2^2.
\]

联合目标为：

\[
L=L_{CE}+\lambda L_{3D}.
\]

这不会改变 T5 decoder 的输出空间、teacher forcing 或自回归生成定义。反向传播时：

- CE 更新 decoder、cross-attention、encoder 和输入表示；
- MSE 更新 geometric head，以及与它相连的 encoder/融合模块；
- 两者在共享参数上形成联合梯度。

因此风险来自联合优化，不是来自 T5 接口或数学形式不允许。

## 3. 何时会破坏 CE/生成效果

设共享参数上的两个梯度为：

\[
g_{CE}=\nabla_\theta L_{CE},\qquad g_{3D}=\nabla_\theta L_{3D}.
\]

总梯度为：

\[
g=g_{CE}+\lambda g_{3D}.
\]

对一次小学习率更新，CE 的一阶变化近似为：

\[
\Delta L_{CE}\approx-\eta\left(\|g_{CE}\|^2+\lambda g_{CE}^{T}g_{3D}\right).
\]

由此得到：

- `cos(gCE,g3D)>0`：辅助目标与 CE 局部协同；
- 接近 0：主要学习不同方向，通常较温和；
- 小于 0：存在梯度冲突；当加权 MSE 足够大时，一步更新可能直接增大 CE。

在 `gCE·g3D<0` 时，为保持局部一阶 CE 下降，需要近似满足：

\[
\lambda < \frac{\|g_{CE}\|^2}{-g_{CE}^{T}g_{3D}}.
\]

这只是局部一阶条件，但它说明：**500 是否安全取决于梯度，而不是 MSE 的显示数值是否很小。**

### 高风险条件

1. MSE target 与生成任务无关、含噪或错误。
2. `lambda` 使 `λ||g3D||` 长期远大于 `||gCE||`。
3. 两个梯度长期负 cosine。
4. MSE 直接强约束 encoder 全部层，削弱 T5 已有语言表示。
5. target 自身漂移或可塌缩，模型通过缩小 latent 而非学习几何来降低 MSE。
6. 在 caption、text2mol、C4 等无同等几何含义的任务上强制应用 MSE。
7. 只观察总 loss；总 loss 下降时，CE 或生成指标可能已经恶化。

### 可能带来收益的条件

1. target 稳定，并确实包含 CE 难以获得的三维结构信息。
2. MSE 只作用于有可靠三维数据和明确 mask 的位置。
3. CE 与 MSE 的梯度规模平衡，或使用 GradNorm/PCGrad 等方法处理。
4. 几何 head/adapter 吸收主要 MSE 压力，而不是让全部 T5 参数被强制重排。
5. 在独立验证集上同时改善 CE、结构恢复和下游三维敏感指标。

## 4. 文献中的相近实现

| 相似层级 | 工作 | 实际做法 | 与当前方案的差别 |
|---|---|---|---|
| 分子 CE + MSE（直接） | 3DMolFormer | 双通道 3D 分子序列预训练使用 token CE + 系数乘以坐标 MSE；本地 paper/3DMolFormer_ICLR2025.pdf 第 6 页式(2) | target 是可观测坐标，backbone 是 GPT；它支持联合损失本身，不证明 E3FP latent target |
| Transformer CE + hidden MSE（直接） | DynaBERT | fixed teacher 的 soft CE 与 embedding/hidden-state MSE 共同优化；按项量级选 (1, 0.1)；本地 paper/supplementary/DynaBERT_2020_NeurIPS.pdf 第 4 页式(3) | encoder-only 压缩任务；支撑权重校准，不支撑分子几何语义 |
| hard CE + normalized hidden MSE（直接） | Patient Knowledge Distillation | 任务 hard CE、soft CE 和归一化 hidden-state MSE 共同优化 | 分类蒸馏与外部 teacher；不是生成式 T5 |
| E3FP + T5 | 3D-MolT5 | E3FP 离散为原子级 3D token，与 SELFIES embedding 相加；使用标准 CE 完成去噪和翻译 | 最接近输入形式，但没有额外 latent MSE，也明确不预测多分量 E3FP token |
| LM + 多目标 | 3D-MoLM | Stage 1 对 Q-Former 使用 molecule-text matching、contrastive、captioning；后续阶段用标准 LM loss | 证明多目标可帮助分子-LM 对齐，但采用分阶段训练，不是同一 T5 内 CE+MSE |
| 图-LM + 多目标 | MolCA | 图 encoder + Q-Former + LM，使用跨模态对齐/生成任务 | 有生成与跨模态目标，但不是 E3FP latent MSE |
| 对比 + CE | FineMolTex | 先做 contrastive warm-up，再联合 contrastive alignment 与 masked-token CE | 证明分子-文本模型可以联合不同 loss，并通过 warm-up 与权重搜索控制训练 |
| 分子连续 latent 重建 | GraphMVP | 2D/3D 对比学习加 variational representation reconstruction | 有连续表示重建，但没有 T5 生成 CE |
| 分子 latent MSE | C-FREE | predictor 用 MSE 预测 target encoder 的子图 latent；target encoder 由 EMA 更新 | 与当前 MSE 最相似，但使用独立 EMA target，且没有 T5 CE |
| masked latent target 设计 | data2vec | full-input EMA teacher 的归一化连续 target 由 masked student 回归；采用 Smooth L1 | 不与 CE 联合；为 EMA、归一化和稳健回归提供跨模态机制证据 |
| T5/encoder-decoder 表示匹配 | Hidden-state distillation | 在 T5/BART 等 encoder-decoder 上把任务损失与 teacher hidden-state matching 结合 | 证明表示级辅助目标与 T5 可兼容，但 target 是外部 teacher，不是同模型 E3FP 分支 |

### 严格结论

文献能够支持以下三句话：

1. T5 可以接受表示级辅助监督。
2. 分子语言模型可以联合生成、匹配、对比或 masked modeling 目标。
3. 分子 2D/3D 表示可以使用连续 latent reconstruction/MSE。

文献不能直接支持：

> “把同一模型生成的 motif-pooled E3FP latent 作为 MSE target，再以 500 倍权重与 T5 CE 同时训练，一定能提升模型。”

这正是本项目必须验证的创新假设。

## 5. 对“会不会破坏模型”的最低判定标准

不能只看训练总 loss，至少需要同时记录：

### 生成能力

- validation CE / perplexity；
- motif exact reconstruction；
- molecule validity；
- caption BLEU/ROUGE/METEOR 或领域指标；
- text2mol exact、Morgan、validity；
- C4/text validation loss，检查语言能力遗忘。

### 几何表示

- raw MSE 和加权 MSE；
- target/pred latent 方差与有效秩；
- 不同构象/分子的可分性；
- gate 均值、分布以及趋近 0/1 的比例；
- 构象敏感下游任务结果。

### 优化诊断

- `||gCE||`；
- `||λg3D||`；
- `cos(gCE,g3D)`；
- encoder 前层、后层、E3FP embedding 和 fusion 层分别统计；
- 梯度裁剪前后的范数；
- CE-only 与 CE+MSE 的相同 step 验证曲线。

若出现以下任一情况，应判断当前 MSE 正在损害模型：

1. validation CE 持续劣于 CE-only，且没有三维任务补偿收益；
2. 文本/分子生成指标下降；
3. latent 方差或有效秩接近 0；
4. gate 普遍关闭，说明模型回避 E3FP；
5. `λg3D` 长期支配共享参数且梯度 cosine 为负。

## 6. 推荐验证顺序

### 第一步：只回答兼容性

使用同一初始化、同一数据子集和相同步数：

```text1
A: motif + E3FP + CE
B: motif + E3FP + CE + 当前 MSE，lambda=1
C: motif + E3FP + CE + 当前 MSE，lambda=10
D: motif + E3FP + CE + 当前 MSE，lambda=100
E: motif + E3FP + CE + 当前 MSE，lambda=500
```

记录 CE、MSE、梯度范数、梯度 cosine 和验证指标。不要先用完整训练规模。

### 第二步：回答 target 稳定性

```text
B1: 当前 same-network detach target
B2: frozen target encoder
B3: EMA target encoder
```

若 B2/B3 明显优于 B1，说明问题不是 CE+MSE 不适配，而是当前 target 构造不稳定。

### 第三步：回答是否值得保留

在 Phase 2 和下游任务上比较：

```text
CE-only E3FP input
vs.
最佳 CE+MSE 配置
```

只有后者在多个 seed 上稳定改善至少一个三维敏感任务，且不显著伤害 caption/text2mol/C4，才能宣称 3D Loss 有效。

## 7. 当前建议

1. 不应因为“T5 默认用 CE”而排除 MSE；二者数学和实现上兼容。
2. 不应直接使用 `lambda=500` 作为默认正确配置。
3. 先以 CE-only 的 E3FP 模型作为强基线。
4. 第一轮只在小规模数据上跑 `λ=1/10/100/500`，测共享参数梯度。
5. 将 MSE 限定在 Phase 1 的可靠三维样本和 mask 位置；C4/text-only 任务不加。
6. 至少加入 EMA/frozen target 对照。
7. 评价标准必须包含生成能力保持，而不能只看 3D MSE 下降。

## 8. 主要依据

- 3D-MolT5，本地 `paper/3dMolt5.pdf`，尤其 PDF 第 5-6 页。
- 3D-MoLM，本地 `paper/3D-MoLM.pdf`，尤其 PDF 第 3、5-6 页。
- FineMolTex，本地 `paper/finemoltex.pdf`，尤其 PDF 第 4、14 页。
- C-FREE，本地 `paper/C-FREE.pdf`，尤其 PDF 第 3-4、9 页。
- 3DMolFormer，本地 paper/3DMolFormer_ICLR2025.pdf，PDF 第 6 页，式(2)。
- DynaBERT，本地 paper/supplementary/DynaBERT_2020_NeurIPS.pdf，PDF 第 4 页，式(3)。
- Patient Knowledge Distillation，Sun et al. (2019)，PDF 第 3 页，式(7)–(8)。
- data2vec，本地 paper/supplementary/data2vec_2022_ICML.pdf，§3.3–3.4。
- Dasgupta & Cohn (2025), *Improving Language Model Distillation through Hidden State Matching*, ICLR 2025。
- Chen et al. (2018), *GradNorm*, ICML 2018。
- Yu et al. (2020), *Gradient Surgery for Multi-Task Learning*, NeurIPS 2020。
