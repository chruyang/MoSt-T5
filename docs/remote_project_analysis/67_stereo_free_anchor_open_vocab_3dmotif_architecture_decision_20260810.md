# Stereo-free anchor + open-vocabulary 3D motif 架构裁决（2026-08-10）

> 状态：现行架构决策；在模型输入表示、立体信息职责和 GraphPorts 角色上取代文档 50--66 的相关主线结论。历史实验及其数值继续有效，但必须按本文重新解释。
>
> motif 原子分区本身的代码差异、药物化学/化学生物学证据与冻结边界见 [文档 68](68_motif_partition_chemical_biology_literature_and_freeze_decision_20260810.md)。该裁决保留当前 ring/non-single-bond union 分区，不把切分算法比较列为正式训练前置实验。

## 0. 结论

MoSt-T5 不应继续把 GraphPorts 当作 T5 的训练语言，也不应原样退回“词表外 pure motif 变成 `<unk>`”的旧实现。现行方案冻结为：

> **ordered anchors 表示 motif 间连接；stereo-free pure motif 表示二维身份；stereo-aware atom E3FP 表示给定构象的局部三维与立体状态；高频 pure motif 使用宏 token，长尾 pure motif 使用无 `<unk>` 的可逆 chemical lexer/subword；GraphPorts 或 RDKit graph 仅留在离线审计层。**

该方案保留初版最正确的两个思想：

1. 提取锚点是为了让多个连接变体共享同一个 pure motif，而不是丢弃连接位置；
2. 从 motif 文本移除 `@`、`@@`、`/`、`\` 是为了让立体状态由 E3FP 单独承载，而不是无意损坏数据。

正式全量训练暂停。下一步只做 CPU 表面派生、开放词表和无泄漏审计。

## 1. 初版锚点语义的重新确认

旧 tokenizer 对一个 logical motif 执行：

```text
exact anchored motif
    -> 按从左到右顺序抽取全部 <n*>
    -> 删除 anchor 字符串，保留原位置产生的 () slot
    -> 输出 ordered anchors + pure motif
```

例如：

```text
[C(<3*>)(<8*>)N] -> <3*> <8*> [C()()N]
```

解码时先积累当前 motif 的所有 anchor，再按 `()` 的从左到右顺序填回；前置和后置 anchor 也由同一确定性规则处理。同一 global anchor ID 在两个 motif 中各出现一次，恢复时连接两侧 attachment atom。一个 motif 含多个 anchor 是正常状态，不是异常路径。

### 1.1 不可跨越的 local motif phrase

模型表面不能把 anchors 放成远离其所属 motif 的独立拓扑流。一个 logical motif 必须编码成不可跨越、内部顺序固定的局部短语：

```text
<motif> <a0> <a3> [C(<slot>)(<slot>)N]
```

其中：

- 所有 anchor occurrence 紧邻其 pure motif；
- anchors 与 `<slot>` 均按原 fragment 从左到右的 slot 顺序排列；
- span corruption、截断和 batching 不得把 phrase 拆开后分别重排；
- pure motif identity 是语义 carrier，anchor 是该 identity 的结构修饰符/局部指针；
- 一个 phrase 可以含零个、一个或多个 anchors。

macro motif 已由一个 opaque token 完整承载身份，本身是自定界 phrase，不再添加边界。只有由多个 chemical token 展开的 fallback motif 使用一个 `<fallback_end>` 后缀：`anchors* + CHEM+ + <fallback_end>`。macro/anchor/fallback/chemical 四个 token namespace 两两不相交，decoder 在 anchors 后若遇 macro 就立即结束该 phrase，否则持续读取 chemical tokens 直到后缀。单前缀也可唯一解码且 token 数相同，但会把 carrier 移到 chemical identity 之前；单后缀让 macro token 与 fallback carrier 都位于 phrase 末端，最接近初版 `anchors* + one motif token` 的 mapping 与几何注入边界。完全隐式的 span-sidecar 版本只保留为 encoder 长度下界；生成时没有输入 sidecar，不能作为统一可独立解析的输出语言。

当前正式 topology augmentation 已进一步保存：

- `motif_slot_anchor_ids`；
- `motif_slot_atom_indices`；
- `motif_slot_source_atom_indices`；
- `cross_motif_bonds`。

因此，模型表面不需要再用 GraphPorts edge/radix/byte token 重复声明同一拓扑。

这一简化有两个强制前提：

1. 当前 CAMT5-derived partition 只能切开 `SINGLE/STEREONONE` 跨 motif 键；若未来 partition 允许其他键型，anchor surface 必须显式携带键属性，不能假装仍可无损恢复；
2. anchor ID 是**分子内、由 canonical linearization 确定的局部指针**，不是跨分子的化学类别。现有 `<0*>...<99*>` 只应表达局部配对/顺序；不得为数据集中每条边扩展一个全局 token，也不得把编号大小解释成化学意义。

## 2. 三个严格分离的表示平面

### 2.1 Identity plane：去立体 motif 身份

进入 T5 的 motif 身份仅包含：

- 元素、形式电荷、同位素和氢计数等稳定化学身份；
- 非立体键级与局部拓扑；
- pure motif；
- ordered anchors 与 slot 顺序；
- motif 顺序和分子边界。

不得包含：

- `@` / `@@`；
- `/` / `\`；
- R/S、E/Z 或其他直接立体标签；
- isomeric GraphPorts motif identity；
- 从坐标直接计算、足以泄露目标的文本描述。

这不是声称 identity 与 state 在统计上完全独立。E3FP 的初始原子不变量仍包含元素和拓扑信息；本文只冻结**接口职责的因子化**。

### 2.2 State plane：stereo-aware E3FP

现有 E3FP 生产参数保留 `stereo=True`。对模型原子 `i` 的 shell IDs：

```text
u_i = ShellEncoder(E3FP[i, 0:4], shell level, atom role)
```

E3FP 是旋转/平移不变、构象相关、身份条件化的离散局部环境状态，不是连续坐标，也不是具有数值距离的标签。folded ID 之间不能做 raw-ID MSE。

### 2.3 Audit plane：完整立体化学真值

权威记录继续保留：

- 原始 SDF 与坐标；
- isomeric canonical identity；
- RDKit atom/bond stereo；
- atom mapping、motif partition 和 cross bonds；
- E3FP 参数及内容哈希；
- GraphPorts/RDKit strict reconstruction 证据。

Audit plane 不直接成为 T5 token。移除模型文本中的立体标记不等于从数据资产删除立体真值。

## 3. Anchor occurrence 就是 endpoint carrier

不再引入独立 endpoint token。每个 anchor occurrence 已唯一绑定一侧 attachment atom，因此在 embedding 层直接融合：

```text
x_anchor(s) = LN(e_anchor_pointer(k) + alpha * W_endpoint * u_attachment(s))
```

其中同一 `<k*>` 的两个 occurrence：

- 共享 anchor-ID embedding，声明它们属于同一跨 motif 连接；
- 分别读取各自 attachment atom 的 E3FP，保留两端不同的局部三维状态。

`e_anchor_pointer(k)` 只取有限、确定性的 molecule-local pointer vocabulary；其化学信息来自所在 motif/slot 与 attachment atom，而不是数字 `k` 本身。后续可消融成“统一 `<anchor>` embedding + pairing sidecar”，判断局部编号 embedding 是否有价值，但首版保留旧 tokenizer 已验证的 ordered-ID 解码语义。

pure-motif carrier 负责 motif core 状态：

```text
g_m = AttnPool({u_i | i is a core atom of motif m})
x_m = LN(e_motif + beta * W_motif * g_m)
```

若 motif 没有 core atom，才回退到全部 owned atoms。这样 attachment 状态主要由 anchor 承载，避免在 motif mean 和 endpoint 中机械复制同一信号。

首个基线使用固定 `alpha/beta` 或固定混合比例，避免自由 gate 直接归零。学习 gate 只能作为后续消融。

## 4. 开放 pure-motif 词表

### 4.1 高频路径

训练集高频 stereo-free pure motif 使用一个 opaque macro token：

```text
<3*> <8*> <MOST:M:00127>
```

macro 的频率、排序和 merge 学习只使用 train split；dev/test 不影响词表拟合。

### 4.2 长尾路径

词表外 motif 不得变成 `<unk>`，也不使用 GraphPorts UTF-8 byte surface。它进入带边界的组合路径：

```text
<3*> <8*> <mf> [ C ( <slot> ) N = C ] </mf>
```

fallback 分三层：

1. stereo-free pure-motif chemical lexer；
2. 在 lexer token 上拟合的小型 train-only BPE/Unigram merge；
3. 有限原子/键/分支/环号/电荷/`<slot>` 词法作为绝对兜底。

chemical lexer 必须覆盖 bracket atom、普通元素、键、分支、环号、形式电荷、同位素和 slot。BPE 只压缩，不承担可逆性；即使零 merge，也必须逐字符串往返且 `<unk>` 计数为零。

可以先探测原始 T5 SentencePiece 是否对所有 pure-motif ASCII surface 精确往返；若其 normalization 改变任何字符，则不用它作为权威 fallback，改用专用 lexer。

### 4.3 待确定问题：chemical lexer 的实现边界

本文已经冻结“长尾 pure motif 必须零 `<unk>`、可逆且不依赖 GraphPorts byte surface”，但**尚未冻结具体 lexer/tokenizer 算法**。当前需调查和实验比较的候选为：

| 候选 | 实现 | 主要优点 | 主要风险 |
|---|---|---|---|
| `T5-SP` | 直接使用 T5 原生 SentencePiece 对带边界的 stereo-free pure-motif surface 编码 | 不增加化学词表；全部 embedding 已预训练 | normalization、跨化学符号切分和未知字符策略可能破坏逐字符串可逆性；token 边界不具备明确化学解释 |
| `CHEM-LEX` | 按 bracket atom、普通元素、键、分支、环号、形式电荷、同位素和 `<slot>` 做确定性最长匹配 | 语法边界明确；零 merge 仍可逆；易做覆盖证明 | 序列可能较长；需要自行冻结 grammar、优先级和异常策略 |
| `CHEM-LEX+BPE` | 先由 `CHEM-LEX` 产生可逆基础 token，再只在训练支持集上学习有限 BPE/Unigram merge | 保留兜底可逆性，同时压缩高频局部模式 | merge 预算和训练语料范围会影响覆盖；必须证明 merge 只压缩、不改变解码语义 |

chemical lexer 的最低合同为：

1. 输入先转换为本文定义的 canonical stereo-free pure-motif surface；lexer 不负责互变异构、芳香化或立体归一化；
2. `<slot>` 是独立结构符号，不能与相邻原子、环号或电荷粘连成不可解释 token；
3. bracket atom 内的元素、同位素、氢数、形式电荷等必须无损；`@`、`@@`、`/`、`\` 等立体标签应在上游 identity projection 后不存在，若意外出现则拒绝而非静默词法化；
4. 编码和解码必须逐字符串固定点一致，不能只比较 RDKit 同构；
5. 任意支持域输入均不能产生 `<unk>`；未知或非法字符必须显式拒绝；
6. token 注册顺序、merge 排序、词法优先级和全部依赖版本均进入 tokenizer contract；
7. opaque whole-motif macro 必须保留其 canonical surface、内容 SHA 和 lexer 展开，才能独立复核 macro 与组合路径语义相同。

原始 T5 词表能否直接承担 fallback 不能只凭“它能输出这些 ASCII 字符”裁决。CPU 探针必须在同一批 canonical pure motifs 上比较三候选的：逐字符串 round-trip、`<unk>`、mean/P95/P99/max 长度、超过模型上限比例、token 边界稳定性和构建/解码吞吐。只有 `T5-SP` 同时通过零 normalization drift 与零 `<unk>`，它才可作为最薄基线；否则 `CHEM-LEX` 是权威兜底，BPE/Unigram 只作为其上压缩层。

### 4.4 待确定问题：最终词表是否使用下游任务训练集

这里必须区分三种时间边界，不能统称为“使用下游词表”：

1. **预训练支持集构词：**只用 Phase-I/Phase-II 预训练 train support 统计 macro 和 merge，之后所有下游冻结同一 tokenizer；
2. **预训练前的 task-aware union：**在最终 Phase-I tokenizer 冻结前，将已登记下游任务的 **train split**（例如 ChEBI-20 train）加入词表学习支持域，并让这些 token 在 Phase I/II 的真实分子语言任务中获得训练；validation/test 不参与；
3. **预训练后扩词：**加载已完成的分子预训练 checkpoint，再为 ChEBI 等任务 `add_tokens/resize_token_embeddings` 并依赖短程下游微调学习新行。

当前相邻正式实现主要支持前两类中的“先冻结化学词表、再进行领域预训练”：3D-MolT5 在 T5-v1.1 基础上加入 SELFIES/control vocabulary 后进行分子多任务预训练；BioT5/BioT5+ 先冻结自然语言--SELFIES联合词表再预训练；CAMT5 的 fragment vocabulary 和 FineMolTex 的 motif/processed graph 也在训练前准备。当前尚未找到足够接近的分子生成先例，证明“在下游阶段新增大量 motif 输出 token，只靠短程 SFT”稳定有效。因此：

- 候选 3 **不进入首版正式方案**，不得在下游训练中动态改变 tokenizer length 或 token ID；
- 候选 1 与候选 2 保持为待实验裁决；
- 无论选择哪一项，最终 tokenizer 都必须在 Phase I 第一个 optimizer update 前冻结，Phase II 和全部下游只读加载。

使用下游 train 构词在监督 specialist 任务中不等同于 validation/test 泄漏，但会改变实验含义：

- 若使用 ChEBI-20 train 学习 whole-motif macro 或 merge，结果应称 **task-aware vocabulary / supervised specialist**；
- 若要报告 frozen-transfer 或 zero-shot retrieval，目标任务的 train/validation/test 均不能影响其词表 token 选择；
- validation 可以在预注册的若干 K/merge budgets 中选择超参数，但不能贡献新 token identity；test 只作最终评估；
- 仅提前注册 token、却不让它在预训练任务中出现，不能解决冷启动。若采用 task-aware union，相应 pure-motif phrase 必须进入 Phase-I/II 的合法训练 exposure；否则应保留组合 fallback，而不是增加未训练的 softmax 行。

首轮词表裁决实验应固定相同 base T5、模型参数预算、成员顺序和训练 token exposure，比较：

| 方案 | macro/merge 学习域 | 回答的问题 |
|---|---|---|
| `V-pretrain` | 仅预训练 train support | 统一冻结词表的开放域迁移能力 |
| `V-pretrain+downstream-train` | 预训练 train + 已登记下游 train；严格排除 valid/test | task-aware union 是否值得在最终预训练前一次性纳入 |
| `V-lexer-only` | 无 whole-motif macro，只用可逆 lexer（可配固定 merge budget） | 宏词带来的收益是否超过单纯组合编码 |
| `V-length-control` | 与 macro 方案接近的 token 长度预算，但不按完整 motif 合并 | 区分 motif 语义收益与“只是序列更短” |

至少报告：各 split 的 motif occurrence/molecule coverage、fallback 比例、seen/unseen motif 分层、输入/目标长度分布、词表参数与 softmax 成本、训练吞吐、ChEBI-20 validity/canonical exact/motif precision-recall，以及 QM9/其他任务是否因 task-aware vocabulary 发生负迁移。最终政策不是“覆盖越多越好”，而是在**统一性、序列长度、token 可学习频次、下游收益和 zero-shot 声明边界**之间作预注册 Pareto 裁决。

## 5. 训练任务、阶段与立体输出边界

### 5.1 当前 geometry 定义对任务方向的约束

本文保留的 E3FP 定义是：**由二维身份条件化、对给定构象/立体状态敏感的 atom-centered 离散局部环境状态**。它不是 identity-free geometry。因而：

- `geometry -> motif identity` 在信息上可能可学，因为 E3FP 本身含元素和拓扑；
- 但该方向不能证明模型学习了 3D，且把“状态反演身份”设为主目标与身份--状态因子化的叙述不一致；
- `G-only -> motif` 从现行推荐任务降为探索/对照，不进入正式首版主线；
- 若未来把 geometry 重新定义为同时包含身份与状态的完整 structural token，才可像 3D-MolT5 一样把 `3D -> 1D` 当通用跨视图翻译，但必须放弃其能单独证明3D的说法。

### 5.2 正式主线保留初版两阶段，而非实验性 S/G 顺序

本项目初版实现的阶段语义是：

- `train1.py / Phase 1`：在 PubChemQC 分子上只路由 MMM，学习 motif/anchor 分子语言；E3FP fusion 从开始即存在，并附带 latent geometry 辅助损失，因此它实际是“语法/结构基础 + 初始3D对齐”，不是纯文本语法；
- `train2.py / Phase 2`：从 Phase-1 checkpoint 继续，加入 MMM、caption、text2mol 与 C4 denoising，目标是跨模态语义和下游表征能力。

这个“先分子结构语言基础、再跨模态/表征提升”的课程思想保留。初版思路实际综合了两个相邻来源：

- **3D-MolT5** 提供原子 E3FP 与分子 token 对齐，以及 1D denoising、1D+3D denoising、3D-to-1D 和 molecule--text translation 的任务先例；但其官方实现是在同一次 multi-task pre-training 中联合这些任务，随后再 instruction tuning，并非本项目的两段预训练日程；
- **FineMolTex** 提供 atom--motif incidence、由 atom GNN 表征池化 motif 表征，以及分子--文本对比和重要性引导的 motif/text masking 先例。其正式代码从已有 GraphMVP atom encoder 与 SciBERT 等基础表征出发进行细粒度跨模态预训练，也没有先训练一种自回归 motif 语法、再切换到第二阶段。

因此，本项目的两阶段不是对任一论文日程的复刻，而是把两类思想组合成适合生成式 motif T5 的课程：Phase I 先让新增的 motif/anchor 语言和初始几何接口稳定可学；Phase II 再借鉴 3D-MolT5 的跨视图生成任务和 FineMolTex 的细粒度 motif--text 对齐思想提升表征。FineMolTex 支持的是 **Phase II 的细粒度对齐内容**，不是“必须分两阶段”的直接证据。

正式主线调整为：

1. **Phase I：motif syntax and structural foundation**
   - 冻结一次 tokenizer、local motif phrase、anchor 语法和 atom/motif mapping，Phase II 不再扩词表或改变 token ID；
   - 以 stereo-free local motif phrase denoising 为主；
   - motif identity corruption 与 anchor-pair corruption 分开采样，首版不同时抹除二者；
   - geometry adapter 从 Phase I 开始就在模型中，避免 Phase II 突然加入随机3D通道；但 geometry 任务方向和权重属于下述待确定项，不再沿用初版 `lambda_3d=500` latent MSE；
2. **Phase II：representation and cross-modal enhancement**
   - 从同一 tokenizer/model schema 的 Phase-I checkpoint 继续；
   - 加入 molecule--text 双向生成、caption、C4/general-text replay 与3D敏感性质任务；
   - 保留非零 Phase-I motif/anchor denoising replay，避免语义训练破坏化学语法；
   - geometry-present 与 control view 使用配对成员/同一 corruption，最终3D证据来自 B2D/F3D 和 geometry-sensitive downstream，而不是 identity CE。

后来使用的 **S-stage state imputation -> G-stage grammar bridge** 只保留为机制诊断/adapter warm-up 候选，不再称为正式 Phase I/II。它的顺序与初版“两阶段”语义不同，也不能取代主线。

当前不冻结 `30/40/15/...` 或 Phase-II 四任务各25%的比例。先以配对 exposure、真实 target count 归一化、梯度尺度和遗忘曲线做小样本探索，再冻结正式采样比例。

### 5.3 性质、caption 与表征任务

encoder identity 可以完全 stereo-free。模型若要区分 stereoisomer 或构象，必须使用 E3FP。主要证据来自：

- QM9 等几何敏感性质；
- aligned E3FP vs zero；
- aligned E3FP vs same-2D/matched shuffle；
- F3D vs 同容量、`useChirality=False` 的 B2D；
- motif-only、anchor-only、motif+anchor 三个注入位置消融。

### 5.4 精确立体分子生成

若任务要求输出指定 stereoisomer，decoder 必须拥有显式恢复接口。可选路径是：

1. encoder identity 去立体，但 decoder target 保留 stereo token；或
2. decoder 生成 stereo-free motif，再由独立 atom/bond stereo head 写回。

不能一边从所有输出表示删除 stereo，一边声称可无损生成完整 stereoisomer。该任务与一般性质/文本任务分开报告。

### 5.5 当前 E3FP 使用策略与目标方案的差异

截至本裁决前，正式 V3 代码执行的是：

1. 每个 model atom 读取四级 folded E3FP ID（4096 state domain），但 adapter 只消费 L1/L2；L0 被视为主要重复二维身份，L3 暂作诊断；
2. L1/L2 共用 state embedding，并加入 level embedding、core/attachment role 和 shell-valid 标记，经小型 MLP 得到 atom memory；
3. GraphPorts motif identity span 形成 query，只对该 motif owned atoms 做 constrained attention，生成 motif carrier geometry；
4. 每个 GraphPorts endpoint token 读取其精确 attachment atom memory；
5. carrier 与 endpoint 都以固定 `geometry_fraction`（V3正式配置为 0.5）混入 stock T5 input embedding；
6. 历史 S-stage 每个 eligible motif 至多遮一个 atom row，预测 L1/L2 categorical IDs；后续 grammar stage 联合训练 T5 与 adapter；B2D 使用同接口 Morgan state，另有 aligned/zero/shuffle 与 carrier/endpoint component diagnostics。

这套代码证明了 atom row、motif ownership、endpoint address 和固定混合的数据/模型接口，但模型 token 仍是 GraphPorts，不能直接冒充本文的新 anchor phrase。目标实现需要：

- 把 GraphPorts identity span 换成 stereo-free local motif phrase；
- 把 GraphPorts endpoint address 平移到 anchor occurrence -> attachment atom；
- 保留 atom memory、motif owned-attention 与 carrier/anchor component ablation；
- 按 5.2 的新阶段重新裁决 state loss，不继承旧 checkpoint 的科学结论。

### 5.6 待确定问题：L0/L3 是否进入正式 atom memory

当前 V3 只消费 L1/L2，但现有证据不能把该选择冻结为最终方案：

- **L0** 未在 G1 中做过输入消融；它包含逐原子的元素、邻居、氢、质量/同位素、形式电荷和 ring 等初始 invariants。opaque pure-motif macro 不显式暴露内部逐原子组合，L0 可能是必要的 atom identity/address 补充；
- 排除显式 L0 也不能使 L1/L2 identity-free，因为 E3FP 高层 identifier 递归使用前一层 identifier，二维身份已进入 L1--L3；
- **L3** 的正式阴性证据只否定其作为4096-way exact categorical target：G1 中 L3 NLL/accuracy 未超过先验且上下文稀疏。该结果不等价于“L3作为输入无价值”；原 G1 文档也只裁决保留 L3 输入/诊断、取消 L3 exact-ID target；
- 3D-MolT5 的直接先例是平均每个 atom 的全部可用 E3FP shell components；这支持“全层输入”作为必要 baseline，但不能证明固定平均对 motif 最优。

因此暂不把“恢复L0/L3”或“继续只用L1/L2”写成既定结论。Phase-I实现前必须完成同初始化、同参数预算的最小消融：

| 候选 | atom-memory 输入 | 作用 |
|---|---|---|
| `L12` | L1+L2 | 复现当前V3基线 |
| `L012` | L0+L1+L2 | 检验显式atom identity补充 |
| `L0123-mean` | 全部可用层固定平均 | 贴近3D-MolT5参考 |
| `L0 + shell-attn(L1:L3)` | L0独立分支、L1--L3带availability mask学习聚合 | 当前最有解释力的候选 |

所有候选均不得把 L0 增益直接称为3D增益；必须报告 `motif-only -> +L0 -> +higher shells` 的增量，并与同接口 B2D 对照。L3可以作为输入，但首版不作为 exact-ID预测目标。该问题在 local motif phrase 数据物化后的128/PF1短程实验中裁决。

### 5.7 2026-08-11 修订：V4 是研究基线，不是正式 atom encoder 冻结

上述消融已完成并产生 V4 `l0_l123_mean` 正向结果，但后续逐行复核确认，3D-MolT5 对
逐原子缺失 shell 已采用明确的 `-1 -> fixed zero embedding -> fixed four-slot mean` 语义。
因此正式候选不再默认继承 V4 的 level embedding、presence bits、attachment-role embedding
和两层 atom MLP。

下一候选采用共享 E3FP 表和固定四槽均值；`atom_is_attachment` 只作 anchor endpoint 路由
与验证。V4继续作为已完成、参数更丰富的历史最佳基线。项目创新集中在 stereo-free motif
phrase、ordered anchor topology、atom-to-motif carrier 与 attachment-specific endpoint，
而不是把自定义 atom encoder 本身包装为贡献。完整修订见文档78。

## 6. 现有数据是否需要重新准备

### 6.1 不需要重算的资产

以下昂贵资产继续使用：

| 资产 | 裁决 |
|---|---|
| PCQM/final-v4 member 与 connectivity split | 原样保留 |
| 原始 SDF 坐标与显式氢投影政策 | 原样保留 |
| production geometry release | 原样保留 |
| atom-row、source-row 映射 | 原样保留 |
| motif partition 与 motif atom groups | 原样保留 |
| cross-motif bonds 与 slot atom sidecar | 原样保留 |
| inherited E3FP `[atom,4]` | 原样保留，不重新计算 |
| SELFIES 2.2.0 atom baseline | 原样保留 |

因此不是重新跑 3.38M 分子的 E3FP，也不是重新选择训练成员。

### 6.2 必须重新派生的资产

现有 GraphPorts/isomeric paired release 与 union tokenizer 不能直接用于新模型，需要派生：

- stereo-free anchored motif surface；
- pure-motif macro census；
- chemical lexer/BPE fallback vocabulary；
- anchor occurrence -> attachment atom row；
- pure motif carrier -> core atom rows；
- 新的 token IDs、offsets、mask roles 和 tensor cache；
- 与新 tokenizer 长度匹配的 union-init checkpoint。

旧 GraphPorts checkpoint 不能作为新表示的效果基线继续训练；T5 公共 base checkpoint仍可作为相同初始化源。

### 6.3 PF1/PF10 与 full-scale 的不同成本

- **PF1/PF10**：现有 paired records 已含 atom SELFIES、motif groups、cross bonds、atom mapping 与 E3FP；可以离线派生新表面，通常不需要重新扫描 SDF，也不需要 RDKit/E3FP 重算。
- **最终 full-scale**：geometry release 为避免泄漏没有保存 raw SMILES，只保存拓扑摘要和数组。若尚无完整 paired molecular surface，需要对原 SDF 做一次顺序流式读取以恢复分子图并生成 anchored motif；仍复用已有 coordinates/E3FP，不做几何重算。

新产物是**训练表示/缓存重物化**，不是**化学与几何数据重建**。

## 7. 历史实验如何重新解释

此前 GraphPorts 模型使用 isomeric motif identity，identity 与 E3FP 同时携带立体信息，违背本文的“stereo-free identity + stereo-aware state”假设。因此：

- GraphPorts v1/v2 gate 仍有效地说明该序列语法的效率/可学习性问题；
- PF1、T3MI、G2、V2/V3/V4 仍有效地淘汰各自具体融合器或目标；
- 它们不能否定初版身份--状态分工，因为原假设没有被干净测试；
- 不能把历史最佳 NLL 迁移成新架构的性能声明。

这些实验的保留价值是负面排除、运行协议和数据映射基础设施；不是最终模型的正面效果证据。

## 8. 创新性裁决

### 8.1 不能主张的内容

以下组成均有明确先例，不能单独声称创新：

- fragment/motif molecular language：CAMT5、Group SELFIES、fragSMILES；
- anchor/dummy connection：CAMT5 与多种 fragment codec；
- E3FP 表示构象：E3FP；
- E3FP 注入 T5：3D-MolT5；
- atom--fragment incidence 或 fragment-level pooling：FineMolTex、FACET 等相邻工作；
- BPE/Unigram 解决开放词表：通用语言模型和分子 subword 工作。

### 8.2 有条件成立的组合创新

目前最可信的创新点不是某个单独模块，而是以下接口组合：

1. **stereo-factorized molecular language**：生成式 T5 的 motif identity 主动去立体，而 stereo-aware E3FP 独立承载给定构象状态；
2. **anchor-as-3D-endpoint**：同一个 ordered anchor occurrence 同时是拓扑配对符和 attachment-atom 3D carrier，不增加图语言 token；
3. **open-vocabulary 3D motif**：高频 pure motif 宏与可逆 chemical subword fallback 共用统一 motif/anchor 几何对齐；
4. **可证伪证据链**：以 stereo-free B2D、aligned/zero/shuffle E3FP 和 geometry-sensitive downstream 区分“额外容量”“二维身份”与“对应三维状态”。

更稳妥的论文表述是：

> We factorize a motif language model into stereo-free motif identity and stereo-aware atom-centered E3FP state. Ordered anchor occurrences serve jointly as cross-motif connection identifiers and local 3D endpoint carriers, while an open-vocabulary motif tokenizer preserves long-tail coverage without exposing an auxiliary graph serialization to T5.

该表述的创新强度目前是**方法组合与接口设计层面的中等创新**。只有当 F3D 在严格对照下稳定优于 B2D，并在 QM9/其他3D敏感任务上形成可重复增益，才能上升为有实证支持的模型贡献。

## 9. 逻辑严密性的必要证明

正式 GPU 前只需要完成下列最小门，不再堆叠无关验证：

1. **零 `<unk>`**：train/dev/test 全部 pure motif 可组合编码；
2. **逐字符串可逆**：anchors + pure motif round-trip 100%；
3. **立体泄漏为零**：identity token 中不存在 `@`、`@@`、`/`、`\` 或其等价 stereo label；
4. **B2D 无手性**：Morgan/ECFP 对照必须 `useChirality=False`，且与 F3D 共享容量和注入位置；
5. **E3FP保留立体差异**：预注册 stereoisomer pairs 的 E3FP 必须可区分，并报告 folded collision；
6. **endpoint绑定正确**：每个 anchor ID 恰有两个 occurrence，每个 occurrence 唯一映射 attachment atom；
7. **anchor语法适用域**：全部跨 motif 键均为 `SINGLE/STEREONONE`；anchor ID 在每个分子内稠密、确定、有界，不产生数据集级开放词表；
8. **输出声明一致**：stereo-free输出不宣称 exact stereoisomer reconstruction；
9. **因果消融**：no-geometry、motif-only、anchor-only、both；F3D 另做 aligned/zero/matched-shuffle；
10. **外部效度**：QM9 或同等级真实几何任务提供最终3D证据，identity CE只作训练指标。

通过这些门之前，不启动正式全量预训练。

## 10. 依次解决问题的执行计划

本计划采用单线关键路径。后一步只消费前一步的冻结产物；失败时只回退到最近一个未冻结决策，不重算 E3FP、不重做已有 geometry release，也不同时改表示、词表、模型和训练任务。前四阶段均以 CPU 为主；在数据与接口冻结前不租 GPU。

### Stage 0：冻结比较口径与资产边界

**输入：** final-v4/PF1 membership、现有 motif partition、slot/anchor sidecar、cross-motif bonds、atom mapping、E3FP `[atom,4]`、已登记的下游 split。

**动作：**

1. 建立一份实验登记表，固定 record 顺序、train/dev/test 边界、随机种子、base T5 snapshot 和允许使用的下游 train 集；
2. 明确哪些任务属于 specialist、frozen-transfer、zero-shot，防止同一词表政策被用于相互矛盾的声明；
3. 把 GraphPorts 定位为离线 reconstruction oracle，不再作为候选模型表面。

**产物：** `surface/vocab experiment contract`。本阶段不训练模型。

**通过门：** 所有后续候选共享同一成员、split、原子行和 E3FP；除待比较因素外不存在数据差异。

### Stage 1：实现并验证 stereo-free anchored surface

**范围：** 先做 128 条定向 fixture，再做 PF1 33,600 条；不直接启动 PF10/full-scale。

**动作：**

1. 从现有 motif/slot sidecar 派生 `pure motif + ordered anchor occurrences + slots`；
2. 同时生成显式 phrase 边界版和 sidecar 隐式成组版，但共用同一 canonical logical representation；
3. 保存 `anchor occurrence -> attachment atom row`、motif span、atom ownership 和 reconstruction sidecar；
4. 用 GraphPorts/RDKit 只作离线 oracle，比较恢复后的 atom/bond/topology，不把其 token 暴露给 T5。

**产物：** versioned anchored-surface codec、PF1 派生记录、reject taxonomy、长度与拓扑报告。

**硬门：**

- 33,600/33,600 表面可编码；
- anchored surface 逐字符串 fixed-point round-trip 100%；
- 每个 anchor ID 在分子内恰有两个 occurrence，且各自唯一绑定 attachment atom；
- 恢复拓扑与源记录一致；
- pure motif identity 中无 `@`、`@@`、`/`、`\\` 或其他立体标签；
- 任一失败都先分类，不以 `<unk>`、GraphPorts surface 或删除样本静默回退。

### Stage 2：裁决 chemical lexer 与 fallback

**候选：** `T5-SP`、`CHEM-LEX`、`CHEM-LEX+BPE/Unigram`。whole-motif macro 暂时关闭，避免把 lexer 与宏覆盖率混成一个变量。

**动作：** 在 Stage 1 的同一 pure-motif 集合上比较 exact decode、normalization drift、`<unk>`、token 边界、序列长度和吞吐；chemical lexer 必须单独证明 bracket atom、同位素、形式电荷、键、分支、环号和 slot 可逆。

**产物：** 三候选逐记录 tokenization report 与冻结 lexer contract。

**选择规则：**

1. 可靠性优先：零 normalization drift、零 `<unk>`、逐字符串可逆；
2. 在可靠候选中再比较 mean/P95/P99/max 长度、超过 512 的比例和吞吐；
3. 若 T5-SP 任一可靠性门失败，直接淘汰为权威 fallback；
4. BPE/Unigram 只能压缩 CHEM-LEX token，不得改变其可逆支持域。

### Stage 3：联合裁决 local phrase、macro 与词表来源

本阶段按顺序消除三个变量，不做全因子暴力搜索：

1. **phrase 边界：** macro 自定界；仅 fallback 使用单个 `<fallback_end>` 后缀。sidecar 隐式成组只作 encoder 长度下界，不作为生成候选；
2. **macro budget：** 在已选 fallback 上逐级增加 train-only 高频 whole-motif macro，绘制词表大小--fallback 长度--softmax 成本 Pareto 曲线；
3. **词表来源：** 比较 `V-pretrain` 与预训练前冻结的 `V-pretrain+registered-downstream-train`。validation/test 永不参与；ChEBI-20 train 若参与，就把该模型明确登记为 task-aware specialist，而不是 zero-shot/frozen-transfer。

**产物：** 唯一 tokenizer snapshot、macro registry、词表 provenance、各 split coverage/长度报告。

**硬门：** tokenizer 在 Phase I 第一个 optimizer update 前冻结；Phase II 和下游不再 `add_tokens/resize`；所有新增 token 必须在 Phase I/II 获得真实训练 exposure，不能只注册冷启动行。

### Stage 4：把 V3 几何接口迁移到 motif/anchor 表面

**动作：**

1. 用 pure-motif phrase 替换 GraphPorts identity span；
2. 用 anchor occurrence 替换 endpoint token address，并读取其 attachment atom memory；
3. 保留 atom memory、motif owned-attention 和 motif/anchor 独立 gate；
4. 构建 `L12`、`L012`、`L0123-mean`、`L0 + shell-attn(L1:L3)` 四个输入候选；
5. 保持 B0/B2D/F3D 的容量、注入位置和训练预算配对。

**产物：** anchored 3D-motif wrapper、collator、共享 tensor cache 和 CPU contract tests。

**硬门：** no-geometry 时数值等价于 raw motif T5；motif-only、anchor-only、both 可独立开关；anchor 几何严格来自对应 attachment atom；L0 增益只称 atom-identity 补充，F3D 增益必须由 higher-shell/对照实验单独证明。

### Stage 5：冻结 Phase I 的任务，而不是先冻结比例

**Phase I 主目标：** motif phrase denoising、anchor/slot 结构恢复和必要的自然语言 replay，先学会新的分子语言。geometry adapter 从初始化即存在，但不再使用 `geometry -> identity` 或 exact folded-ID classification 作为主要 3D 证据。

**待比较的最小几何辅助：**

- 仅把 aligned state 作为输入，继续标准 denoising；
- aligned 对 zero/matched-shuffle 的配对一致性或检索辅助；
- L1 为主、L2/L3 低权重的 shell 辅助诊断；
- geometry modality dropout，避免模型永远走二维捷径。

**实验顺序：** 128 条接口 smoke -> PF1 单 seed failure screen -> 仅保留候选做多 seed/PF10。先以真实 target token 数和梯度尺度决定 sampling weight，再冻结任务比例。

**通过门：** syntax/anchor 指标可学；加入几何不破坏 B0 语法；F3D 对 zero 或 same-2D matched-shuffle 呈稳定敏感，并且不能被等容量 B2D 解释。identity CE 本身不作为 3D 成功门。

### Stage 6：冻结 Phase II 与阶段连续性

**动作：** 从同一 Phase-I checkpoint 启动 Phase II，加入 molecule--text 双向生成、caption、FineMolTex 式 motif--text 细粒度对齐、C4/general-text replay 和 3D-sensitive property objective；保留非零 Phase-I denoising replay。

**产物：** 完整 checkpoint/resume、optimizer/scheduler/RNG、数据游标和每任务 exposure manifest。

**通过门：** Phase II 后 anchor reconstruction、motif denoising和自然语言能力没有明显灾难性遗忘；相同 checkpoint 恢复后 batch/mask/参数轨迹可续接；任务权重按真实 target/exposure 报告，不只写名义概率。

### Stage 7：用真实下游任务裁决 3D 声明

**优先顺序：**

1. QM9 或其预注册子集：首个真正的 3D-sensitive 外部效度门；
2. PubChemQC 属性与 PubChem 3D caption：扩大 3D 与生成能力证据；
3. ChEBI-20、PubChem descriptive/caption：分子--文本能力；
4. MoleculeNet 四任务：一般 2D transfer；
5. zero-shot retrieval：跨模态表征；
6. USPTO-50K：可选，不阻塞主线。

**因果矩阵：** 至少比较 B0、B2D、F3D-aligned、F3D-zero、F3D-same-2D-shuffle，并在 F3D 中报告 motif-only、anchor-only、both。先用单 seed 筛除明显失败；只有最终候选才做足够重复与置信区间。

**3D 通过门：** F3D 在至少一个预注册 3D-sensitive 指标上优于同容量 B2D，并对 paired 3D state 的 zero/shuffle 扰动产生方向一致的性能退化；否则只保留为一般 atom-context 模块，不宣称 3D-motif 贡献。

### Stage 7.5：正式全量前的 10% 阶段调度等预算裁决

在进入 full-scale 预训练前，只做一次 10% 数据规模的等预算比较，用来回答“Phase I -> Phase II 的课程顺序是否优于从第一个 update 起联合训练”，不重复此前的表示、词表或几何接口搜索。

| 条件 | 训练日程 | 其余合同 |
|---|---|---|
| A：staged | 先执行 Phase I，再从同一 checkpoint 继续 Phase II；Phase II 保留预注册比例的 Phase-I replay | 与 B 完全相同 |
| B：joint-from-start | 从第一个 update 起联合采样 Phase-I 与 Phase-II 全部任务 | 与 A 完全相同 |

两组必须固定相同的：

- 总训练 token 数与每类任务的累计 target-token exposure；
- 初始化 checkpoint、tokenizer、macro registry、model/fusion config；
- 数据成员、split、成员顺序、corruption seed 与允许使用的数据来源；
- optimizer、全局学习率积分、precision、effective batch、checkpoint 和评估节点；
- 总计算预算，不允许因某组收敛较慢而单独延长。

比较指标只使用已经冻结的门：motif/anchor 恢复、Phase-II molecule--text 指标、ChEBI-20 text2mol、QM9/其他 3D-sensitive 指标，以及 Phase-II 后的结构能力遗忘。另报告真实 wall time 和吞吐，但 worker 数等纯性能参数不作为科学差异。

**预注册决策规则：** 若 A 在结构保持与至少一个 Phase-II/3D 主指标上形成方向一致的优势，且不明显损害其他主指标，则正式全量采用 A；若两者接近或优势不稳定，则采用实现更简单、也更接近 3D-MolT5 joint pre-training 的 B。该 10% 比较是训练日程选择门，不作为最终论文性能结论，也不触发对既有小规模实验的重跑。

### Stage 8：扩大规模与正式训练

只有 Stage 1--7 的对应门通过，并完成 Stage 7.5 的 10% 等预算阶段调度裁决后，才执行 full-scale 正式预训练。此时再使用训练专用 mmap/flat-tensor cache、multi-worker DataLoader、pin-memory/prefetch 和多 GPU；权威 canonical release 只作审计源，不进入 GPU 热路径。

正式训练前冻结：surface/schema hash、tokenizer snapshot、macro registry、E3FP level policy、model/fusion config、Phase-I/II task mixer、split、优化协议、checkpoint/resume 和下游评估合同。

### 10.1 资源规划与停止条件

| 阶段 | 主要资源 | 是否需要租 GPU | 停止条件 |
|---|---|---:|---|
| 0--3 | 本机或 CPU 实例 | 否 | surface/lexer/词表任一可靠性硬门失败 |
| 4 | CPU 单测与小 tensor smoke | 否 | 接口不能保持 B0 等价或 anchor 绑定错误 |
| 5 | 先 CPU，后单卡 4090 | 仅短程 | 语法不可学，或 F3D 完全被 B2D 解释 |
| 6 | 单卡；候选冻结后可双卡并行 | 是 | 阶段切换造成明显遗忘或不可恢复 |
| 7 | 单卡筛查；最终候选再多卡/多 seed | 是 | 真实3D任务无 paired-state 敏感性 |
| 7.5 | 10% 数据；按可用 GPU 并行 A/B | 是 | 等预算合同不闭合，或两组数据/任务 exposure 不一致 |
| 8 | 按规模租多卡 | 是 | 仅在全部前置合同冻结后启动 |

当前已完成 Stage 1 与 Stage 2 的 PF1 CPU 门，正在执行 **Stage 3**。在 Stage 1--4 完成前，不再运行 V2/V3 GraphPorts 训练，也不因 GPU 空闲提前启动新的正式实验。

## 11. 相邻工作与边界

- 3D-MolT5, ICLR 2025: https://proceedings.iclr.cc/paper_files/paper/2025/file/3d4dc72d715bd6415d356293079adf3d-Paper-Conference.pdf
- CAMT5, Findings EMNLP 2025: https://aclanthology.org/2025.findings-emnlp.1221/
- E3FP, Journal of Medicinal Chemistry 2017: https://doi.org/10.1021/acs.jmedchem.7b00696
- Group SELFIES, Digital Discovery 2023: https://doi.org/10.1039/D3DD00012E
- fragSMILES: https://pmc.ncbi.nlm.nih.gov/articles/PMC11779804/
- FineMolTex code artifact: https://doi.org/10.5281/zenodo.15501037

这些工作支持各个组成模块的合理性，但没有直接证明本文的组合有效；最终结论必须由上述对照实验建立。

## 12. CPU 执行记录（2026-08-10）

### 12.1 Stage 1：anchored surface

- PF1 33,600/33,600 派生成功，reject 为 0；没有读取 SDF、没有重算 E3FP，GraphPorts 未进入 model-facing surface；
- 共得到 7,262 类 pure motif、241,799 个 motif occurrence；每个分子 anchor occurrence 为 0--30；
- pure motif 中 `@`、`/`、`\` 泄漏为 0，跨 motif 键全部为 SINGLE；source anchor 配对先按两个 motif 端点重建，再确定性重编号为 molecule-local dense edge ID；
- sidecar 隐式边界长度 `p50/p95/p99/max = 19/34/40/46`；
- 每 motif 一个显式前缀边界长度 `26/46/54/62`。旧双边界候选为 `33/58/68/78`，已淘汰。

旧统计先证明“每 motif 单前缀”明显短于双边界；随后在真实 macro+chemical fallback 展开后，进一步收窄为“macro 自定界、仅 fallback 单后缀”。完全隐式版本没有 standalone generative decode，故不再进入 NLL 候选；双边界与全 motif 前缀均淘汰。

### 12.2 Stage 2：lexer 与无损底线

在相同 33,600 条、7,262 类 pure motif 上：

| 候选 | `<unk>` | exact decode drift | member length p50/p95/p99/max |
|---|---:|---:|---:|
| T5-v1.1 SentencePiece | 0 | 0 | 34/45/49/68 |
| bounded chemical lexer | 0 | 0 | 36/46/50/68 |
| FineMolTex/MolBART 式 atom-wise regex | 0 | 0 | 38/49/53/66 |
| UTF-8 byte floor | 0 | 0 | 38/50/54/74 |

T5-SP 在当前 PF1 上可逆且略短，但这只是经验覆盖；bounded chemical lexer 仅使用 29 个实际词法单元，并由有限元素/数字/键/分支/环/slot 语法提供形式上的开放词表底线。冻结候选因此采用“whole-motif macro 压缩 + bounded chemical lexer fallback”；T5-SP 保留为长度/性能比较，不作为唯一可靠性证明。当前长度已经很低，不先引入额外 BPE/Unigram 变量。

### 12.3 Stage 3 初步：ChEBI-20 train 是否进入词表来源

只读取 ChEBI-20 **train** 的 26,407 条，validation/test 未参与。全部记录可由同一 molecule-native linearizer 投影，得到 5,774 类 pure motif、492,227 occurrences，reject 为 0。它与 PF1 的类型交集仅 824，但 PF1 已有类型覆盖 ChEBI occurrence 的 96.23%，说明差异主要是长尾而不是概率质量主体。

以 512 个宏为例：

- PF1-train-only ranking：PF1 train mean identity tokens/motif = 1.63，ChEBI train = 1.88；
- PF1 与 registered ChEBI-train 等语料质量平衡 ranking：PF1 = 1.67，ChEBI = 1.56，其中只新增 59 个 ChEBI-only 宏；
- 512 个 768 维宏 embedding 约 39 万参数，不构成主要容量负担。

因此保留两个科学声明不同的候选：general checkpoint 使用 pretrain-only registry；若 ChEBI train 在 Phase I/II 获得真实 exposure，则可在第一次 optimizer update 前一次性冻结 balanced specialist registry。后者必须明确标注 task-aware specialist，不能称为 ChEBI zero-shot；任一方案都禁止下游阶段再 `add_tokens/resize`。

对 ChEBI-20 train 重新执行同一冻结 linearizer 并显式复核每个 legacy anchor 恰出现两次后，26,407/26,407 仍通过，实际最大 molecule-local anchor count 为 275（dense ID 上界 274）。因此不采用预留 512 个 anchor token 的保守方案；当前已登记任务只注册 `<0*>`--`<274*>`，后续新增任务若超出该域，必须在 Phase I 首次训练前重新冻结统一 tokenizer，而不能在下游微调时扩词。

基于 `base_vocab_size=32100`、512 个候选 macro、275 个 anchor token 和 153 个 opaque chemical-lexer token，已生成两种 macro 来源下的正式 hybrid 候选及 encoder-only 长度对照：

| macro 来源 | phrase 边界 | 新增普通 token | 候选最终词表 |
|---|---:|---:|---:|
| PF1 train only | implicit sidecar（仅长度下界） | 940 | 33,040 |
| PF1 train only | fallback-only single suffix | 941 | 33,041 |
| PF1 + registered ChEBI train balanced | implicit sidecar（仅长度下界） | 940 | 33,040 |
| PF1 + registered ChEBI train balanced | fallback-only single suffix | 941 | 33,041 |

chemical token 一律映射为 `<MOST:CHEM:...>` opaque ordinary token，不把原始 `.`、括号或元素字符串注册成 Hugging Face AddedToken；这避免改变自然语言的原始 SentencePiece 切分。PF1 全量 33,600 条独立回放显示：pretrain-only 512 macro 覆盖 95.73% motif occurrences，fallback-only 单后缀长度 `p50/p95/p99/max=25/38/43/56`；balanced specialist 覆盖 95.41%，长度为 `25/39/43/56`；两者均 0 个样本超过 512，且 33,600/33,600 standalone decode exact。单前缀与单后缀长度逐记录完全相等，后缀因保留初版 carrier 位置而优先。相比每 motif 单前缀的 `30/47/54/64`（pretrain-only），hybrid 在不牺牲生成闭包的情况下几乎达到 implicit sidecar 的 `25/37/43/55`。

当前计划只冻结候选 token 顺序、ID 和 provenance hash，尚未 resize 模型或取得 training admission。下一 tokenizer 决策只需比较 macro 来源/预算，不再为边界形式单独消耗 GPU。
