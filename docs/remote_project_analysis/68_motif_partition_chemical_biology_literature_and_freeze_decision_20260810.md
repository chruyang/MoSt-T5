# Motif 切分的化学生物学文献核验与主线冻结裁决

> 日期：2026-08-10  
> 状态：主线方法学裁决  
> 适用范围：MoSt-T5 的 motif 原子分区；不同时裁决 anchor 表面语法、motif 词表、E3FP 聚合器或训练任务比例  
> 核心结论：保留当前 `ring/non-single-bond union` 切分，不再把切分算法性能比较列为正式训练前置实验

## 1. 本文回答的问题

本文只回答以下问题：

1. 当前生产 motif 分区到底是什么；
2. 它是否与 CAMT5 官方实现完全相同；
3. CAMT5 的精确切分得到哪些真实文献支持；
4. 当前变体是否具有药物化学和化学生物学依据；
5. 是否存在一个已经公认、可以直接替代当前规则的“更优”切分；
6. 在 motif 切分不是论文主要研究变量的前提下，主线应冻结什么、可以声称什么、不能声称什么。

本文不把“相邻领域采用 fragment”误写成“该论文直接证明了我们的精确算法”，也不要求为一个非核心变量开展大规模切分基线实验。

## 2. 当前生产切分的精确定义

### 2.1 图论定义

给定分子图 \(G=(V,E)\)，当前实现构造以下 seed atom sets：

1. RDKit `AtomRings()` 返回的每个环原子集合；
2. 每条非 `SINGLE` 键的两个端点集合。

随后：

1. 只要两个 seed sets 共享至少一个原子，就对其做传递合并；
2. 每个合并后的连通集合成为一个 motif；
3. 没有进入任何 seed set 的原子各自成为 singleton motif；
4. motif 按原始原子索引确定性排序。

对应代码为：

- `most_t5_next/r1/adapter/mol_linearizer.py::_canonical_motif_groups`
- 当前实现入口位于该文件约第 218 行。

因此当前规则可正式命名为：

> **ring/non-single-bond union partition**

或在论文中称为：

> **CAMT5-inspired rigid-conjugated local motif partition**

这里的 `rigid-conjugated` 是近似性描述，不是“motif 内不存在任何构象自由度”的严格力学断言。

### 2.2 该规则的直接结果

- 稠合环和螺环会因共享原子而合并；
- 环内所有原子会作为一个结构域保留；
- 相交的双键、三键、芳香键/共轭非单键区域会合并；
- 一般饱和链原子若不属于环或非单键，通常形成 singleton motifs；
- motif 间边界通常位于单键；
- anchor、连接顺序和 E3FP carrier 地址是在分区之后定义的，不应与分区规则混为一谈。

## 3. 与 CAMT5 官方实现的关系

### 3.1 CAMT5 论文中的抽象定义

CAMT5 将以下原子群视为 motif token：

1. 构成环结构的原子；
2. 由非单键连接的原子；
3. 不属于上述两类的原子作为 singleton tokens。

论文将其解释为较为刚性、能体现共振等结构上下文的原子群，并引用 Anslyn 与 Dougherty 的物理有机化学教材作为化学依据。随后以 DFS 线性化 motif tree。

来源：

- Seojin Kim et al., *Training Text-to-Molecule Models with Context-Aware Tokenization*, Findings of EMNLP 2025.  
  <https://aclanthology.org/2025.findings-emnlp.1221/>
- Eric V. Anslyn and Dennis A. Dougherty, *Modern Physical Organic Chemistry*, 2006.

### 3.2 CAMT5 官方代码中的实际判定

CAMT5 官方源码执行的是：

1. 枚举所有化学键；
2. 如果一条键的两个端点原子都 `IsInRing()`，不切断；
3. 如果键不是 `SINGLE`，不切断；
4. 切断剩余单键；
5. 以 `FragmentOnBonds` 后的 connected components 作为 motifs。

官方镜像位置：

- `reference_repos/CAMT5_official_src_5875a0a/model/representation.py`，约第 286–309 行。

### 3.3 当前实现并非 CAMT5 的代码级复制

二者在大多数环、稠环和共轭链上结果相近，但并不数学等价。

最清楚的反例是联芳基单键：

```text
aryl ring — rotatable single bond — aryl ring
```

- CAMT5 官方代码：单键两端原子都在环中，因此保留该键，两个环可能进入同一个 motif；
- 当前生产规则：两个环集合不共享原子，中间键又是 `SINGLE`，因此得到两个 motifs。

所以不能写：

> We exactly adopt the CAMT5 motif partition.

可以写：

> We adopt a CAMT5-inspired partition and make rotatable inter-ring single bonds explicit motif boundaries, yielding transitive unions of ring and non-single-bond atom sets.

该差异并非无依据的任意偏离。对于“motif 承载局部 3D 状态”的目标，把可旋转联芳键留在 motif 间边界，比把整个联芳体系假设成单一刚性单元更容易解释。不过轴手性联芳键等情况也说明它仍然只是近似规则。

## 4. 真实文献证据分级

### 4.1 A 级：对当前方法类别的直接模型证据

#### CAMT5

CAMT5 直接证明了：

- 环/非单键结构上下文可以作为 T5 子结构 tokenization；
- motif-level representation 可用于 text-to-molecule；
- 其具体 tokenization 在论文的 ChEBI-20 设置中优于所比较的 BRICS 和 t-SMILES 表面；
- DFS 在其消融中优于 BFS。

CAMT5 不能直接证明：

- 当前生产变体与官方规则完全等价；
- 该分区对 3D 表征最优；
- 每个 motif 都是标准功能团或生物活性基团。

来源：<https://aclanthology.org/2025.findings-emnlp.1221/>

### 4.2 B 级：药物化学与化学生物学的机制支撑

#### Bemis–Murcko molecular frameworks

Bemis 与 Murcko 用 ring、linker、framework 和 side-chain atoms 组织已知药物结构。这直接支持“环系与连接区是药物化学中有意义的结构层次”，但它提供的是 scaffold 分析框架，不是我们当前的细粒度分区算法。

- Guy W. Bemis and Mark A. Murcko, *The Properties of Known Drugs. 1. Molecular Frameworks*, J. Med. Chem. 1996.  
  <https://doi.org/10.1021/jm9602928>

#### 分子柔性与可旋转键

Veber 等基于超过 1100 个候选药物发现，以可旋转键数量度量的分子柔性与口服生物利用度相关。这支持把普通可旋转单键视作局部结构域之间的自然边界之一，但它没有提出 motif tokenizer。

- Daniel F. Veber et al., *Molecular Properties That Influence the Oral Bioavailability of Drug Candidates*, J. Med. Chem. 2002.  
  <https://doi.org/10.1021/jm020017n>

#### 药物中的环系

针对上市药物的环系分析显示，ring systems 和 molecular frameworks 是药物空间、治疗领域和 target classes 的稳定分析单位。关于二维/三维环片段的研究还表明，环形 fragment 的形状多样性与不同蛋白靶标类别相关。

- Richard D. Taylor et al., *Rings in Drugs*, J. Med. Chem. 2014.  
  <https://doi.org/10.1021/jm4017625>
- Matteo Aldeghi, Shipra Malhotra, David L. Selwood and Ah Wing Edith Chan, *Two- and Three-dimensional Rings in Drugs*, Chemical Biology & Drug Design 2014.  
  <https://doi.org/10.1111/cbdd.12260>

这些工作支持“环作为药化和化学生物分析单位”，但不能证明“大环一定刚性”或“任何整个环都应对应一个单独 token”。

#### BRICS

BRICS 使用 retrosynthetically interesting chemical substructures 和 medicinal-chemistry concepts 构造 drug-like fragment spaces。它支持“沿有化学意义的连接形成 fragment”这一更宽泛原则。

- Jörg Degen et al., *On the Art of Compiling and Using 'Drug-Like' Chemical Fragment Spaces*, ChemMedChem 2008.  
  <https://doi.org/10.1002/cmdc.200800178>

BRICS 主要优化逆合成和片段库质量，不直接优化 T5 语法、局部刚性或 motif–E3FP 对齐。

#### RECAP

RECAP 根据化学知识对生物活性分子做电子切分，以寻找富含 biologically recognized elements、privileged motifs 和 structures 的 building blocks。它为 fragment 的生物活性语义提供了直接药化背景。

- Xiao Qing Lewell et al., *RECAP—Retrosynthetic Combinatorial Analysis Procedure*, J. Chem. Inf. Comput. Sci. 1998.  
  <https://doi.org/10.1021/ci970429i>

RECAP 同样不是针对生成式 T5 或给定构象 3D 状态设计的分区。

#### Ertl functional groups

Ertl 提出了规则驱动的功能团识别算法，并在 ChEMBL 生物活性化合物上提取功能团。该工作直接支持“功能团是连接结构、反应性、毒性、药化性质与化学命名的语义单位”。

- Peter Ertl, *An Algorithm to Identify Functional Groups in Organic Molecules*, J. Cheminformatics 2017.  
  <https://doi.org/10.1186/s13321-017-0225-z>

但 functional groups 可以嵌套、相交或与 ring/scaffold 层次不一致，因此它更适合作为重叠语义注释或辅助目标，而不是未经裁决就替换唯一无损分区。

### 4.3 B/C 级：分子机器学习中的 fragment/motif 先例

#### JT-VAE 与 Hierarchical Graph Generation

JT-VAE 以有效化学子结构组成 junction tree，Hierarchical Graph Generation 则显示更大的结构 motifs 可以改善大分子生成。这些工作支持“原子并非唯一合适的生成单位”和“attachment resolution 应被显式建模”。

- Wengong Jin et al., *Junction Tree Variational Autoencoder for Molecular Graph Generation*, ICML 2018.  
  <https://proceedings.mlr.press/v80/jin18a>
- Wengong Jin et al., *Hierarchical Generation of Molecular Graphs Using Structural Motifs*, ICML 2020.  
  <https://proceedings.mlr.press/v119/jin20a>

它们不是 T5 tokenizer，也没有证明当前 ring/non-single union 是唯一分区。

#### t-SMILES

t-SMILES 将 fragment graph 转换为树式分子语言，并系统使用 JTVAE、BRICS、MMPA 和 Scaffold 等不同 fragmentation algorithms。其结果的重要方法学含义是：fragment serialization 可以与多种切分共存，不同切分在不同任务中可能表现不同。

- Juan-Ni Wu et al., *t-SMILES: A Fragment-Based Molecular Representation Framework for De Novo Ligand Design*, Nature Communications 2024.  
  <https://doi.org/10.1038/s41467-024-49388-6>

这反过来说明领域内没有被确定为所有任务通用最优的唯一 fragmentation。

## 5. 文献能够支持什么，不能支持什么

### 5.1 可以支持的主张

当前证据足以支持：

1. 环系、共轭/非单键区域和 linker/rotatable-bond boundaries 是成熟的化学结构抽象；
2. fragment/motif 表示在药物设计、分子生成和分子—文本建模中有直接先例；
3. 把环/共轭区域作为局部结构上下文，比纯原子 token 更接近化学中层级化的结构理解；
4. 对本项目而言，把普通可旋转单键留在 motif 间，可为“局部身份短语 + 局部 3D 状态 + 跨 motif 构象关系”提供清楚接口；
5. 当前规则足以作为固定主线分区，不需要在正式架构研究前开展 BRICS/RECAP/Murcko/Ertl 的完整性能竞赛。

### 5.2 不能支持的主张

现有证据不能支持：

1. 当前精确算法由某篇化学生物学论文直接提出；
2. 当前算法是药物化学唯一、普遍或理论最优的分区；
3. 每个 motif 都是标准功能团、药效团或 privileged substructure；
4. motif 内完全刚性、motif 间一定自由旋转；
5. motif 本身足以确定蛋白结合功能；
6. 当前分区对 3D 下游任务优于所有其他分区。

## 6. 当前规则的化学边界与不构成阻断的例外

### 6.1 酰胺及其他受限单键

某些形式上为 `SINGLE` 的键具有显著部分双键性质，例如酰胺 C–N。当前规则可能把它们分到不同 motifs。这说明规则是“环/非单键结构域近似”，不是完整的旋转势能分类器。

这不要求当前立刻引入复杂 SMARTS 规则，原因是：

- motif 分区的首要职责是确定、无损、可学习；
- E3FP 仍保留真实给定构象下的原子环境；
- anchor/attachment 保留跨 motif 连接；
- 把所有共振例外编码进分区会使规则、词表和复现边界显著复杂化。

### 6.2 宏环和非刚性环

环不等于绝对刚性，大环尤其可能具有显著构象自由度。因此论文应使用 `locally coupled ring/conjugated domain`，而不是把所有 motif 称为 rigid bodies。

### 6.3 联芳轴手性

当前规则会切开普通联芳单键，这通常符合局部构象域分离；但部分联芳键存在受限旋转乃至稳定轴手性。这同样属于近似边界，而不是 codec 错误。

### 6.4 功能团被跨 motif 切开

如果某个功能团的核心关系依赖形式单键，当前规则可能将其拆分。该风险主要影响“motif 必然有自然语言名称”的强主张，不破坏无损重构或 atom-level E3FP 使用。

## 7. 为什么不把药化规则直接替换成主分区

不同规则优化不同目标：

| 方法 | 主要目标 | 不直接替换当前规则的原因 |
|---|---|---|
| CAMT5 官方规则 | text-to-molecule 上下文 tokenization | 与局部3D状态目标并不完全一致；联环单键可能合并 |
| BRICS | 可合成、drug-like fragment space | 偏逆合成；不保证局部刚性、文本语义或短序列 |
| RECAP | 生物活性分子的逆合成 building blocks | 依赖反应知识，目标不是统一生成语言 |
| Bemis–Murcko | scaffold/framework 分析 | 粒度过粗，侧链处理不适合作唯一 token 分区 |
| Ertl functional groups | 化学功能和语义识别 | 功能团可能重叠/嵌套，不天然形成唯一原子分区 |
| Pharmacophore | 蛋白结合所需的空间功能模式 | 与靶标和构象相关，经常重叠，不能作为通用无损 tokenizer 分区 |

不存在一种方法同时被证明最优于：

- 无损生成；
- T5 语法简洁；
- 自然语言对齐；
- 药化功能团；
- 逆合成；
- 局部 3D 状态；
- 蛋白结合药效团。

因此，继续搜索“领域已确定的唯一最佳切分”不会得到可靠答案。

## 8. 主线冻结裁决

### 8.1 冻结内容

正式主线冻结当前：

> 由环原子集合和非单键端点集合的传递并集构成 motifs，未覆盖原子构成 singleton motifs；普通跨域单键由显式 anchor/attachment 接口保存。

不切换到 BRICS、RECAP、Murcko 或 Ertl 分区，也不要求在正式训练前完成切分性能基线矩阵。

### 8.2 冻结理由

1. **直接模型依据**：CAMT5 已验证同类别的 motif-aware T5 tokenization；
2. **物理化学依据**：环、非单键、共轭/共振形成局部耦合结构上下文；
3. **药物化学依据**：ring systems、linkers、rotatable bonds、drug-like fragments 和 functional groups 均是成熟分析层次；
4. **3D 接口一致性**：把可旋转单键显式留在 motif 边界，有利于区分 motif 内局部状态和 motif 间构象关系；
5. **工程性质**：当前规则唯一、确定、可重编号核验并可无损恢复；
6. **研究范围**：本项目创新重点是 motif identity、anchor topology 与 E3FP state 的组织和训练，而不是提出新的 fragmentation algorithm。

### 8.3 不再要求的实验

正式主线不再要求：

- 当前切分 vs CAMT5 exact 的完整训练比较；
- 当前切分 vs BRICS/RECAP/Murcko/Ertl 的下游矩阵；
- 为证明切分“全局最优”而消耗 GPU 预算。

如果未来结果暴露明确的分区归因故障，例如大量关键化学短语被系统切碎、序列长度失控或 3D 聚合无法定位，才重新打开该裁决；不能仅因存在其他 fragmentation 文献而重开。

## 9. 仍然必须保留的最小质量门

这些是实现正确性检查，不是切分性能基线：

1. 每个原子恰好属于一个 motif；
2. motif 之间不重叠且并集覆盖全部模型原子；
3. 原子重编号后按 canonical correspondence 得到相同化学分区；
4. anchor occurrence 能恢复全部跨 motif bonds，包括多 anchor motif；
5. encode–decode 保持正式规定的化学身份与立体边界；
6. motif/anchor 序列长度满足模型上限；
7. atom-to-motif、attachment atom 和 E3FP row 严格同轴。

这些门用于防止实现错误，不能被描述为对 motif 科学优越性的消融。

## 10. 推荐论文表述

### 10.1 中文

> 我们采用受 CAMT5 启发的局部环—共轭 motif 分区。具体而言，重叠环集合与非单键端点集合通过共享原子的传递闭包合并，未被覆盖的原子形成单原子 motif；可旋转的跨 motif 单键由有序锚点显式保存。该设计近似局部耦合的结构域，而非宣称给出通用药效团或药物化学最优分区。其依据来自 CAMT5 的上下文感知分子 tokenization、物理有机化学中的环/共轭耦合，以及药物化学中 ring–linker、rotatable-bond 和 fragment 层级的成熟使用。

### 10.2 英文

> We use a CAMT5-inspired rigid-conjugated local motif partition. Overlapping ring atom sets and non-single-bond endpoint sets are merged by transitive closure, while uncovered atoms form singleton motifs. Rotatable inter-motif single bonds are retained through ordered anchor occurrences. The partition is intended as a deterministic and lossless approximation of locally coupled structural domains, rather than a universal pharmacophore, functional-group, or medicinal-chemistry-optimal decomposition.

### 10.3 创新性边界

切分规则本身不作为主要创新点。可主张的组合创新应落在：

- stereo-free/local motif identity phrase；
- 多 anchor 的确定连接接口；
- anchor occurrence 与 attachment-atom 3D carrier 的统一；
- atom-level E3FP 到 motif-level state 的受限聚合；
- motif/anchor 生成语言与 3D 敏感任务的联合训练证据。

## 11. 最终结论

从真实文献层面可以确定：

1. **没有**一个被化学生物学和药物化学共同确立、适用于所有生成和表征任务的唯一最佳 motif 切分；
2. 当前精确传递并集规则不是某篇药化论文的逐字复现，因此不能声称“药化标准算法”；
3. CAMT5 为其方法类别提供直接 T5 证据；Bemis–Murcko、Veber、药物环系研究、BRICS、RECAP 和 Ertl 分别提供环系、柔性边界、药化片段和功能语义的机制依据；
4. 这些证据足以把当前切分冻结为本项目的主线结构先验；
5. 论文应把它描述为有文献依据的、确定无损的局部环—共轭结构域近似，而不是最优功能团或药效团分区；
6. 后续研究资源应投入 E3FP 层级消费、motif/anchor 注入、训练目标和 3D 敏感下游任务，而不是继续比较非核心 fragmentation baselines。

## 12. 参考文献

1. Kim, S. et al. Training Text-to-Molecule Models with Context-Aware Tokenization. Findings of EMNLP, 2025. <https://aclanthology.org/2025.findings-emnlp.1221/>
2. Anslyn, E. V.; Dougherty, D. A. *Modern Physical Organic Chemistry*. University Science Books, 2006.
3. Bemis, G. W.; Murcko, M. A. The Properties of Known Drugs. 1. Molecular Frameworks. *J. Med. Chem.* 1996. <https://doi.org/10.1021/jm9602928>
4. Veber, D. F. et al. Molecular Properties That Influence the Oral Bioavailability of Drug Candidates. *J. Med. Chem.* 2002. <https://doi.org/10.1021/jm020017n>
5. Degen, J. et al. On the Art of Compiling and Using 'Drug-Like' Chemical Fragment Spaces. *ChemMedChem* 2008. <https://doi.org/10.1002/cmdc.200800178>
6. Lewell, X. Q. et al. RECAP—Retrosynthetic Combinatorial Analysis Procedure. *J. Chem. Inf. Comput. Sci.* 1998. <https://doi.org/10.1021/ci970429i>
7. Ertl, P. An Algorithm to Identify Functional Groups in Organic Molecules. *J. Cheminformatics* 2017. <https://doi.org/10.1186/s13321-017-0225-z>
8. Taylor, R. D.; MacCoss, M.; Lawson, A. D. G. Rings in Drugs. *J. Med. Chem.* 2014. <https://doi.org/10.1021/jm4017625>
9. Aldeghi, M.; Malhotra, S.; Selwood, D. L.; Chan, A. W. E. Two- and Three-dimensional Rings in Drugs. *Chemical Biology & Drug Design* 2014. <https://doi.org/10.1111/cbdd.12260>
10. Jin, W.; Barzilay, R.; Jaakkola, T. Junction Tree Variational Autoencoder for Molecular Graph Generation. ICML 2018. <https://proceedings.mlr.press/v80/jin18a>
11. Jin, W.; Barzilay, R.; Jaakkola, T. Hierarchical Generation of Molecular Graphs Using Structural Motifs. ICML 2020. <https://proceedings.mlr.press/v119/jin20a>
12. Wu, J.-N. et al. t-SMILES: A Fragment-Based Molecular Representation Framework for De Novo Ligand Design. *Nature Communications* 2024. <https://doi.org/10.1038/s41467-024-49388-6>
