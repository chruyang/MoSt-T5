# 09 文献目录、页级索引与缺口记录

> 本文记录实际检查过的本地论文、补充下载结果和仍需补齐的证据。页码均指 PDF 物理页码，不等同于论文印刷页码。

## 1. 本地原有论文及用途

| 文件 | PDF 页数 | 可支撑主题 | 重点页 |
|---|---:|---|---|
| `paper/t-smiles.pdf` | 15 | fragment/motif、多尺度表示、连接拓扑、生成 | 1–4 |
| `paper/MoIT5.pdf` | 39 | MolT5；T5 分子—文本联合预训练、C4+ZINC、caption/text2mol | 1、3–5 |
| `paper/finemoltex.pdf` | 17 | 分子级与 motif 级对齐、重要 motif/word masking、cross-attention | 1–3 |
| `paper/CAMT5.pdf` | 19 | motif-aware tokenization、importance-based pretraining | 1、2、4、8 |
| `paper/biot5.pdf` | 22 | SELFIES/文本统一 T5、多类分子—文本任务 | 1–2 |
| `paper/biot5+.pdf` | 25 | 多任务 instruction tuning、IUPAC、数值 tokenization | 1–4 |
| `paper/bindgpt.pdf` | 9 | 自回归三维坐标 token、旋转增强、RL | 3–4；仅外围参考 |
| `paper/Adaptive receptive field graph neural networks.pdf` | 11 | 自适应多 hop 感受野 | 1–3；仅为多尺度概念类比 |
| `paper/3dMolt5.pdf` | 26 | E3FP 三维 token、原子级 1D/3D 对齐、七类预训练任务 | 2–6 |
| `paper/3D-MoLM.pdf` | 20 | 3D encoder、projector、分阶段 molecule-text alignment | 1–4 |
| `paper/C-FREE.pdf` | 21 | 分子 2D/3D context-to-target latent MSE、EMA target encoder | 3–4 |
| `paper/3DMolFormer_ICLR2025.pdf` | 22 | 分子 3D 双通道序列中的 token CE + 坐标 MSE 直接先例 | 6，式(2) |

## 2. 已下载并验证的补充论文

目录：`paper/supplementary/`

| 文件 | 状态 | 用途 | 重点页/来源 |
|---|---|---|---|
| `T5_2020_JMLR.pdf` | 有效，67 页 | text-to-text、C4、span corruption、15% corruption | 24；JMLR 21(140) |
| `GraphMVP_2022_ICLR.pdf` | 有效，32 页 | 2D/3D SSL、跨视图对齐、连续表示空间重建 | 1–5；ICLR 2022 |
| `MoleculeNet_2018.pdf` | 有效，65 页 | 分子 ML benchmark、split、metric、数据集 | 1–6；Chemical Science 2018 |
| `LoRA_2022_ICLR.pdf` | 有效，26 页 | 冻结预训练权重、低秩下游适配 | 1–2；ICLR 2022 |
| `SimSiam_2021_CVPR.pdf` | 有效，9 页 | stop-gradient 与 collapse 的机制类比 | 1–2；CVPR 2021 |
| `Gated_Multimodal_Units_2017.pdf` | 有效，17 页 | 可学习乘性门控的通用多模态融合 | 1–2；ICLR workshop 2017 |
| `SELFIES_2020.pdf` | 有效，9 页 | 鲁棒分子字符串表示 | NeurIPS workshop/arXiv |
| `Uni-MolPlus_2024.pdf` | 有效，11 页 | 廉价 RDKit 构象与目标高质量构象的差异、3D QC 预测 | 1–3；注意这是 Uni-Mol+，不是原始 Uni-Mol |
| `data2vec_2022_ICML.pdf` | 有效，15 页 | masked student、full-input EMA teacher、target normalization、Smooth L1 latent regression | §3.3–3.4 |
| `DynaBERT_2020_NeurIPS.pdf` | 有效，12 页 | Transformer 内 soft CE + embedding/hidden-state MSE 及按量级设权 | 4，式(3) |

## 3. 下载异常与不可引用文件

| 文件 | 状态 | 处理原则 |
|---|---|---|
| `E3FP_2017_JMedChem.pdf` | 仅 1816 字节，内容是 HTML/站点响应，不是 PDF | 禁止作为本地 PDF 引用；使用 PubMed/PMC 正文与 DOI |
| `SGPT_2022_accidental_download_DO_NOT_CITE.pdf` | 有效 PDF，但内容为 SGPT，不是 Uni-Mol | 明确禁止用于本项目证据；因仓库禁止未授权删除而保留并改名 |

## 4. 权威在线来源

- T5：Raffel et al., 2020, *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*. JMLR：<https://www.jmlr.org/papers/v21/20-074.html>
- E3FP：Axen et al., 2017, *A Simple Representation of Three-Dimensional Molecular Structure*. DOI：<https://doi.org/10.1021/acs.jmedchem.7b00696>；PMC：<https://pmc.ncbi.nlm.nih.gov/articles/PMC6075869/>
- GraphMVP：Liu et al., ICLR 2022：<https://openreview.net/forum?id=xQUe1pOKPam>
- Uni-Mol：Zhou et al., ICLR 2023：<https://openreview.net/forum?id=6K2RM6wVqKu>
- MoleculeNet：Wu et al., 2018：<https://doi.org/10.1039/C7SC02664A>
- LoRA：Hu et al., ICLR 2022：<https://openreview.net/forum?id=nZeVKeeFYf9>
- SimSiam：Chen & He, CVPR 2021：<https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Exploring_Simple_Siamese_Representation_Learning_CVPR_2021_paper.html>
- GMU：Arevalo et al., 2017：<https://arxiv.org/abs/1702.01992>
- SELFIES：Krenn et al.：<https://arxiv.org/abs/1905.13741>
- 3DMolFormer：Hu et al., ICLR 2025：<https://openreview.net/notes/edits/attachment?id=Mer4HrLWeI&name=pdf>
- DynaBERT：Hou et al., NeurIPS 2020：<https://proceedings.nips.cc/paper_files/paper/2020/file/6f5216f8d89b086c18298e043bfe48ed-Paper.pdf>
- Patient Knowledge Distillation：Sun et al., EMNLP 2019：<https://aclanthology.org/D19-1441.pdf>
- data2vec：Baevski et al., ICML 2022：<https://proceedings.mlr.press/v162/baevski22a/baevski22a.pdf>

## 5. 证据缺口

### 必须补实验，不能只补引用

1. 当前 local atom-to-motif attention 的优越性。
2. 当前 gate 公式和空映射处理的实际收益。
3. E3FP level sum 相对 concat/attention 的收益。
4. moving latent target 的稳定性与非塌缩性。
5. `lambda_3d=500`、shell dropout、四任务 25% 等数值。
6. Morgan 相似 motif embedding 复制初始化。
7. 单构象和四层/4096 bit 对目标任务是否足够。

### 可继续补原始文献

1. RDKit ETKDG 构象生成原始论文（Riniker & Landrum, 2015），用于构象生成参数审计。
2. ECFP 原始论文（Rogers & Hahn, 2010），用于 Morgan/Tanimoto 相似度边界。
3. Bemis–Murcko scaffold 原始论文，用于 split 的结构定义。
4. 多任务采样/梯度平衡方法，用于替代固定四任务等比例。
5. molecular conformer ensemble 与构象不确定性文献，用于评估单构象假设。

## 6. 引用使用规则

1. 引用必须精确到“文献支持的命题”，不能用一篇 3D 论文支持任意 3D 实现。
2. 论文采用了相似机制时标为“间接支撑”，不能标为“已证明当前方法”。
3. 精确超参数原则上属于本项目实验结论，除非完全复现并满足同一数据、模型与任务条件。
4. 网页摘要用于定位；正式文档优先引用论文 PDF、期刊页、DOI 或 OpenReview。
5. 本地 PDF 页码必须在上下文中人工复核；关键词命中只是索引，不是自动证据。

## 7. 可复用索引工具

`tools/literature_keyword_index.py` 可只读扫描 `paper/` 下 PDF，并输出关键词出现页。示例：

```powershell
python tools/literature_keyword_index.py paper --terms E3FP motif "span corruption"
```

该工具不能判断支持或反驳关系；最终证据等级仍需人工阅读上下文。
