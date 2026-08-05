# 02 代码库目录与职责

## 1. 版本边界

远端 `/root/autodl-tmp/MoSt-T5` 是实际训练现场，包含大量未提交代码；本地工作区是相关源码副本，但并非所有关键文件都与远端一致。

2026-07-17 哈希核验结果：

- `tokenization/motif_tokenizer.py`：本地与远端 SHA-256 一致。
- `model/modeling.py`、`dataset/dataset.py`、`dataset/dataset2.py`、`train1.py`、`train2.py`：本地与远端哈希不同。

因此：

- tokenizer 稳定性问题可直接在本地源码定位。
- 历史训练流程和超参数以远端脚本、远端 checkpoint 状态为准。
- 本地源码适合继续重构，但修复前应先建立远端关键源码快照或 commit。

## 2. 远端 Git 状态

远端主仓库：

- 14 个已跟踪文件被修改。
- 3 个已跟踪文件被删除。
- 约 220 个未跟踪条目。
- `git diff --stat` 约 15,672 行新增、118,314 行删除。

核心修改涉及：

- 参数定义。
- LMDB 数据集和 collator。
- CAMT5 motif 表示。
- MoSt-T5 模型与 1D/3D 融合。
- motif/text/E3FP tokenizer。
- 训练启动脚本。

远端所有 286 个 Python 文件均通过 AST 语法解析，但“语法正确”不等于训练逻辑或数据对齐正确。

## 3. 主要目录

### 3.1 `model/`

| 文件 | 作用 |
|---|---|
| `configuration.py` | 在 `T5Config` 上增加 E3FP vocabulary、shell 层数和融合类型 |
| `modeling.py` | MoSt-T5 核心模型：多层 E3FP embedding、atom-to-motif 局部注意力、门控融合、3D 重建头 |
| `CAMT5/representation.py` | 把 SMILES 拆成带连接锚点的 motif 序列，并支持解码恢复 |
| `CAMT5/config.py` | CAMT5 representation/tokenizer 配置类型 |

`modeling.py` 是模型思路的中心文件；`representation.py` 决定 motif 序列的化学语义和 atom mapping 的基础。

### 3.2 `tokenization/`

| 文件/目录 | 作用 |
|---|---|
| `text_tokenizer.py` | T5 文本 tokenizer，并注册 `[MMM]:`、`[Caption]:`、`[Text2Mol]:`、`[Denoise]:` |
| `motif_tokenizer.py` | 将 SMILES 线性化为 motif + anchor token，产生 motif token 到原 motif 的映射 |
| `e3fp_tokenizer.py` | 从分子构象生成每原子、每 shell 的 E3FP bit ID |
| `e3fp/` | 内嵌的 E3FP 实现 |
| `3d_tokenization/e3fp/` | 另一份 E3FP 源码副本 |

本地比较显示两份 E3FP 目录内容基本相同。重复 vendoring 增加了导入路径不确定性和维护成本。

### 3.3 `dataset/`

| 文件 | 作用 |
|---|---|
| `dataset.py` | Phase 1 数据读取、importance-weighted 非坍缩 mask、几何掩码 collator |
| `dataset2.py` | Phase 2 四任务数据路由和多任务 collator |
| `view_lmdb.py` | LMDB 结构检查和样本浏览 |
| `check_duplicates.py` | LMDB 重复样本检查 |
| `check_lmdb_id_continuity.py` | key 连续性/缺失分析 |
| `vertify.py` | 样本字段与数据完整性检查 |

两个 Dataset 都采用只读 LMDB，并在 worker 内延迟打开环境，避免将 LMDB handle 直接跨进程传递。

### 3.4 `process/`

主要职责：

- 从 PubChem/PubChemQC 生成 E3FP。
- 生成 atom-to-motif mapping。
- 构建 Phase 2 LMDB。
- 统计文本 IDF 权重。
- 构建/合并 motif vocabulary。
- 运行部分下游任务。

远端 `process/` 下还有复制的 `tokenization/`、脚本副本和大量 notebook checkpoint，说明实验代码曾通过复制目录推进。后续应将可复用逻辑收敛到一个包中。

### 3.5 `vocabs_process/`

用于：

- 统计 PubChem、QM9、ChEBI、USPTO、反应预测数据的 motif coverage。
- 生成 20K/25K 词表。
- 评估扩词表收益。
- 检查 motif 重建覆盖率。

这是判断 vocabulary 规模合理性的实验工具链，但当前 tokenizer 的 ID 不稳定问题会优先于 coverage 优化。

### 3.6 `moleculenet/`

包含 MoleculeNet 数据加载、分子图构建、scaffold/random split 等代码，服务于分类和回归下游评估。

### 3.7 根目录训练/评估脚本

| 文件 | 作用判断 |
|---|---|
| `train1.py` | Phase 1：纯 MMM + 1D/3D 几何对齐预训练 |
| `train2.py` | Phase 2：MMM、caption、text2mol、C4 denoise 四任务训练 |
| `run_train.sh` | Phase 1 多 GPU 超参数入口 |
| `run_train2.sh` | Phase 2 八 GPU 超参数入口 |
| `run.py` | ADMET 回归/分类 encoder probing 或 LoRA 实验 |
| `run_property_prediction.py` | 生成式性质预测 |
| `run_regression_property.py` | encoder + regression head 数值回归 |
| `run_generative_property.py` | 生成式 QM9/性质任务 |
| `run_qm9_discriminative.py` | QM9 判别式性质任务 |
| `run_chebi_text2mol.py` | 文本到分子生成 |
| `run_mol2text_camt5_style.py` | 分子到文本描述 |
| `run_admet_*`、`run_mpnn_*` | ADMET、MPNN 融合和消融实验 |

## 4. 主入口与次级入口

建议把当前文件按可信度和用途分为三层：

### 主线入口

- `train1.py`
- `train2.py`
- `run_train.sh`
- `run_train2.sh`
- `model/modeling.py`
- `dataset/dataset.py`
- `dataset/dataset2.py`
- 三类 tokenizer

这些文件直接决定已保存 Phase 1/2 checkpoint 的含义。

### 下游验证入口

- MoleculeNet 分类/回归。
- QM9 判别式/生成式性质预测。
- ChEBI text2mol。
- mol2text caption。

它们用于判断预训练表示是否迁移有效，但当前存在多套实现，指标和数据划分必须逐脚本确认。

### 实验与历史副本

- `*-Copy*.py`
- `.ipynb_checkpoints/`
- `test*.py`
- `old.py`
- `representation_new.py`
- `process/tokenization/` 中的复制包

这些内容不能直接当作生产入口，应先确认是否被 import 或启动脚本引用。

## 5. 代码组织问题

1. 缺少统一的项目 README、环境锁定文件和一键配置。
2. 数据路径大量硬编码为 `/root/autodl-tmp/...`。
3. 多套下游脚本重复实现 dataset、collator、分布式 gather、评估与保存逻辑。
4. E3FP 包存在至少两份副本，远端 `process/` 下还有额外副本。
5. notebook checkpoint、`.pyc`、模型、数据和源码混在 Git 工作区。
6. 训练配置主要存在 shell 脚本中，没有与 checkpoint 一一绑定的结构化 manifest。
7. 最终模型目录未保存 tokenizer 和词表映射。
8. 本地与远端源码已经分叉，缺少版本标识。

## 6. 推荐的目标目录结构

```text
most-t5/
├── configs/
│   ├── pretrain_phase1.yaml
│   ├── pretrain_phase2.yaml
│   └── downstream/*.yaml
├── src/most_t5/
│   ├── model/
│   ├── tokenization/
│   ├── data/
│   └── evaluation/
├── scripts/
│   ├── prepare_data/
│   ├── train/
│   └── evaluate/
├── tests/
│   ├── test_tokenizer_determinism.py
│   ├── test_atom_motif_alignment.py
│   ├── test_collators.py
│   └── test_checkpoint_roundtrip.py
├── manifests/
│   ├── datasets/
│   └── checkpoints/
├── docs/
└── outputs/              # Git ignore
```

短期不需要马上搬动远端文件；应先通过文档和 manifest 建立逻辑边界，再逐步重构。
