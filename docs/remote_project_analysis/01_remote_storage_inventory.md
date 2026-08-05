# 01 远端存储与数据目录清单

## 1. 实例状态

盘点时间：2026-07-17（Asia/Shanghai）。

| 项目 | 状态 |
|---|---|
| CPU | 0.5 核 |
| 内存 | 2 GB |
| GPU | 当前实例无 GPU |
| 系统盘 `/` | 30 GB，已用约 18 GB，59% |
| 数据盘 `/root/autodl-tmp` | 112 GB，已用约 90 GB，81% |
| 文件存储 `/root/autodl-fs` | 200 GB，已用约 378 MB |
| 活动任务 | 无训练或推理进程；仅 Jupyter、TensorBoard、AutoDL 服务 |

当前远端没有发现名称包含 `dock`、`vina` 或 `casf` 的项目目录。这一实例主要用于 MoSt-T5 分子多模态训练，而不是此前的 Vina benchmark。

## 2. 数据盘一级目录

| 路径 | 实际占用 | 类型 | 作用判断 |
|---|---:|---|---|
| `/root/autodl-tmp/MoSt-T5` | 45 GB | 核心工作区 | 代码、词表、数据副本、模型、checkpoint、日志和下游实验 |
| `/root/autodl-tmp/.Trash-0` | 26 GB | 回收站 | 被移入回收站的 PubChemQC LMDB 和少量脚本 |
| `/root/autodl-tmp/3D-MoIT` | 17 GB | 数据资产 | PubChem、PubChemQC、C4 的 LMDB 与拆分文件 |
| `/root/autodl-tmp/3D-MolT5` | 3.0 GB | 上游代码/数据处理 | 3D-MolT5 Git 仓库、本地 LMDB、词表处理代码 |
| `/root/autodl-tmp/e3fp-pubchemqc-prop` | 211 MB | Parquet 数据集 | PubChemQC 性质任务 |
| `/root/autodl-tmp/e3fp-mol-instructions-qm9` | 39 MB | Parquet 数据集 | QM9 指令与性质文本 |
| `/root/autodl-tmp/e3fp-pubchem-com` | 25 MB | Parquet 数据集 | PubChem 描述/生成数据 |
| `/root/autodl-tmp/e3fp-pubchem-des` | 25 MB | Parquet 数据集 | 与 `pubchem-com` 当前完全重复 |
| `/root/autodl-tmp/e3fp-mol-instructions-forward-reaction-prediction` | 19 MB | Parquet 数据集 | 正向反应预测 |
| `/root/autodl-tmp/e3fp-chebi-molgen` | 16 MB | Parquet 数据集 | ChEBI 文本-分子生成 |
| `/root/autodl-tmp/e3fp-uspto-50k` | 6.4 MB | Parquet 数据集 | USPTO-50K 反应数据 |

## 3. `3D-MoIT` 数据组成

`3D-MoIT` 当前几乎完全是训练数据，不是完整训练代码。

### 3.1 PubChemQC：约 8.7 GB

主要资产：

- `pubchemqc_database.lmdb`：逻辑大小约 3.87 GB，实际约 3.7 GB。
- `properties.csv`：约 398 MB。
- `pretrain/`：约 3.7 GB，属于拆分或展开后的预训练内容。
- `train/`：约 750 MB。
- `valid/`、`test/`：各约 94 MB。
- scaffold split、白名单、黑名单 JSON：用于数据隔离和下游划分。

判断：同一语料同时保留 LMDB 和展开后的 split 文件，存在格式级重复，但训练脚本可能分别依赖这些形式，不能仅按容量判断删除。

### 3.2 PubChem：约 7.0 GB

主要资产：

- `3d-pubchem-all.lmdb`：约 1.3 GB。
- `pretrain/`：约 5.6 GB。
- `train/`：约 84 MB。
- `valid/`：约 19 MB。
- `test/`：约 14 MB。
- `2d_computed_properties_all.csv`：约 35 MB。

Phase 2 启动脚本实际读取：

- `pubchem/pretrain/phase2_pubchem_final.lmdb`
- `pubchem/pretrain/phase2_text_weights.json`

### 3.3 C4 文本

- `c4_pretrain.lmdb`：约 0.94 GB。
- 用于 Phase 2 的 `[Denoise]:` 通用文本去噪任务，以降低纯化学多任务训练对 T5 语言能力的遗忘。

## 4. 外层 E3FP Parquet 数据集

记录数来自各数据集 `README.md` 元数据。

| 数据集 | Train | Validation | Test | 总量约 | 主要字段/用途 |
|---|---:|---:|---:|---:|---|
| PubChemQC property | 2,463,404 | 308,024 | 308,248 | 3,079,676 | `idx_3d`、task、SMILES、instruction、output、molecule_fp、SELFIES |
| QM9 instructions | 347,774 | 1,928 | 1,928 | 351,630 | instruction、output、molecule_fp、SELFIES、SMILES |
| Forward reaction | 124,384 | 1,000 | 1,000 | 126,384 | instruction、input、output、molecule_fp |
| PubChem com/des | 46,532 | 3,885 | 7,746 | 58,163 | CID、task、坐标、SMILES、文本、molecule_fp |
| USPTO-50K | 40,008 | 5,001 | 5,007 | 50,016 | reactant/product SELFIES、molecule_fp |
| ChEBI MolGen | 26,407 | 3,301 | 3,300 | 33,008 | CID、SMILES、molecule_fp、SELFIES、instruction、input |

已实测：`e3fp-pubchem-com` 与 `e3fp-pubchem-des` 的 train、validation、test 三个 Parquet 文件逐一字节比较完全相同。重复量约 25 MB，对空间影响很小，但反映命名与数据版本管理不够清晰。

## 5. `MoSt-T5` 容量组成

| 子目录/文件 | 占用 | 内容判断 |
|---|---:|---|
| `checkpoints/` | 约 18 GB | Phase 2、生成性质任务等 checkpoint |
| `MoSt-T5-Phase1-Final/` | 约 17 GB | Phase 1 最终模型及 80K–100K 五个完整 checkpoint |
| `dataset/` | 约 3.8 GB | PubChem LMDB 与 split |
| `autodl-tmp/` | 约 3.1 GB | 多个外层数据集的展开/转换副本 |
| `MoSt-T5-Phase2-Final/` | 约 1.2 GB | 去除 optimizer 的最终推理模型 |
| `MoSt-T5-Phase2-Final.zip` | 约 1.1 GB | 最终模型压缩包 |
| `checkpoint-test_final-20000/` | 约 1.2 GB | 另一实验模型 |
| `process/` | 约 307 MB | 数据处理脚本及 C4 分片等 |
| `ablation_logs/` | 约 112 MB | MoleculeNet/ADMET 消融日志 |

## 6. 模型与 checkpoint 关系

### 6.1 Phase 1

- 最终训练步数：100,000。
- 记录 epoch：约 26.259。
- 训练时间：64,506.77 秒，约 17.9 小时。
- `train_loss`：10.3231。
- 保存 checkpoint：80K、85K、90K、95K、100K，每个约 3.1 GB。
- `MoSt-T5-Phase1-Final/pytorch_model.bin` 与 `checkpoint-100000/pytorch_model.bin` 已验证完全一致。

每个完整 checkpoint 除约 1.19 GB 模型权重外，还包含约 2.36 GB optimizer 状态，用于恢复训练。

### 6.2 Phase 2

- 最终训练步数：30,000。
- 记录 epoch：约 101.835。
- 保存 checkpoint：24K、27K、30K，每个约 3.4 GB。
- checkpoint 日志末次 loss：24K 0.8874；27K 0.8854；30K 0.8961。
- 没有验证集、`best_metric` 或 `best_model_checkpoint`。

以下三份模型权重已验证完全相同：

1. `MoSt-T5-Phase2-Final/pytorch_model.bin`
2. `checkpoints/MoSt-T5-Phase2-Final/pytorch_model.bin`
3. `checkpoints/MoSt-T5-Phase2-Final/checkpoint-30000/pytorch_model.bin`

ZIP 内也是同尺寸的 Phase 2 最终权重。推理只需要其中一份完整模型目录；恢复训练则还需要 optimizer、scheduler、RNG 和 trainer state。

## 7. 回收站

`.Trash-0` 实际占用约 26 GB。主要文件：

| 文件 | 实际占用约 |
|---|---:|
| `pubchemqc_final.lmdb` | 7.2 GB |
| `pubchemqc_final_FAT.lmdb` | 7.2 GB |
| `pubchemqc_e3fp.lmdb` | 5.3 GB |
| `pubchemqc_e3fp_shrunk.lmdb` | 5.3 GB |
| `pubchemqc_e3fp 2.lmdb` | 62 MB |
| `e3fp-pubchemqc-prop/` | 278 MB |

部分 LMDB 的逻辑长度显示为 1 TB，但属于稀疏文件；真实磁盘占用应以 `du` 为准，而不是 `ls` 的逻辑文件长度。

当前只完成识别，没有执行清空回收站。

## 8. Python 环境

`/root/miniconda3` 约 19 GB：

| 环境 | 占用约 | 推测用途 |
|---|---:|---|
| `admet_ai` | 6.8 GB | ADMET/下游性质任务 |
| `3dmolt5` | 5.9 GB | MoSt-T5 / 3D-MolT5 主环境 |
| `asap` | 623 MB | ASAP 数据或任务脚本 |
| base 环境及共享库 | 约 5–6 GB | Jupyter、TensorBoard、基础依赖 |

Hugging Face 缓存约 1 GB，其中 hub 缓存约 961 MB、datasets 缓存约 56 MB。

## 9. 资产保留层级建议

本节只是分类，不代表授权删除。

| 层级 | 建议内容 |
|---|---|
| A：不可丢失 | 代码改动、最终 tokenizer 映射、词表、最终模型、训练配置、数据划分、关键日志 |
| B：可恢复训练 | 最新完整 checkpoint、optimizer、scheduler、RNG、trainer state |
| C：历史比较 | 旧 checkpoint、预测 TSV、TensorBoard runs、消融日志 |
| D：可再生成 | HF cache、部分展开数据、重复 Parquet、编译缓存 |
| E：待确认 | 回收站 26 GB、重复最终模型、ZIP、无清晰实验归属的 checkpoint |

若未来只保留推理模型、不再恢复训练，空间候选超过 50 GB；但必须先完成 tokenizer/checkpoint 可信度审计，并逐路径确认。
