# 05 证据记录与开放问题

## 1. 已完成的只读核验

| 编号 | 证据 | 结论 |
|---|---|---|
| E01 | `du` 盘点 `/root/autodl-tmp` | 90 GB 主要由 MoSt-T5、回收站和 3D-MoIT 构成 |
| E02 | 检查活动进程 | 当前没有训练/推理任务 |
| E03 | `git status` / `git diff --stat` | 远端是大量未提交实验工作的现场，不能自动清理 |
| E04 | checkpoint `trainer_state.json` | Phase 1 到 100K；Phase 2 到 30K；均无可靠 best validation 记录 |
| E05 | `cmp` 模型文件 | Phase 1 final=100K；Phase 2 final=30K 且有多份完全重复 |
| E06 | 比较 PubChem com/des Parquet | 三个 split 完全相同 |
| E07 | 检查 `.Trash-0` 稀疏 LMDB | 逻辑 1 TB 不代表真实占用；实际回收站约 26 GB |
| E08 | AST 解析远端 Python | 286 个 Python 文件语法可解析 |
| E09 | 本地/远端 SHA-256 | tokenizer 相同，其他关键训练文件已分叉 |
| E10 | 两个独立进程读取 `set` 词表 | motif 顺序不一致，token ID 非确定性风险得到直接验证 |
| E11 | `printenv PYTHONHASHSEED` 与启动脚本 | 当前环境和脚本均未固定 hash seed |
| E12 | 最终模型文件清单 | 未保存 tokenizer、vocab mapping 或 tokenizer config |

## 2. 证据等级

- 已实测：直接在远端执行只读命令得到。
- 代码直接证据：源码明确表现出的逻辑。
- 合理推断：基于 PyTorch/Transformers/Python 行为推断，但历史运行环境未完全保存。
- 待验证：必须通过实验回答。

## 3. 历史 checkpoint 开放问题

1. 历史多卡训练时各 rank 的 `PYTHONHASHSEED` 是否相同？
2. 是否存在未发现的 tokenizer JSON、SentencePiece 副本或 token-to-id 映射？
3. Phase 1 训练日志是否记录了关键 motif 的 ID？
4. 是否能从某个仍保存的 Python 进程、容器镜像或 notebook 输出恢复 rank 0 映射？
5. 即使 rank 0 映射可恢复，其他 rank 是否使用相同映射？
6. checkpoint 在固定 deterministic tokenizer 上的 motif reconstruction 性能是多少？

## 4. 数据开放问题

1. Phase 1 `pubchemqc_final.lmdb` 的真实样本数、字段完整率和重复率。
2. Phase 2 `phase2_pubchem_final.lmdb` 的真实样本数与任务字段覆盖率。
3. atom mapping 的平均覆盖率、零覆盖样本比例和截断后覆盖率。
4. E3FP 生成失败率、全 padding 比例和 shell 缺失率。
5. 20K/25K vocabulary 在各数据集上的 `<unk>` 比例。
6. 预训练数据与 MoleculeNet/QM9/ChEBI test 的分子和 scaffold 重叠率。
7. PubChem `com/des` 为什么是相同内容：命名错误、下载重复还是任务本来相同？

## 5. 模型开放问题

1. gate 的平均值、分布以及各层训练过程变化。
2. 局部注意力是否集中在合理原子，还是接近均匀。
3. 不同 E3FP shell 的 embedding 范数和贡献。
4. `lambda_3d=500` 是否导致语言重建被压制。
5. Phase 2 中 3D loss 降到 1 后是否仍有作用。
6. 移除 3D、随机打乱 atom mapping、随机打乱 E3FP 后性能下降多少。
7. 3D target 是否需要 EMA/frozen teacher。

## 6. 评估开放问题

1. 每个下游结果对应哪个脚本、commit、checkpoint、seed 和 split？
2. `eval_results.txt` 中多个重复 tag 分别属于哪些 run？
3. `ablation_results.txt` 为什么没有写入指标？
4. 生成式性质预测的解析成功率是多少？
5. Text2Mol 的 validity、uniqueness、similarity 是否按全测试集而非成功子集统计？
6. 分类指标是否处理单类 fold 和缺失标签？

## 7. 建议新增的 checkpoint manifest

每个模型目录至少保存：

```json
{
  "experiment_id": "phase1_clean_v1",
  "git_commit": "...",
  "dirty_diff_sha256": "...",
  "model_config": "config.json",
  "training_config": "resolved_training_config.json",
  "tokenizer_dir": "tokenizer/",
  "vocab_sha256": "...",
  "token_to_id_sha256": "...",
  "dataset_manifest_sha256": "...",
  "split_manifest_sha256": "...",
  "python": "...",
  "pytorch": "...",
  "transformers": "...",
  "rdkit": "...",
  "e3fp_source_sha256": "...",
  "cuda": "...",
  "seed": 42,
  "world_size": 8
}
```

## 8. 文档更新规则

后续每完成一项核验，应在本文件增加：

- 日期。
- 使用的数据/代码版本。
- 命令或脚本名称。
- 结果摘要。
- 对可行性判断产生的变化。

不要覆盖旧结论；若结论改变，应记录“原判断—新证据—新判断”。
