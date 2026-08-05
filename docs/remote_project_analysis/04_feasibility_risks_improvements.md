# 04 可行性、风险与改进优先级

## 1. 总体判断

### 研究思路

结论：有可行性，且具有清晰的可检验创新点。

最有价值的假设是：局部 atom-to-motif 三维融合能比纯字符串、纯 motif 或全局分子指纹提供更好的跨模态与下游迁移表示。

### 当前工程与证据

结论：尚不足以确认已训练 checkpoint 可靠验证了该思路。

原因不是模型一定无效，而是 tokenizer、版本、评估和实验追踪存在足以混淆结论的问题。继续直接扩大训练规模，信息收益较低。

## 2. P0：必须先处理的问题

### P0-1 motif token ID 非确定性

状态：`已实测`。

代码路径：`tokenization/motif_tokenizer.py`。

问题机制：

1. 词表逐行读入 `self.motif_vocab_set`。
2. 调用 `self.tokenizer.add_tokens(list(self.motif_vocab_set))`。
3. Python `set` 迭代顺序依赖 hash seed。
4. 不同 Python 进程可能对同一 motif 分配不同 token ID。

远端验证：对同一 `vocab_20k.txt` 启动两个独立 Python 进程，仅检查 `list(set)` 中 `[C]、[O]、[N]、[C()]、[C()=O]` 的位置：

```text
进程 A: [9236, 6145, 6058, 16820, 6706]
进程 B: [5070, 8577, 4174, 3405, 8023]
```

同时：

- 远端当前没有设置 `PYTHONHASHSEED`。
- `run_train.sh` / `run_train2.sh` 没有设置它。
- Phase 1/2 最终模型目录没有 tokenizer 文件。

可能影响：

- 多卡不同 rank 将不同 motif 的梯度写入同一 embedding 行。
- decoder 同一 ID 在不同 rank 表示不同 motif。
- Phase 2 假设旧 ID 语义保持不变，但新 tokenizer 可能已经重排。
- 下游加载时无法知道训练阶段的确切 ID-to-token 映射。

修复：

```python
motif_tokens = []
seen = set()
for line in vocab_file:
    token = parse(line)
    if token and token not in seen:
        motif_tokens.append(token)
        seen.add(token)
self.tokenizer.add_tokens(motif_tokens)
```

必须额外执行：

- 每次保存 model 时同步 `tokenizer.save_pretrained(output_dir)`。
- 保存原始 vocabulary 文件及 SHA-256。
- 保存 `token -> id` JSON manifest。
- DDP 启动后在所有 rank 比较 tokenizer mapping hash，不一致立即退出。
- Phase 2 扩词表时验证旧 token 的 ID 逐项完全一致。

### P0-2 历史 checkpoint 可信度未知

状态：`高风险推断，待最小实验确认`。

无法仅凭 checkpoint loss 判断是否受到 tokenizer 多 rank 混乱影响。建议：

1. 不再基于当前 checkpoint 继续昂贵训练。
2. 用当前模型做固定小样本 motif reconstruction。
3. 分别以多个独立 tokenizer 进程加载同一 checkpoint，比较输出是否剧烈变化。
4. 构造已知 100–1000 个分子的 deterministic mapping，检查 checkpoint 是否能稳定重建。
5. 若无法恢复训练时 mapping，则把当前 checkpoint 降级为“历史实验资产”，重新进行小规模干净训练。

### P0-3 缺少验证集和最佳 checkpoint 选择

状态：`代码直接证据`。

Phase 2 没有 eval dataset、没有 `best_metric`，最终模型只是 30K 训练终点。24K/27K/30K 的训练 loss 也不能代表生成或下游质量。

修复：

- 为四个任务分别建立固定 validation subset。
- 每 N steps 计算任务级指标。
- 使用多指标选择，不只看总 loss。
- 保存 best-by-MMM、best-by-caption、best-by-text2mol 和综合 checkpoint。

## 3. P1：影响科学结论的问题

### P1-1 3D reconstruction target 是移动 latent target

状态：`代码直接证据 + 风险推断`。

target 由当前模型的 E3FP embedding 和 fusion attention 生成，只是通过 `no_grad/detach` 阻断 target 分支梯度。随着模型更新，target 也变化。

风险：

- 目标可能漂移。
- 模型可能学到内部自洽而非真实几何。
- `lambda_3d=500` 时可能过度影响 Phase 1。

改进对照：

- 冻结 teacher 或使用 EMA teacher。
- 直接预测固定 E3FP bit/shell 分类目标。
- 预测距离矩阵、原子对距离 bins 或构象不变量。
- 对 `lambda_3d` 做 0、1、10、100、500 消融。

### P1-2 任务路由配置失真

状态：`代码直接证据`。

远端 `dataset2.py` 接收 `task_probs`，但实际硬编码四任务等概率。配置日志可能让人误以为概率可调。

修复：使用传入概率并在训练日志记录每个任务的实际样本计数。

### P1-3 实验日志混写

状态：`已实测`。

`eval_results.txt` 有约 160 个条目，但只有约 60 个唯一 tag，51 个 tag 重复，最多重复 4 次。多个 run 被追加到同一文件。

影响：全局最好指标无法可靠对应某个配置或 checkpoint。

修复：每次 run 使用唯一 ID，输出独立目录：

```text
outputs/{experiment}/{timestamp}_{git_sha}_{seed}/
```

### P1-4 消融汇总为空

`ablation_results.txt` 中 BBBP、BACE、ClinTox 等 ROC-AUC 字段为空。必须从原始日志重抽取，并保留解析失败报告。

### P1-5 数据泄漏与 split 证据不足

需要核验：

- Phase 1/2 预训练集是否包含下游 test 分子。
- PubChemQC whitelist/blacklist 是否真正传入当前训练入口。
- MoleculeNet 是否统一使用 scaffold split。
- 生成任务是否按 canonical molecule 去重，而不只是字符串去重。

建议用 canonical SMILES、InChIKey 和 Bemis–Murcko scaffold 三层检查交集。

## 4. P2：工程与性能问题

### P2-1 本地/远端代码分叉

关键文件哈希不同，checkpoint 又没有记录完整源码版本。应先冻结远端 snapshot，再将修复迁入本地受控分支。

### P2-2 重复代码包

E3FP 至少有两份相同源码，远端 `process/` 下还有 tokenization 副本。应保留一个受测试版本，并通过包导入。

### P2-3 LMDB 初始化扫描全部 key

Dataset 启动时把所有 LMDB key 读入 Python list。百万级数据会增加启动时间和内存。可使用连续整数 key、只保存 key index 文件或 memory-mapped key manifest。

### P2-4 无效样本静默降级

E3FP 生成失败时返回全 padding tensor，Dataset 可能把它当成长度非零的合法样本。建议返回显式状态并统计：

- 构象生成失败率。
- E3FP 为空率。
- atom mapping 覆盖率。
- `<unk>` motif 比例。

超过阈值时训练应中止，而不是静默继续。

### P2-5 训练脚本配置不一致

- Phase 2 注释称单卡，但固定 8 GPU。
- `--optim adamw_torch_fused` 重复。
- 路径、旧 vocabulary size、GPU 数量硬编码。
- 没有 requirements/lockfile 和环境版本 manifest。

建议迁移到 YAML 配置，并在输出目录保存解析后的完整配置。

## 5. 下游评估风险

### 5.1 多套实现并存

性质预测存在顶层、`process/`、MPNN、LoRA、全量微调等多套脚本。它们的：

- tokenizer 用法。
- E3FP shell 数。
- atom mapping 生成方法。
- split 文件。
- pooling。
- metric。

并不完全一致，不能混在同一表中直接比较。

### 5.2 某些旧脚本可能绕过 MotifTokenizer

部分 property 脚本直接调用底层 Hugging Face tokenizer 对 SMILES 编码，而不是 `MotifTokenizer.encode`。这会让预训练使用的 motif 语义与下游输入不一致，应标记为旧/实验实现并验证。

### 5.3 E3FP level 配置不统一

有脚本使用 `fp_level=4`，而主模型常用 `e3fp_num_levels=4`，含义分别可能是 5 层与 4 层。需要统一约定：`num_levels=4` 对应 level 0–3。

## 6. 推荐改进顺序

### 第一阶段：让实验可解释

1. 修复 deterministic tokenizer。
2. 保存 tokenizer、vocab 和 mapping hash。
3. 建立 checkpoint manifest。
4. 固定小规模 train/validation/test。
5. 加入 tokenizer、mapping、collator 单元测试。

### 第二阶段：最小可行性实验

比较：

- T5 + SMILES。
- T5 + motif。
- motif + 全局 E3FP。
- motif + 局部 atom-to-motif E3FP。
- 上述模型加/不加 3D reconstruction。

只使用小数据和短训练，先判断增益方向和实现正确性。

### 第三阶段：训练目标优化

- 扫描 `lambda_3d`。
- 比较固定/EMA 3D target。
- 比较不同任务采样策略。
- 检查 gate 使用率、3D attention 熵和 mapping coverage。

### 第四阶段：正式规模训练

只有在 deterministic 小实验能够稳定复现后，才进行大规模多卡训练。

## 7. 可行性判定门槛

满足以下条件后，才能认为思路得到初步支持：

1. 相同 seed 重复训练指标波动可控。
2. tokenizer mapping 跨进程和保存/加载完全一致。
3. 局部 3D 融合在至少两个结构敏感任务上稳定优于 motif-only。
4. 优势在 scaffold split 和多个 seed 上成立。
5. 数据泄漏检查通过。
6. 3D 增益不是由参数量增加解释：需要参数量匹配 baseline。
7. 消融表能追溯到独立日志、配置和 checkpoint。
