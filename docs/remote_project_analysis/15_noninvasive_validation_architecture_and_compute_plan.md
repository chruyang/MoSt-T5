# 15 非侵入式 P0 验证架构与计算资源计划

> 制定日期：2026-07-30  
> 目标：保留当前 MoSt-T5 主干作为可复核基线；用旁路验证证明真实数据流；仅在证据明确后，以最小候选模块实现修复或方法创新。  
> 主参考：3D-MolT5 官方仓库（commit `82dbe088e424f19fa713dbd657f5235990bd324f`）与 ICLR 2025 论文；它分开 3D tokenization、模型、微调和评测脚本，但本项目不照搬其原子级对齐。

## 1. 决策：两层目录，而不是第二套项目

```text
当前主干（冻结、只读验证）
  train1.py / train2.py / model/ / dataset/ / tokenization/ / process/
                │
                ├── p0_validation/     # 立即建立：旁路验证，只调用主干
                │
                └── most_t5_next/      # 仅在 P0 证明必要后：单模块候选改动
```

不要复制整套 `model`、`dataset` 或 `tokenization` 到新目录。那样会制造第二份行为可能漂移的实现，最终无法证明训练使用的是哪一条路径。原主干保留为 baseline；每份报告都记录其文件 SHA-256、词表 hash、LMDB hash、启动命令和环境版本。

3D-MolT5 的公开仓库也将 `3d_tokenization` 与 `3d_molt5`、`finetune_scripts`、`eval_scripts` 分开，并以薄入口协调配置和训练。[官方仓库](https://github.com/QizhiPei/3D-MolT5) 因此是“职责分层”的参考，而不是可直接替换本项目 motif 聚合逻辑的模板。

## 2. `p0_validation/` 的最小结构

```text
p0_validation/
├── README.md
├── spec.py                 # 路径、phase、seed、阈值和 release 标识
├── baseline_factory.py     # 原样构建 Phase 1/2 tokenizer、dataset、collator、model
├── raw_audit.py            # 只读 LMDB 全量扫描，不经过 Dataset 容错
├── contracts.py            # Record/Item/Batch/Forward trace 断言与 Finding
├── trace.py                # 固定 seed 调用真实 Dataset -> Collator -> Model
├── smoke.py                # 真实 forward/backward、finite/gradient 和显存 profile
├── launch_check.py         # shell、解析后参数、路径、词表、任务路由核验
├── run_p0.py               # 唯一 CLI；不承载化学或训练业务逻辑
├── fixtures/p0_cases.json  # golden key、record hash、seed、允许差异
└── reports/<run-id>/       # manifest、jsonl、摘要；不复制原始 LMDB/大张量
```

核心调用接口保持六个，不为每个张量变换构造一层抽象：

```text
build_baseline(spec, phase) -> components
scan_lmdb(spec) -> audit_summary
trace_index(components, index, seed) -> pipeline_trace
assert_contract(trace, phase) -> findings
run_smoke(model, batch, device) -> smoke_result
write_release_report(...) -> manifest + report
```

`baseline_factory` 必须逐参数镜像 `train1.py` 和 `train2.py` 的 tokenizer/model/dataset/collator 构造；P0 不使用“自行推断的等价配置”。Trainer 只负责把 `model(**inputs)` 的 loss 交给 Hugging Face，因此 smoke 可直接调用真实 model forward，避免再建立一个训练抽象。

## 3. 应复用与不应复用的粒度

| 语义边界 | 直接调用的主干 | P0 只增加的能力 |
|---|---|---|
| record → item | `GSMATDataset.__getitem__` | 记录实际 key、task、fallback/重试、SMILES、token/atom-map/E3FP 摘要。 |
| item → batch | Phase 1/2 Collator | 捕获 mask 前后张量、sentinel labels、padding、文本段位移。 |
| batch → model | `MoStT5ForConditionalGeneration.forward` | 用 forward hook 记录 GSMATEmbeddings、GeoSemanticFusion、encoder、geometry head 的 shape/finite 性。 |
| output → report | 原始 `MoStT5Output` | 提取 logits shape、CE、3D loss、有效 mask 数、梯度范数、显存峰值。 |

不应在 P0 中重写 `linearize`、`decode_linear`、E3FP 计算、motif tokenizer、完整 Dataset/Collator 或 `MoStT5ForConditionalGeneration.forward`。它们正是需要被验证的化学/训练语义。`raw_audit.py` 可以直接读取实际 LMDB，原因是 Dataset 当前会在读取失败时跳到下一索引、E3FP 缺失时从 SMILES 重算、宽度不符时 pad/截断；这些容错会掩盖原始 release 的问题。

对 `process/` 脚本同样不应在 P0 中批量重跑。P0 验证对象是**实际训练 LMDB**；只有定位某个 golden record 时，才单独调用无副作用的纯函数复算该样本。

## 4. 首个测试：验证层本身不得改变主干

先证明下面的等式，再讨论任何候选改进：

```text
直接调用当前主干 == 经 p0_validation 调用当前主干
```

对同一 record、词表、checkpoint 和 Python/NumPy/PyTorch CPU/CUDA seed，逐级比较：

1. `MotifTokenizer.encode` 的 token IDs、mapping 和 topology round-trip；
2. Dataset item 的 task、text/motif IDs、E3FP shape、atom map；
3. Collator batch 的所有 tensor、labels、sentinel 顺序、合并后的 `mask_positions`；
4. eval forward 的 logits、CE、3D loss（浮点使用预定义容差）；
5. train-mode 最小 backward 的有限性、存在的梯度和梯度范数。

报告还必须记录 `PYTHONHASHSEED`。当前 motif 词表由 `set → list` 注册，因此不同进程可能产生不同 token ID；这一项是阻断项，不能用一次成功的 trace 掩盖。

现有 Collator 只输出合并后的 `mask_positions`。P0 可在不改算法的前提下由原始 `input_ids` 与 masked `input_ids` 的差异观测：

```text
joint_mask    = (masked_input_ids != original_input_ids)
geo_only_mask = mask_positions & ~joint_mask
joint_mask | geo_only_mask == mask_positions
```

这足以证明当前真实行为。只有后续要让两类 mask 成为正式 loss API 时，才把它们变成候选 Collator 的显式字段。

## 5. `most_t5_next/` 的最小迁移规则

候选目录不承载“翻新版项目”，只放确有必要的差异模块，例如：

```text
most_t5_next/
├── tokenization/stable_motif_tokenizer.py  # 确定排序并冻结 vocab hash
├── data/phase2_dataset.py                  # TaskRouter、严格 mapping/error 策略
├── data/collators.py                        # joint/geo-only mask 显式输出
├── model/geometry_target.py                 # 仅在 EMA target 方案进入实验时新增
├── train_phase1.py
└── train_phase2.py
```

未改变的模块明确从原主干导入，禁止用 `sys.path` 阴影覆盖 `dataset` 或 `tokenization`。每迁移一个模块，先运行 baseline parity；若该模块是有意修复，则在 `fixtures/p0_cases.json` 中预先写明允许差异和新的化学真值。

最紧急的例子是 Phase 2 mapping：本地 `process/build_phase2_ready_lmdb.py` 用 `_, _, mapping = linearize(sub_smi)`，而当前 `linearize` 返回 `(frag_string, atom_mapping, [])`。若远端 LMDB 的构建哈希与此一致，保存的 mapping 可能为空，导致训练的 `atom_to_motif_map` 都为 `-1`。在没有对远端 source/hash 与全量 LMDB 做验证前，不能把它当成已确认的远端故障；但它必须排在 P0 最前面。

## 6. GPU 租用策略：先测，再扩容

当前代码加载 `google/t5-v1_1-base`，并额外增加四层 E3FP embedding、motif-atom attention、geometry head 和无梯度 3D target forward。Phase 1 配置为长度 512、每卡 batch 64、累积 4；Phase 2 为长度 768、每卡 batch 8、累积 16，并开启 gradient checkpointing。真实峰值还由 LMDB 的 atom count p99/max 决定，远端当前拒绝连接，尚未取得这一分布。

| 目标 | 推荐 GPU | 其他资源 | 说明 |
|---|---|---|---|
| P0 全量数据审计 | 无 GPU | 8–16 vCPU、64 GB RAM、至少 250 GB NVMe | 先查数据，不为扫描租 H100。 |
| P0 五类任务 smoke / 显存测量 | 1×48 GB L40S 或 RTX 6000 Ada；24 GB 仅 batch=1 冒烟 | 16 vCPU、64 GB RAM、250 GB NVMe | 用真实 p99 和最大样本测 OOM 边界。 |
| 单卡开发 | 1×48 GB | 24–32 vCPU、64–128 GB RAM、500 GB NVMe | 从 micro-batch 1/2 开始，不能默认 Phase 1 的 64 能装入。 |
| 首轮训练 / 系统消融 | 同机 4×A100 80 GB 或 H100 80 GB | 32+ vCPU、128 GB RAM、1 TB NVMe | DDP 下以梯度累积固定全局 batch；先做 NCCL smoke。 |
| 顶刊主结果 / 多 seed 重复 | 同机 8×H100 80 GB SXM/NVSwitch；8×A100 80 GB 为替代 | 64+ vCPU、256 GB RAM、1 TB+ NVMe | 只用于最终方案和关键消融。 |
| 80 GB 被真实 profile 证明不足 | 4/8×H200 141 GB | 同上 | 条件升级，不预设为必要。 |

L40S 具有 48 GB ECC 显存并支持 BF16，但没有 NVLink，适合单卡校准；A100 80 GB 具有高带宽 HBM2e 和 NVLink，适合稳定的多卡 DDP；租 4/8 卡时须确认同机、`nvidia-smi topo -m` 拓扑及是否具备 NVLink/NVSwitch，而不是只看商品标题。[L40S 规格](https://www.nvidia.com/en-au/data-center/l40s/)；[A100 规格](https://www.nvidia.com/en-us/data-center/a100/)。3D-MolT5 以 8×80 GB A100 进行其 255M 规模的完整预训练，可作为顶刊级全量训练的上界参照，但不能替代本项目的真实 profile。[3D-MolT5 论文](https://arxiv.org/html/2406.05797)

## 7. 付费资源前的强制校准

先短租 1×48 GB，依次完成：

1. 统计每类任务的 `input_len`、`label_len`、`atom_count`、E3FP 宽度、mask 数量的 p50/p95/p99/max；
2. 对 Phase 1 MMM、Phase 2 MMM/caption/text2mol/denoise 各选 p99 与最大样本，按 micro-batch `1, 2, 4, 8, ...` 进行 3 次 warm-up 与 10 个真实优化 step；
3. 分别 profile CE-only、E3FP fusion without geometry loss、完整 3D target + MSE；
4. 保存 `max_memory_allocated`、`max_memory_reserved`、OOM 点、tokens/s、GPU 利用率、dataloader 等待和 NCCL 时间；显存保留至少 20% 余量；
5. 在 4 卡节点先通过 DDP/NCCL smoke，再决定是否为最终多 seed 结果租 8 卡。

当前 Phase 1 的 launcher 虽写 16 个 data-loader workers，但 `train1.py` 会以 `DataArguments.num_workers` 覆盖，默认实际为 4；Phase 2 才使用 16 workers。这也是必须保存 resolved config、而非仅保存 shell 脚本的原因。
