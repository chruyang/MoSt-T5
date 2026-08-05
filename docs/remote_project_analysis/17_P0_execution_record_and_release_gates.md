# 17 P0 执行记录：数据、张量流、确定性与放行门

> 执行日期：2026-07-30  
> 结论：**P0 未通过，禁止以现有 Phase-1 checkpoint 启动正式 Phase-2 训练。**  
> 范围：只读审计远端 MoSt-T5 主干、LMDB、checkpoint 与启动脚本；所有新产物仅写入 `/root/autodl-fs/most-t5-p0/` 和 `/root/autodl-tmp/p0-scratch/most-t5/`。未修改原始代码、原始 LMDB 或已有 checkpoint。

## 1. 主干审查结论

P0 已经确认两类正面事实：

1. 当前 Phase-2 分子 LMDB 和 C4 LMDB 在 pickle/schema、SMILES、文本非空、E3FP shape/range 等基础结构层面总体可读；四个 Phase-2 路由能够在**受控单进程、固定种子、少量真实记录**上产出符合接口约束的张量。
2. Phase-1 checkpoint 的词表扩容存在一个可验证的**数值级**正确加载顺序：先按原词表严格加载，再扩容并初始化新增行；它能保留原有 52,306 行的所有权重。

但这不足以放行训练。最关键的失败不是“某个张量 shape 不匹配”，而是 checkpoint 中第 `i` 行 embedding 到底对应哪个 motif 已无法由现有资产证明。由于 motif token ID 在不同 Python 进程中会变化，当前 Phase-1 权重、当前 Phase-2 tokenizer 与原始 Phase-1 tokenizer 之间没有可审计的语义映射。即使采用正确的扩容顺序，也只能保留未知语义的数值行。

因此，下面的 P0 证据应当用于决定修复顺序，而不能作为“当前系统可复现实验结果”的背书。

## 2. 已冻结的运行事实

| 项目 | 已确认事实 | 含义 |
|---|---|---|
| 机器 | 单张 RTX 4090，24 GB 显存；当前可见物理 GPU 数为 1 | `run_train2.sh` 声明 8 卡，不能在这台机器直接执行原启动命令。 |
| 代码现场 | `/root/autodl-tmp/MoSt-T5`，Git HEAD `d051236`，工作树有既存改动 | P0 manifest 冻结了核心文件和工件 hash；不能把 Git commit 单独当作训练现场。 |
| Phase-1 正式数据 | launcher 指向的 `pubchemqc_final.lmdb` 不存在 | 不能把回收目录或 Trash 中的候选 LMDB 静默替换为正式数据。 |
| Phase-1 候选数据 | recovery LMDB 有 3,899,644 条数据，且为 heavy-atom E3FP；无文本字段 | 可作只读 trace 候选，不能替代正式来源或支撑“三模态 Phase-1”叙述。 |
| Phase-2 分子数据 | `phase2_pubchem_final.lmdb`：301,655 条数据记录 | 全量结构审计完成；存在需要显式政策处理的 source-explicit-H 边界类。 |
| C4 数据 | `c4_pretrain.lmdb`：300,000 条文本记录 | 全量基础可读性检查通过；长度截断分布另见第 6 节。 |
| Phase-1 checkpoint | `checkpoint-100000`，`vocab_size=52,306` | 没有保存 tokenizer、added-token 映射或完整 token-ID manifest。 |
| Phase-2 词表 | 当前构造后 `vocab_size=57,306` | 不能假设前 52,306 个 ID 仍代表 Phase-1 中相同的 motif。 |

## 3. P0 通过的检查与其边界

| 检查 | 结果 | 它证明了什么 | 它没有证明什么 |
|---|---|---|---|
| 全量原始数据审计 | Phase-2、C4 通过基础 schema / pickle / SMILES / E3FP 检查；recovery Phase-1 候选可读 | 当前文件不是普遍性损坏或不可读 | 正式 Phase-1 数据来源真实、构象能量/立体化学/坐标的化学物理正确性、几何语义统一或历史训练可复现 |
| 四任务数据流 trace | MMM、caption、text2mol、denoise 在固定 `PYTHONHASHSEED` 和随机种子下可重复 | 当前主干在受控单进程样本上的 Dataset → Collator 张量接口可走通 | 多 worker、多卡、历史 checkpoint 的 token 语义正确，或真实训练已完全确定性 |
| Phase-2 文本审计 | 分子文本和 C4 文本均非空；text weight 文件范围正常 | 文本字段/权重没有明显缺失或非法值 | 截断不会改变任务结论或造成语料偏差 |
| 词表扩容 sidecar gate | “严格加载 52,306 → 扩容 57,306”保留全部旧行；当前 `train2.py` 顺序丢失全部旧行 | 数值级 checkpoint 扩容修复方案是可行的 | 保留的旧行仍对应正确 motif |
| 小型随机 forward | 主要接口可运行、无立即崩溃 | 基本工程可执行性 | 当前 checkpoint 的端到端科学有效性；该测试不能替代恢复 checkpoint 语义后的真实 smoke test |

## 4. 阻断项：按优先级处置

| ID | 严重度 | 已观察事实 | 为什么阻断 | 最小放行条件 |
|---|---|---|---|---|
| G1 | **P0 / 致命** | `MotifTokenizer` 先把 motif 放入 Python `set`，随后 `list(set)` 注册 tokenizer。不同 `PYTHONHASHSEED` 下相同 motif 获得不同 ID；P1 checkpoint 未保存 tokenizer 快照。即使固定 hash seed，20k → 25k 的 set 构造也会移动已有 motif ID。 | embedding、decoder input 和 LM head 的第 `i` 行没有可证明的 motif 身份；所有“从 P1 初始化 P2”的比较都会混入不可解释的词表置换。 | 找回原 P1 的完整 `id_to_token` 映射并按 token 语义重排所有相关矩阵，**或**使用稳定 tokenizer 重新训练 P1。之后每个 checkpoint 必须携带 tokenizer 和 hash。 |
| G2 | **P0 / 阻断** | Phase-1 launcher 指定的正式 LMDB 不在机器上；仅有来源未闭合的 recovery 候选。 | 无法确认 P1 训练实际使用了什么数据和 E3FP 原子约定，也不能复现实验。 | 找回带 provenance manifest 的正式 P1 release；若不能找回，明确将 recovery 视为新数据版本并重新建立、验证和训练。 |
| G3 | **P0 / 阻断** | `run_train2.sh` 固定 `NUM_GPUS=8`，当前主机物理 GPU 数为 1。 | 现有 launcher 不代表当前机器的真实、可控运行配置。 | 在不改变研究配置含义的前提下建立单卡 profile / dry-run，保存 resolved config、有效 global batch、显存和吞吐；多卡训练另做 DDP/NCCL 验收。 |
| G4 | **P0 / 阻断** | Phase-2 全量扫描发现 353 条 source-explicit-H / 同位素 H / 质子记录，其 `atom_mapping` 指向 E3FP padding；共出现 1,508 个无有效几何注入的 motif group。 | 标准 RDKit `AddHs` 的 padded H 行是设计内行为；但这 353 条是源 SMILES 自带 H，当前实现会静默让相应 motif 不接收 3D 信号。“所有 3D motif 都被增强”的说法不成立。 | 选择并实现一项可审计政策：过滤并记录这些样本、重建该类 E3FP、将 source-explicit-H 显式映射为无 3D view，或单独做消融；生成新不可变 release 后重审。 |
| G5 | **论文主张阻断** | P1 recovery 候选没有文本；P1 collator 实际只消费 motif、E3FP 与 atom map。 | “P1 已进行 3D、2D、文本三模态预训练”的叙述与当前可验证实现不一致。 | 修正文稿为 P1 的 2D-motif + 3D 预训练、P2 才引入文本，或建立同谱系的文本条件 P1 并重新验证。 |

### G1 的两个彼此独立的错误

G1 不能仅靠“设置一个固定 seed”解决：

1. **历史不可恢复性**：历史 P1 启动脚本未固定并保存 `PYTHONHASHSEED`，checkpoint 又没有 tokenizer。固定今天的 seed 不能反推出当时的映射。
2. **跨阶段重排性**：即使 P1 和 P2 使用同一个固定 hash seed，往 `set` 中加入更多 motif 后，容器布局和迭代顺序仍可让原 motif 的 ID 移动。因此把 `0:52306` 当作“原 P1 词”没有语义保证。

这也是为什么“先修 `train2.py` 的 embedding 初始化”是必要但不充分的：它修复了数值覆盖，未修复 token 身份。

## 5. 当前代码中的真实任务与损失流

下表是固定种子 trace 的接口事实，而不是设计愿景。

| P2 任务 | 编码器输入 | E3FP / atom map | 标签 | 3D MSE 实际状态 |
|---|---|---|---|---|
| MMM | 文本前缀 + 文本 + motif | 有；文本区为 dummy 行，atom map 偏移到 motif 区 | masked span 目标 | **唯一会形成 `mask_positions` 并计算几何 MSE 的任务** |
| Caption | 分子 motif（含 3D 输入） | 有 | description token | `mask_positions` 为空，几何损失为零 |
| Text2Mol | 文本 prompt | 空 3D view | motif sequence | 几何损失为零 |
| Denoise | C4 text | 空 3D view | T5 span-denoising target | 几何损失为零 |

因此，当前实现应准确称为“MMM 子任务上的 CE + **masked latent-geometry/self-distillation MSE**，与其他任务的 CE 联合训练”，而不是“四类任务都叠加 MSE”。这里的 target 是同一模型从未遮蔽 E3FP 融合表示在线构造的 latent target，并非对原始三维坐标或离散 E3FP ID 的直接回归。任何后续 `lambda_3d` 消融也必须同时报告 MMM 采样比例、有效 3D mask 数、MSE 原始值和加权值。

固定 trace 中的两个真实分子 batch 示例如下：MMM 的 `input_ids` 为 `[2, 470]`、`e3fp_ids` 为 `[2, 465, 4]`，有 12 个 motif 几何 mask 位置；caption 为 `[2, 46]` 与 `[2, 41, 4]`。Text2Mol 与 Denoise 的 E3FP shape 均为 `[2, 0, 4]`。这些是 golden trace 证据，不能外推成完整训练的长度分布。

## 6. 非阻断但必须进入实验设计的事实

| 主题 | 测得结果 | 研究/工程含义 |
|---|---|---|
| caption label 截断 | 16,384 个确定性抽样中 1,667 条（10.175%）超过 512 | 目标文本有不可忽略的 prefix truncation；结果中应报告该比例并做长度敏感性分析。 |
| Text2Mol 文本截断 | 1,685 / 16,384（10.284%）超过 512 | 文本到分子任务可能丢失末尾条件。 |
| C4 截断 | C4 实际使用 512，而非 launcher 标出的 768；4,955 / 16,384（30.243%）超过 512 | 当前是文档前缀截断，可能带来位置偏差；应在方法中明确或改为可复现窗口策略。 |
| task special-token 元数据 | TextTokenizer 注册任务 token 后，MotifTokenizer 再写入自己的 `additional_special_tokens`，使 task token 不在实际 `all_special_ids` 中 | 当前手工前缀仍可跑，但依赖 generic special-token API 的逻辑会脆弱；应合并已有 special token 并添加回归测试。 |
| 任务采样配置 | Dataset 内部默认任务权重与 `train2.py` 传入后的 resolved `phase2_task_probs` 不是同一概念 | golden trace 只验证路由契约，不能代替真实训练的任务暴露比例；每次训练必须持久化 resolved task probs 与实际计数。 |
| 随机性 | task 由 NumPy 全局 RNG 选取，mask / E3FP dropout 使用 Torch 全局 RNG；模型中还有调试性 `torch.randint` | 固定 `PYTHONHASHSEED`、Python/NumPy/Torch seed 和 worker seed，并删除或隔离调试随机数。 |
| Phase-1 workers | `run_train.sh` 传入 workers=8，但 `train1.py` 最终用 DataArguments 默认值 4 覆盖 | 吞吐/资源估算必须以 resolved config 为准。 |

## 7. 证据索引

所有远端报告均在共享文件存储，换机后仍可读取。它们只包含摘要、hash、统计与小型 trace，不复制 LMDB 或 checkpoint。

| 证据 | 远端路径 | 主要结论 |
|---|---|---|
| 完整 Phase-2/C4 语义审计与 release manifest | `/root/autodl-fs/most-t5-p0/releases/p0-raw-phase2-semantics-v3-20260730T2330/` | 301,655 条分子记录、300,000 条 C4 记录；输出 G2/G3/G4 等 gate。 |
| recovery Phase-1 原始审计 | `/root/autodl-fs/most-t5-p0/releases/p0-raw-full-20260730T2255/` | 候选 LMDB 可读但缺少正式 provenance，且无文本。 |
| 四任务 tensor flow 与确定性 trace | `/root/autodl-fs/most-t5-p0/reports/p0-trace-20260730T2238+0800/` | 固定 hash seed 的两次 JSON trace 字节一致；记录四任务 shape、mask 与随机源。 |
| 文本契约审计 | `/root/autodl-fs/most-t5-p0/reports/p0-text-contract-20260730T2245+0800/full-16384/` | 文本完整性、长度分布、任务 token 元数据及 text weights。 |
| tokenizer 谱系 gate | `/root/autodl-fs/most-t5-p0/reports/p0-tokenizer-gate-20260730T2315/` | P1/P2 在不同 hash seed 下 token map 不同；checkpoint 无 tokenizer 工件。 |
| embedding 初始化 gate（复跑） | `/root/autodl-fs/most-t5-p0/reports/p0-embedding-gate-v2-20260730T2330/embedding_gate.json` | 正确顺序保留五个矩阵的所有旧行；现有 `train2.py` 顺序使五个旧前缀均丢失。 |
| 旁路验证脚本 | 本地 `p0_validation/`，远端 `/root/autodl-fs/most-t5-p0/harness/p0_validation/` | 脚本调用现有主干，不替代或覆盖原实现。 |

## 8. 继续前的最小修复序列

按以下顺序执行，避免把数据谱系问题误当成模型改进：

1. **冻结当前 P0 现场，不启动 P2。** 保留上述报告、代码 hash 与原始 checkpoint，所有修复在 `most_t5_next/` 或新 release 目录完成。
2. **先处理 tokenizer 谱系。** 搜索历史运行目录、归档镜像、日志附件和其它机器是否保存过 P1 tokenizer / `added_tokens.json` / manifest。若找到，构造 token-string 到旧/new ID 的双向表，并按 token 语义重排 `shared`、encoder input embedding、GSM embedding、decoder input embedding 与独立 LM head。若找不到，P1 必须使用稳定、文件顺序去重的 tokenizer 重训。
3. **为新 tokenizer 建立 release contract。** 不允许 `set -> list` 产生词表顺序；显式保留词表文件顺序或排序规则。每个 checkpoint 保存完整 tokenizer、`id_to_token`、special-token 表、词表源文件 hash、构造代码 hash 和 `PYTHONHASHSEED`。
4. **选择统一的 3D 原子宇宙政策。** 推荐新 release 明确采用 heavy-atom E3FP，或显式维护 `heavy_to_e3fp_row`。将 source-explicit-H 的处理写入数据 schema、reject/warning 清单和消融实验，不依赖 padding 的隐式语义。
5. **修复加载与可复现启动。** 在 tokenizer 语义已恢复后，将 P2 初始化改为“原 vocab 严格加载 → 扩容 → 仅初始化新增 token 行”；合并 special tokens；固定所有随机源和 worker seed；保存 resolved config；单卡先做 profile，不调用 8 卡 launcher。
6. **重跑 P0，并新增真实 checkpoint smoke。** 在 G1/G2/G4 关闭后，用真实 checkpoint 对 P1 MMM 与四个 P2 任务做 forward/backward、finite、梯度和 loss-routing 验收；然后才进入 `lambda_3d`、MSE 形式、motif 划分的科学消融。

## 9. P0 签字条件

以下条件全部满足前，P0 状态保持 `BLOCKED`：

- 正式 P1/P2 数据 release、E3FP 规范、provenance、tokenizer 和 checkpoint 可一一追溯；
- P1 checkpoint 的每个 token ID 有可审计的 motif 身份，P1→P2 的 embedding 继承按 token 语义而不是行号完成；
- source-explicit-H 边界类有明确、全量可审计的处理并完成重新验收；
- 启动命令与真实硬件、resolved config、随机状态一致；
- 文稿中的 P1/P2 模态与实际数据流一致；
- 真实 checkpoint 的五条路径 smoke test 和随后的容量 profile 通过。

在这之前，继续讨论 CE+MSE 权重、EMA target、motif 粒度或下游增益会缺少可信的因果基础。
