# R0 主干审视：实际数据流与 P0 放行裁定

**裁定：P0/R0 不放行训练（BLOCKED）。**

这不是对研究方向的否定。两阶段设计仍可作为可检验的主线；但当前实现不能把
“P1 已学到 3D+语法、P2 已在其上完成对齐”作为可信实验前提。若现在直接训练，
数据泄漏、token-ID 语义漂移、未继承的 checkpoint embedding 和无效几何 target
会混在一起，任何 CE/MSE 或 motif 改进都无法归因。

所有检查均在远端完成，且未把数据资源下载或复制到本机。

## 实际流程与原设想的对应关系

| 阶段 | 原设想 | 当前代码的真实行为 | 主干判断 |
| --- | --- | --- | --- |
| P1 | 3D 与“语法”基础学习 | 固定 MMM；collator 不将文本传给模型，只进行 motif 序列复原 + 3D 掩码恢复 | 可称 **2D-motif + 3D structural pretraining**，不能称文本/三模态预训练 |
| P2 | 多模态对齐 | MMM、caption、text2mol、denoise 四任务默认各 25% | 总体方向符合对齐；但各任务的 3D 参与不相同 |
| P2 MSE | 基于 E3FP 融合 3D 监督 | 仅 MMM 在有几何 mask 时计算 CE + masked latent-geometry MSE；caption/text2mol/denoise 不产生该 MSE | 应准确表述为 **MMM 的 latent geometry reconstruction/self-distillation**，不是坐标回归，也不是所有任务共享的 3D Loss |

关键代码证据：

- P1 MMM 固定于远端 `train1.py:171-173`；文本虽在 dataset 层生成，但
  `GSMATPretrainingCollator` 只读取 motif/E3FP/map（`dataset/dataset.py:395-420, 450-458`）。
- P2 task 采样见 `train2.py:279-289`、`dataset/dataset2.py:113-153`；四任务的
  3D 路由分别见 `dataset/dataset2.py:503-586`。
- MSE target、mask gate 和 CE 加权相加见 `model/modeling.py:273-286, 310-344`。

## R0 已证实的数据成员边界

| 分支 | 原始正式 membership | 当前可用 E3FP membership | 不可静默忽略的差异 |
| --- | ---: | ---: | --- |
| P1 PubChemQC pretrain | 3,119,717 | 3,119,714 | 3 条 H₂ / explicit-H E3FP 不支持样本 |
| P2 PubChem pretrain | 301,658 | 301,655 | 3 条单原子、无 pairwise-distance 的 E3FP 样本 |

P1 原始 LMDB 与官方 3,899,647 个 split union 完整一致；P2 的 301,658 条
pretrain 与其 12k/1k/2k holdout 完整且互斥。因而两个“三条差异”不是 split
误用，而是可定义的几何表示边界。完整证据见：

- [P1 成员审计](19_R0_phase1_membership_audit.md)
- [P2 成员与 singleton 策略](20_R0_phase2_membership_and_singleton_policy.md)

## 必须关闭的硬阻断

1. **P1 whitelist 未接入。** `train1.py:165-174, 178-187` 将
   `whitelist_path=""` 传入 dataset；而只有非空且存在的名单才过滤
   （`dataset/dataset.py:62-68`）。若直接换向 raw LMDB，会把下游集合混入 P1。
2. **P1 launcher 指向不存在的 final LMDB。** `run_train.sh:14` 的
   `pubchemqc_final.lmdb` 不存在；当前主路径 E3FP LMDB 还是一个 1-TB map-size
   但零 entry 的稀疏占位文件，明确禁止用作输入。
3. **旧 tokenizer/checkpoint 不具语义可恢复性。**
   `tokenization/motif_tokenizer.py:79-93` 使用 `set → list(set)`；旧 P1
   checkpoint 不保存 tokenizer snapshot。
4. **P1→P2 加载顺序会丢弃 embedding 继承。** `train2.py:252-277` 先以 P2
   词表构造模型、以 `ignore_mismatched_sizes=True` 加载 P1、之后才 resize。词表
   尺寸不同时，shared/encoder/decoder/LM head 的 P1 权重不应被视作已继承。
5. **几何 target 并非对所有 motif 有效。** 已有 P0 审计记录 353 个 atom mapping
   指向 E3FP padding、1,508 个 motif group 无几何。该情形下 fusion gate 清零，
   被选中时 MSE target 退化为零向量，不能宣称所有 motif 都获得 3D 增强。
6. **当前 launcher 不适用于这台机器。** `run_train2.sh:4,12` 固定 8 GPU，
   当前实例只有 1×RTX 4090 24 GB。

## 词表合约的 R0 结论

新的 sidecar gate 已证明可以离线、跨三个 `PYTHONHASHSEED` 进程稳定构造并审计
完整 token-ID 映射；真实 tokenizer release 尚未满足输入条件。详见
[R0 tokenizer 合约](21_R0_deterministic_tokenizer_contract.md)。

因此，新主线的铁律是：**P1 开始前一次性冻结词表；P2 不扩容；tokenizer snapshot
随 checkpoint 保存；P1→P2 仅允许同一 manifest 下 strict load。** 无法恢复旧
token-ID 映射时，旧 P1 checkpoint 只能作历史参考，不能作为新主线初始化。

## P0 放行前的最小顺序

1. 在额外的新目录实现 manifest-driven 数据适配层；保留原始仓库和旧 launchers
   不动。
2. 冻结 P1/P2 允许成员清单、下游 identity-exclusion 结果、三条 reject ledger 和
   所有输入 source hash。
3. 先决定 explicit-H/H₂、单原子、padding-mapped、无几何 motif 的策略；训练时用
   可审计的 loss mask，而非让它们静默归零。
4. 基于只允许的训练成员生成“频率降序、词典序 tie-break”的 motif 文件，使用
   真实本地 T5 snapshot 通过 tokenizer gate。
5. 在新代码路径验证：tokenizer 长度、embedding resize、`config.vocab_size`、
   batch tensor shape、loss mask 和 strict checkpoint load。
6. 以单卡 profile 做一个固定小样本 smoke；之后才开始新的 P1，再以相同 tokenizer
   进入 P2。

## 关于是否立即切换到 PCQM4Mv2

目前不建议因旧 P1 工程问题而立即替换数据源。3D-MoIT PubChemQC 已提供一个可
追溯的结构预训练基线；PCQM4Mv2 远端 archive 可作为后续 P1 扩展候选，但须先完成
SDF/SMILES identity、坐标—原子对应、下游泄漏排除、motif/E3FP adapter 和同一
manifest 审计。它应进入 R1 的对照实验，而不是绕过本 R0 的阻断条件。

## 主干结论

接受：两阶段“P1 结构建模 → P2 跨模态对齐”的科学叙事，前提是精确限制其目标。

拒绝：用当前 launcher、旧 P1 checkpoint 或未冻结词表启动新的 P1/P2；也拒绝在
353/1,508 个几何例外未处理时讨论 MSE 系数和 motif 划分的性能结论。

