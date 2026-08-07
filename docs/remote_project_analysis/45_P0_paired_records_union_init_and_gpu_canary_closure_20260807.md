# P0 paired records、统一初始化与真实 GPU canary 收口（2026-08-07）

状态：**P0 已闭合。真实 128 条 A/M paired records、显式 inherited-E3FP、union tokenizer、共享模型初始化，以及单张 RTX 4090 上 A0/A1/M0/M1 的 BF16 前向与反向全部通过。**

本文接续文档 44，并以最终实测结果替代其中“graph+ports、A/M producer 和 GPU canary 尚未完成”的阶段性状态。这里的通过只证明化学语义、数据流和模型 I/O 已闭合，不证明 motif 或 3D 已带来性能增益。

## 1. 最终执行结论

| 环节 | 结果 | 科研边界 |
|---|---:|---|
| inherited-E3FP overlay | 128/128 raw parity；128/128 replay | 显式 duplicate-pointer inheritance；仅为 canary 样本 |
| paired A/M producer | 128 prepared、128 materialized、128 wire replay、0 reject | A/M 来自同一个 projected SDF Mol |
| union tokenizer | 32,499 tokens | 42 个 opaque motif macros；其余 identity 使用 graph+ports fallback |
| CPU 四格 collate | A0=A1 CE、M0=M1 CE、A1/M1 geometry-row parity 全通过 | 不截断、不替换成员 |
| union-init | base tokenizer 32,100 → union 32,499 | 四格使用同一个完整初始化状态的独立副本 |
| RTX 4090 canary | A0/A1/M0/M1 均完成 BF16 forward/backward | 无 optimizer step、无训练权重保存 |

因此，下一阶段不再继续增加 producer 审查，而应转入短程“能否学习”的实验，再进入必要的四格架构比较。

## 2. 官方参考实现带来的裁决

本地保存了三个官方源码的轻量镜像；仅含源码和小型源码资产，不含数据集、模型权重或 checkpoint：

- `reference_repos/3D-MolT5_official_src_82dbe088`；
- `reference_repos/CAMT5_official_src_5875a0a`；
- `reference_repos/FineMolTex_official_src_c976faa`。

### 2.1 3D-MolT5：保留 atom→SELFIES 主干，不照搬弱回退

[3D-MolT5 官方实现](https://github.com/QizhiPei/3D-MolT5)使用 canonical SMILES 的 `_smilesAtomOutputOrder`，再借助 SELFIES decoder attribution 把原子级 E3FP 放到对应 token。这支持本项目 A 分支的基本路径。

本项目保留该简洁主干，但没有照搬 `strict=False` 回退、机械取 `item[-1]` 或失败后生成全 `-1` 的做法。最终 A codec 只使用 strict SELFIES；处理断连符导致的 attribution 压缩索引；要求 carrier 在完整原子域形成双射；并验证 strict-isomeric identity、CIP、键表和 SELFIES 重编码。

### 2.2 CAMT5：stereo 是切分后恢复的结构信息

[CAMT5 官方实现](https://github.com/Songhyeontae/CAMT5)并非简单丢弃 stereo。它在 motif 切分前保存 R/S 和 BondStereo，重连后重新解析支持原子并恢复 E/Z。项目旧的简化版 `model/CAMT5/representation.py` 直接调用 `RemoveStereochemistry()`，不能代表 CAMT5 官方方案。

本项目 graph+ports codec 采用同一原则，但不用长字符串补丁：motif 内部双键保留 stereo 与支持引用；若支持原子是 dummy port，则在重连后解析到另一 motif 中的真实原子，再调用 RDKit stereo 接口恢复。两条真实 C=N 结构各重复重建 100 次，并进行共 52 次 atom-renumber，strict-isomeric identity 全部一致。

CAMT5-derived partition 会合并 ring 和所有 non-single bond 的两端，因此正式跨 motif connection 收窄为 `SINGLE/STEREONONE`。模型 graph grammar 不再重复编码恒定的 bond/stereo token，长度由 `4 + 5E` 表示，固定 GPORTS token 集从 273 缩为 263。这是依据真实 partition 删除无信息字段，不是额外补丁。

### 2.3 FineMolTex：支持 motif 对齐动机，不承担无损 codec 证据

[FineMolTex 正式代码归档](https://doi.org/10.5281/zenodo.15501037)保留完整 atom graph，并额外建立 atom→motif 映射和 motif 标签。这支持 motif-level masking、图文细粒度对齐与 motif editing 的研究动机；但它不是可逆 motif 线性化 codec，不能用来证明 anchor 删除或端口重连无损。因此本项目继续使用完整 graph+ports identity，不把删除 anchor 后的 pure motif 当作无损身份。

## 3. 已冻结的数据流

```text
PCQM SDF Mol
  -> source atom tag + frozen explicit-H projection
  -> one projected RDKit Mol
       -> coordinates[A,3]
       -> raw E3FP replay parity
       -> duplicate-pointer inherited E3FP[A,4]
       -> frozen motif partition and cross-motif SINGLE edges
       -> A surface: strict SELFIES + atom carriers
       -> M surface: port-aware identities + compact connection stream
  -> one union tokenizer
  -> paired A/M records with the same source map and inherited E3FP rows
  -> A0/A1/M0/M1 collators
  -> one shared union-init T5 state
  -> four independently loaded wrappers
```

四格只改变两个声明变量：

- A 与 M：atom SELFIES 粒度或 motif graph+ports 粒度；
- 0 与 1：不注入或注入同一 inherited-E3FP geometry rows。

A0/A1 内部 CE batch 完全相同，M0/M1 内部 CE batch 完全相同；A1/M1 使用同一模型原子域、source mapping 和 E3FP 行，只允许 carrier mapping 不同。

## 4. 真实 paired-128 结果

远端 artifact：

```text
/root/autodl-tmp/most-t5-r1-canary/paired-identity-128-v1-run3
```

| 指标 | 结果 |
|---|---:|
| scheduled / prepared / materialized / replayed | 128 / 128 / 128 / 128 |
| reject | 0 |
| atom 数 A | min 5；median 14；max 18 |
| motif 数 M | min 2；median 7；max 14 |
| cross-motif edges | min 1；median 6；max 13 |
| motif identity occurrences / unique | 939 / 145 |
| macro identities / covered occurrences | 42 / 836 |
| fallback identities / occurrences | 103 / 103 |
| observed SELFIES symbols | 45 |
| A 未腐化输入长度 max | 34 |
| M 未腐化输入长度 max | 112 |
| A/M 超过 512 | 0 / 0 |
| 全遮蔽 target 上界 A/M | 38 / 74 |
| sentinel 最大需求 A/M | 19 / 15（容量 100） |

这 128 条同时覆盖 macro 与 fallback 两条 motif 身份路径。该样本上的宏列表与 32,499 token 规模只是 canary-bound candidate；最终词表 K 仍应从允许进入正式训练的数据域统计，不能用 valid/test 选择。

### 4.1 遮蔽难度不是天然相等

在相同 `p=0.15`、epoch 0、seed 0 下：

- A 分支选择 276 / 1,792 个原子，约 15.40%；
- M 分支选择 189 / 939 个 motif，约 20.13%；
- 这些 motif 覆盖 437 / 1,792 个原子，约 24.39%，并覆盖约 30.36% identity tokens。

因此，A/M 的原始 CE 不能直接作为粒度优劣证据。正式比较需同时报告 selected unit、实际被遮蔽原子比例和 identity-token 比例；必要时用匹配后的 corruption budget 做敏感性实验，而不是增加新的模型架构。

## 5. union-init 与 GPU 实测

统一初始化与 GPU artifact：

```text
/root/autodl-tmp/most-t5-r1-canary/union-init-128-v1-run1
/root/autodl-tmp/most-t5-r1-canary/four-grid-gpu-smoke-v1-run1
```

基础 tokenizer 为 32,100，原始 T5 config vocab 为 32,128，最终 union vocab 为 32,499。初始化从 32,100 起覆盖所有新语义 ID，包括原 checkpoint 中 tokenizer 不可达的 32,100–32,127 行；T5 input embedding 与 untied LM head 分别使用冻结随机流。geometry fusion 使用同一 seed，四格完整 wrapper state 数值相同、存储独立。

GPU 环境为 RTX 4090、PyTorch 2.1.0+cu118、Transformers 4.45.2、BF16，batch size 2。固定使用 frozen membership 的前两条记录，四格均无样本替换、无截断、无 optimizer step。

| 条件 | geometry | input max | target max | CE loss | gradient norm | peak allocated memory |
|---|---:|---:|---:|---:|---:|---:|
| A0 | no | 28 | 10 | 55.015625 | 29,375.58 | 2.259 GB |
| A1 | yes | 28 | 10 | 56.546875 | 37,813.53 | 2.334 GB |
| M0 | no | 19 | 44 | 59.670311 | 212,276.43 | 2.287 GB |
| M1 | yes | 19 | 44 | 62.064064 | 29,005.38 | 2.337 GB |

全部 loss 和 gradients 均为 finite 且 gradient norm 非零。这里的 loss 数字没有性能含义：新增词元和融合层尚未训练，A 与 M 的 target 结构和遮蔽难度也不同。它们只证明真实数据能够贯穿 tokenizer、collator、T5 和 geometry fusion 的前向/反向。

最终在 nmb1 的冻结运行环境中，对本轮 E3FP overlay、atom/graph codecs、union tokenizer、paired producer/wire/builder、union-init、四格 wrapper/training adapter 与 GPU runner 做了定向组合回归：`65 unittest + 48 pytest = 113/113 pass`。该数字只覆盖本轮正式路径，不把无关旧模块混入验收口径。

## 6. P0 已证明与尚未证明的内容

已证明：

- inherited E3FP 能从同一 SDF Mol 重算，并与 frozen raw payload 对照；
- A/M 从同一个 projected Mol 生成，原子行、source mapping 和 3D rows 一致；
- motif 身份能够通过 graph+ports 回环，真实 R/S、E/Z 与端口支持关系可保留；
- union tokenizer 不在训练时扩展；
- A0/A1/M0/M1 能从同一初始化运行真实 BF16 forward/backward。

尚未证明：

- motif 粒度优于 atom 粒度；
- inherited E3FP 能提升生成或下游效果；
- 当前 macro K 是全量数据的最优值；
- 128 条数据能代表正式预训练分布；
- CE 之外的 MSE/teacher 现在有必要。

后一组问题必须由学习曲线、四格比较和下游指标回答，不应再通过增加 producer 检查来回答。

## 7. 下一阶段最短路线

1. 在 paired-128 上为四格各做短程 overfit/learnability run，确认 loss 可下降、checkpoint 可恢复；只比较各条件相对自身初值的学习曲线。
2. 物化一个冻结的 PF-1 小比例训练集，报告实际 corruption budget；只运行 A0/A1/M0/M1 四个必要单元，不加入 C0/C2D-L、legacy MSE 或其他拼盘架构。
3. PF-1 显示稳定趋势后，再把相同四格扩大到 10% 数据用于架构裁决；其后只对胜出方案做完整预训练。
4. MSE/EMA teacher 保留为条件式后续实验：只有 CE-only 四格证明 3D 信号可学习但仍存在明确表示缺口时才加入。

单张 4090 足以完成第 1 步；PF-1 也可单卡顺序运行。进入 10% 四格或三天内并行推进最终预训练时，再切换为 4–8 张 4090，避免现在为数据流工作浪费多卡资源。
