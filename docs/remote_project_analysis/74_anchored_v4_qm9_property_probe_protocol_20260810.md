# Anchored V4：QM9 三维敏感属性门

日期：2026-08-10

## 为什么更换评价目标

PF-1 和 anchored V4 confirmation 的目标是 motif identity denoising。该目标允许模型依赖
motif 文本与二维拓扑捷径，因此 matched E3FP 不敏感不能单独否定 E3FP，也不能据此选择
`L0+L1/L2` 或 `L0+L1/L2/L3`。下一门改为分子性质回归，使几何输入至少有机会影响与电子
结构相关的输出。

本门不是最终 QM9 benchmark，而是架构机制筛查。首轮严格复用 3D-MolT5 已发布 QM9
instruction 数据中的 HOMO、LUMO 和 gap，单位保持 Hartree；暂不混入另一个 QM9 数据源
中的偶极矩、极化率或能量标签。

## 数据与切分

- 来源：3D-MolT5 QM9 training parquet；其 published validation shard 因分子重叠不参与冻结。
- split unit：exact SMILES molecular identity。
- 固定 seed：`20260810`。
- 请求分子：train/dev/test = 8000/1000/1000。
- 冻结 membership：10,004 个 exact state；公开 instruction paraphrase 不作为独立样本。
- 属性缺失不是 reject：同一三输出 head 使用显式 target mask。
- 两个 state 的官方 `molecule_fp` 全为 `-1`，从所有 cell 共同剔除；最终 10,002 records：
  train/dev/test = 8003/1000/999。

切分和标签冻结器为
`most_t5_next/p2/freeze_qm9_3dmolt5_probe_subset_v1.py`，训练缓存构建器为
`most_t5_next/p2/build_qm9_anchored_probe_cache_v1.py`。

## 原子轴与语言表面

parquet 的 `molecule_fp` 位于 SELFIES token 轴，结构符号和显式立体辅助 H 的行均为 `-1`。
缓存构建执行：

1. 严格选取 L0 非负的 atom-producing E3FP 行；
2. 在 anchored identity 侧移除 stereochemistry，再删除显式 H；
3. 要求 heavy-atom 数量与 E3FP atom rows 完全相等；
4. 使用冻结的 ring/non-single motif partition；
5. 使用 stereo-free pure motif + ordered anchors；
6. 按排序后的 edge ID 重编号模型 anchor，与正式 persisted-pair 路径一致；
7. 使用冻结 macro registry，未注册 motif 走可逆 chemical lexer + 单 fallback suffix；
8. 不向模型暴露 GraphPorts token。

E3FP 负责可能的立体/构象状态；motif 文本不重复暴露 stereochemistry。

## 四个 cell

| cell | atom state | shell mode | 目的 |
|---|---|---|---|
| B0 | zero | 参数拓扑与 L0+L1/L2 cell 对齐 | 无状态基线 |
| B2D | stereo-free Morgan radius 0..3 | `l0_l12_mean` | 坐标无关、参数匹配二维控制 |
| F3D-12 | E3FP | `l0_l12_mean` | L0 identity + L1/L2 local state |
| F3D-123 | E3FP | `l0_l123_mean` | 检查稀疏 L3 是否有额外价值 |

本子集中非负 E3FP 槽计数为 L0/L1/L2/L3 =
87,931/87,931/87,914/1,696。L3 仅覆盖约 1.9% atom rows，因此 F3D-123 若不优于
F3D-12，首先解释为当前数据的 L3 支持稀疏，不解释为一般性的 L3 无效。

## 模型和训练

- 同一个 anchored T5-v1.1-base union-init；同一个 adapter/head seed。
- encoder 输入为未 corruption 的 anchored motif sequence。
- 回归表征为 T5 encoder hidden 的 attention-mask mean pooling。
- head：`Linear(H,H) -> LayerNorm -> GELU -> Dropout(0.1) -> Linear(H,3)`。
- train-only mean/std 标准化；loss 为所有可用 target entries 的均值 MSE。
- 报告每项原始 Hartree MAE/RMSE，不使用 numeric text generation 作为本门混杂因素。
- full encoder fine-tuning；AdamW，LR `3e-4`，warmup 100 updates + cosine，weight decay 0。
- batch 256，drop_last=false，8 workers，pin memory，prefetch factor 4。
- 30 epochs；约 960 optimizer updates；不 early stop，不按 cell 单独调参。

3D-MolT5 的 QM9 脚本同样使用 batch 256、LR 3e-4、100 epochs 和 8 workers；本门缩短
为 30 epochs 是机制筛查预算，不冒充官方最终任务复现。

## 因果消融与裁决

对 B2D/F3D 在 dev 和 test 分别评价：

- aligned both；
- carrier only；
- endpoint only；
- zero state。

因此 endpoint 是否对性质任务有贡献可以直接由同一 checkpoint 的 CE-free property ablation
判断，不再用 matching head 的路由可学性替代 decoder/下游因果证据。

进入下一阶段至少要求：

1. F3D aligned 稳定优于 B0/zero；
2. F3D 相对 B2D 有一致的性质增益，或在后续更强 3D-sensitive QM9 targets 上取得增益；
3. carrier/endpoint 消融方向可解释且不是单纯参数增加；
4. dev 与 test 方向一致。

HOMO/LUMO/gap 仍可被二维结构较强预测，因此本门失败不直接否定 E3FP；通过也不构成完整
3D 表征证明。若三格接近，下一步应扩展到标准 QM9 的 `mu/alpha/R2/U0` 等更明确的
几何敏感标签，而不是回到 motif identity 重建任务反复试验。

## 执行状态

- CPU cache：PASS，10,002 records，2 rejects（全空官方 E3FP）。
- BF16 smoke：batch 64 和 batch 256 四格均 forward/backward PASS。
- batch 256 峰值 reserved 约 10.24 GiB（尚未含 Adam states）；正式训练观测约 14.7 GiB。
- 正式四格输出：`/autodl-fs/data/most-t5-r1/qm9-anchored-probe-v1-r2-b0693bb`。

