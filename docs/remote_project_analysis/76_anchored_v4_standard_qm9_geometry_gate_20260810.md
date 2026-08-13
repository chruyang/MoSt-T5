# Anchored V4：标准 QM9 几何属性门

日期：2026-08-10

## 已有证据的合并裁决

Anchored V4 confirmation 与首轮 QM9 HOMO/LUMO/gap 探针共同支持以下结论：

- `l0_l123_mean` 明显优于 `l0_l12_mean`，因此 L3 不能在当前阶段删除；
- carrier 是主要状态入口，endpoint 是有增量价值的 attachment-local 补充入口；
- F3D 已优于无状态 B0，但尚未稳定优于同容量、无坐标的 B2D；
- identity-denoising 与 HOMO/LUMO/gap 都不足以单独证明对应三维构象被使用。

因此下一门不再重复 motif identity CE，也不扩大预训练规模。它只更换监督标签，保留同一 anchored 文本、同一 atom-state 接口、同一初始化和训练预算。

## 标准 QM9 标签与数据绑定

标准 QM9 SDF/CSV 已在本地固定：

- `dataset/qm9-standard-v1/gdb9.sdf`
- `dataset/qm9-standard-v1/gdb9.sdf.csv`

标签为：

| target | 单位 | 主要意义 |
|---|---|---|
| `mu` | D | 偶极矩，对电荷分布与构象方向敏感 |
| `alpha` | bohr^3 | 各向同性极化率，仍受组成与大小强烈影响 |
| `r2` | bohr^2 | 电子空间延展，具有直接的空间尺度含义 |
| `u0` | Hartree | 0 K 内能，强烈受元素组成和分子大小影响 |
| `u0_atom` | kcal/mol | 0 K 原子化能，作为能量类补充指标 |

join 不依赖公开文件的行号猜测，而使用 canonical isomeric SMILES 将冻结的 3D-MolT5 QM9 probe records 与标准 SDF/CSV 关联。最终结果：

- probe records：10,002；
- 唯一成功 join：9,958；
- train/dev/test：7,971 / 992 / 995；
- 统一排除：44，其中包括标准源缺失和 canonical identity 一对多；
- 所有 cell 使用完全相同的 9,958 条记录，不替换、不猜测、不按 cell 单独过滤。

target overlay：`tmp/qm9_standard_target_overlay_v1_r3_b0693bb`。

## 最小实验矩阵

只运行三个 cell：

1. B0：zero state；
2. B2D：stereo-free Morgan，作为坐标无关且参数匹配的原子环境控制；
3. F3D：E3FP，`l0_l123_mean`。

`F3D-l0_l12_mean` 已连续两轮弱于 `l0_l123_mean`，本门不再重复。三格保持：

- 同一 union-init、adapter/head seed；
- batch 256，`drop_last=false`；
- 8 workers、pin memory、prefetch factor 4；
- 30 epochs，约 960 optimizer updates；
- AdamW，LR 3e-4，100 update warmup 后 cosine；
- full encoder fine-tuning；
- train-only mean/std 标准化；
- 每个 target 单独报告原始单位 MAE/RMSE，同时报告 standardized MAE；
- 不对不同单位的 raw MAE 做无意义宏平均。

对 B2D/F3D 还在同一 checkpoint 上报告 both、carrier-only、endpoint-only、zero 四种输入消融。

## 科学门槛

本轮核心不是寻找一个总体最好看的平均数，而是检查一致方向：

1. `mu` 与 `r2` 是优先的几何敏感指标；
2. `alpha/u0/u0_atom` 分别报告，不得掩盖 `mu/r2`；
3. F3D 必须相对 B2D 出现一致、可解释的收益，才能支撑 E3FP 的三维特异性；
4. F3D 仅优于 B0 只说明 atom-state side channel 有用，不足以证明三维增益；
5. endpoint 的价值由同一 F3D checkpoint 的 both 与 carrier-only 差异判断；
6. 若 F3D 仍不优于 B2D，则停止扩大预训练，转而修改 E3FP 组织方式或增加直接 atom-memory readout，而不是继续增加 identity CE 预算。

## 实现状态

- 标准 QM9 overlay builder：PASS；
- overlay 真实全量 join：PASS（9,958 条）；
- 动态五目标 dataset/collator：PASS；
- 真实 cache 全量 CPU replay：40 batches / 9,958 members PASS；
- 定向测试：13/13 PASS；
- 三格 launcher 已就绪：`most_t5_next/p2/launch_qm9_standard_anchored_probe_v1.py`；
- 尚未启动 GPU 训练。

