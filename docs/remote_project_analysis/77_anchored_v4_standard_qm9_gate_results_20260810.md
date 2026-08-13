# Anchored V4：标准 QM9 几何属性门结果

日期：2026-08-10

## 1. 执行状态

三格均完成，launcher `status=pass`，无 OOM、非有限 loss 或数据 reject：

1. B0：zero state；
2. B2D：stereo-free Morgan，`l0_l12_mean`；
3. F3D：E3FP，`l0_l123_mean`。

共同数据为标准 QM9 overlay 唯一关联成功的 9,958 条记录：train/dev/test =
7,971/992/995。每格 30 epochs、960 optimizer updates、batch 256、LR 3e-4、8 workers。
单格 wall 约 185--198 秒，三格含初始化和写盘总计约 10 分钟。

远端正式产物：

`/autodl-fs/data/most-t5-r1/qm9-standard-anchored-probe-v1-b0693bb`

## 2. 主结果

不同 target 单位不同，不能宏平均原始 MAE。以下首先报告以 train-only std 归一化后的
五目标宏平均 MAE，越低越好：

| split | B0 | B2D | F3D-123 | F3D 相对 B2D |
|---|---:|---:|---:|---:|
| dev | 0.40507 | 0.38164 | **0.36596** | **-4.11%** |
| test | 0.41129 | 0.38271 | **0.35781** | **-6.51%** |

这是当前项目第一次在相同 anchored 文本、相同注入位置和参数容量下，F3D 在 dev 与 test
总体上都优于坐标无关 B2D，而不只是优于无状态 B0。

## 3. Test 的逐目标结果

| target | 单位 | B0 | B2D | F3D-123 | F3D 相对 B2D |
|---|---|---:|---:|---:|---:|
| `mu` | D | 0.86961 | 0.83654 | **0.79655** | **-4.78%** |
| `alpha` | bohr^3 | 2.95321 | **2.80092** | 2.89075 | +3.21% |
| `r2` | bohr^2 | 104.4560 | 103.7936 | **96.4449** | **-7.08%** |
| `u0` | Hartree | 16.7093 | 13.4433 | **10.1888** | **-24.21%** |
| `u0_atom` | kcal/mol | 83.2045 | 78.7278 | **78.2934** | **-0.55%** |

dev 中方向稳定的目标为：

- `r2`：F3D 98.2750，B2D 103.7584；
- `u0`：F3D 10.0975，B2D 13.9257。

`mu`、`alpha` 和 `u0_atom` 的 dev/test 方向并不全部一致：F3D 在 test 的 `mu` 更好，
但 dev 略差；`alpha` 在两者均略差；`u0_atom` 只在 test 略好。因此不能把五项都表述为
稳定三维增益，也不应进行结果后单项挑选。

## 4. Carrier 与 anchor-endpoint

F3D test standardized macro MAE：

| 输入组件 | standardized MAE |
|---|---:|
| both | **0.35781** |
| endpoint only | 0.57843 |
| carrier only | 0.80791 |
| zero | 1.18817 |

同一 checkpoint 下，both 明显优于任一单通道，说明 anchor occurrence 作为 attachment-local
endpoint 不是死路径；它与 motif carrier 形成互补地址。该消融是训练后删除输入的分布外
诊断，不能替代独立训练的 B0/B2D/F3D 主比较，但足以支持首版保留 carrier+endpoint。

## 5. 科学裁决

本门为**有边界的通过**：

- 通过项：F3D 在 dev/test standardized aggregate 均优于 B2D；预注册、空间含义较直接的
  `r2` 在 dev/test 均稳定改善；`u0` 也一致改善；carrier+endpoint 联合路由确实被消费；
- 未通过的强声明：并非所有目标都由 F3D 改善；只有单一 seed；QM9 每个分子只有一个构象，
  仍不能证明同一身份下的构象辨别；E3FP 含二维身份，因此不能把所有收益都归因于纯几何。

因此允许冻结当前首版架构并进入下一阶段：

- stereo-free anchored pure motif：保留；
- bounded chemical lexer fallback：保留；
- motif carrier + anchor endpoint：保留；
- B2D：继续作为正式训练和下游评估的必要对照；
- F3D shell：冻结 `l0_l123_mean`；
- 不再为 `l0_l12_mean`、identity-denoising 或相同 QM9 小门增加预算；
- 不引入坐标 GNN/SE(3) 模块；
- 可以开始 Phase-I/Phase-II 的正式数据、词表、mmap cache 与训练协议冻结。

### 2026-08-11 修订

这里的“冻结 `l0_l123_mean`”现解释为**冻结为历史最佳 V4 研究基线**，而不是规定正式
预训练必须保留 V4 的 level/role/presence/MLP atom encoder。后续复核确认3D-MolT5已对
缺失 L3 使用固定零embedding并在四个槽上直接平均。正式候选因此先采用同一原子层语义，
只保留本项目的 motif carrier 与 anchor endpoint 双路组织；V4结果不作废，也不重复训练。
详见文档78。

这不是论文最终效果结论。正式论文仍需在更完整的下游矩阵中报告 B0/B2D/F3D，并优先
复核 `mu/R2`、PubChemQC 属性和 3D caption；如可获得可信同身份多构象数据，再增加构象
对应敏感性，而不使用 RDKit 生成构象冒充真值。

## 6. 资源事实

- GPU 峰值 allocated 约 12.0 GiB，reserved 约 21.3 GiB；
- 单卡 4090 可稳定运行 batch 256；
- 8 workers 下 GPU 训练阶段通常为 80--97% 利用率；
- 当前无需为该规模增加 GPU，后续多卡只应用于彼此独立的 cell 或正式大语料并行。
