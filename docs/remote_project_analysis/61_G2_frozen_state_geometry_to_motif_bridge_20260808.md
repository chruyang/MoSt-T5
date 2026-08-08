# G2：冻结 G1b 状态编码器的 geometry-to-motif CE 桥接

> 状态更新：G2 已完成；实际 run4 因文件存储配额改在 `autodl-tmp` 保存最终 checkpoint。结果与裁决见 `62_G2_result_geometry_channel_not_state_specific_20260808.md`。下方启动命令保留为预注册执行记录，不再用于重跑。

## 1. 本阶段回答的问题

G1 已证明 motif 内的 E3FP 状态可被 Deep Sets 编码器学习，且对同尺寸错配几何和同一 2D 身份下的构象变化敏感。G2 不再继续比较集合编码器，而是回答下一层问题：

> 在 motif 身份全部不可见、GraphPorts 拓扑仍可见时，冻结的 G1b 几何状态能否给 T5 的 motif 身份恢复带来稳定的增量？

这不是下游性能实验，也不是“纯 3D”实验；它是在固定拓扑条件下检查几何状态能否进入语言模型并被 CE 使用。

## 2. 配对实验

- **G2-C（代码条件 M0）**：全部 motif identity span 替换为 T5 sentinel；仅保留 GraphPorts connection skeleton 和 motif 顺序。
- **G2-G（代码条件 M1）**：输入、标签、成员顺序、优化器和初始化与 G2-C 完全相同；额外在每个 motif sentinel carrier 注入冻结 G1b Deep Sets 表征。

两格均使用标准 T5 sentinel CE，不增加 MSE、teacher、预测头或新词表。

几何支路为：

1. 复用 G1b 的 shared E3FP embedding、level embedding、core/attachment role、atom `phi`、Deep Sets mean pooling 和 motif `rho`；
2. G1b 全部参数冻结；
3. 仅训练 `LayerNorm(128) + Linear(128, d_model)`；
4. 投影结果只加到已有 motif carrier，非 carrier token 不变。

## 3. 数据与优化合同

- 数据：PF-1 run3，train 30,240 / dev 3,360；
- corruption：`mask_probability=1.0`，每条记录的全部 motif identity 被遮蔽；
- protocol：micro-batch 64，gradient accumulation 2，nominal effective batch 128；短尾批沿既有 `drop_last=False` 语义处理；
- updates：1,000；step 0/250/500/750/1000 固定 dev；
- checkpoint：step 500 保留评估 marker，step 1,000 保存完整恢复状态；G2 是分钟级机制筛选，不重复保存两份约 3GB 的同格权重；
- 输入流水：每进程 12-worker 一次性 validated-record cache warm-up，随后 ordered prefetch depth 2；
- 两张 GPU 时每卡一格，必须分别设置 `CUDA_VISIBLE_DEVICES=0` 和 `1`。

## 4. 预注册判定

仅在两格完成后由合并器判定：

- `G2-G final NLL <= 0.98 * G2-C final NLL`；
- `G2-G accuracy >= G2-C accuracy - 0.01`；
- G2-G 完整 dev 的 same-size shuffled-minus-aligned `delta NLL >= 0.01`。

三项同时通过，才接受该桥接进入正式第一阶段预训练；否则先修改几何桥接，不直接用扩大数据量掩盖机制失败。

## 5. 当前无卡验证结果

- 新 G2 定向测试：7/7 PASS；
- 相关 production/wire/paired/adapter/wrapper 回归：50 项通过；
- 真实 run3 dev 首批 8 条回放：token shape `(8, 67)`，E3FP shape `(8, 17, 4)`，72 个 attachment 标记进入冻结编码器，62 个真实 motif carrier 获得有限增量；
- G1b checkpoint 使用快盘可读副本：`/root/autodl-tmp/g1b-motif-state-l12-deep-v3-20260808/final_state.pt`；
- nmb1 当前为无卡 0.5-vCPU 模式，因此未启动 T5-base GPU forward。

## 6. 两卡启动命令（开卡后直接执行）

公共路径：

```bash
CODE=/root/autodl-tmp/MoSt-T5-g1-20260808
PY=/root/miniconda3/envs/3dmolt5/bin/python
RELEASE=/root/autodl-tmp/most-t5-r1-pf1-run3-e07cc8e/pf1-paired-release-v1-run3
BASE=/autodl-fs/data/most-t5-r1/base-models/google-t5-v1_1-base/b5fc947a416ea3cb079532cb3c2bbadeb7f800fc
INIT=/root/autodl-tmp/most-t5-r1-pf1-run3-e07cc8e/pf1-union-init-v1-run3
G1=/root/autodl-tmp/g1b-motif-state-l12-deep-v3-20260808/final_state.pt
OUT=/autodl-fs/data/most-t5-r1/g2-frozen-g1b-20260808
mkdir -p "$OUT"
```

G2-C：

```bash
cd "$CODE"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
nohup "$PY" -m most_t5_next.p2.run_g2_geometry_to_motif_ce_v1 \
  --paired-release "$RELEASE" --base-model-snapshot "$BASE" \
  --base-tokenizer-snapshot "$BASE" --union-init-dir "$INIT" \
  --g1-checkpoint "$G1" --output-dir "$OUT/G2-C" \
  --geometry-fusion-seed 20260808 --condition-id M0 \
  > "$OUT/G2-C.log" 2>&1 < /dev/null &
```

G2-G：

```bash
cd "$CODE"
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
nohup "$PY" -m most_t5_next.p2.run_g2_geometry_to_motif_ce_v1 \
  --paired-release "$RELEASE" --base-model-snapshot "$BASE" \
  --base-tokenizer-snapshot "$BASE" --union-init-dir "$INIT" \
  --g1-checkpoint "$G1" --output-dir "$OUT/G2-G" \
  --geometry-fusion-seed 20260808 --condition-id M1 \
  > "$OUT/G2-G.log" 2>&1 < /dev/null &
```

两格完成后：

```bash
"$PY" -m most_t5_next.p2.merge_g2_geometry_to_motif_ce_v1 \
  --control-run "$OUT/G2-C" --geometry-run "$OUT/G2-G" \
  --output "$OUT/g2_paired_decision.json"
```

## 7. 下一决策

- G2 通过：进入同一 2D 身份多构象的 T5 级验证，随后冻结第一阶段正式架构；
- G2 未通过但 shuffle 敏感：优先调整投影/注入位置；
- G2 未通过且 shuffle 不敏感：冻结编码器的状态没有被 T5 使用，先停止扩展预训练规模。
