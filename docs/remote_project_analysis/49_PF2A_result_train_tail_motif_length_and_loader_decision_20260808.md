# PF-2A 结果、训练尾批、motif 长度与数据供给裁决

> 日期：2026-08-08  
> 状态：PF-2A 配对实验完成；可变尾批代码门通过；33,600 条长度审计完成；下一 GPU 机制实验冻结为独立 F-Gate  
> 范围：本文件只裁决当前 1% failure screen 的机制与计算路径，不宣称最终预训练架构、最终词表或下游优势

## 1. PF-2A 已完成，结论不是“E3FP 不可行”

PF-2A 在完全配对的 motif 条件下比较：

- `M0-R`：无几何输入；
- `M1-F`：3D-MolT5 风格的一张共享 E3FP 表、固定四 slot 均值、motif 内 atom 均值、carrier 上 `0.5 identity + 0.5 geometry`；
- 两格都使用同一 run3 release、tokenizer、union-init、成员顺序、mask、优化器与 `63×2` 历史冻结协议。

正式结果：

| 指标 | M0-R | M1-F | 配对关系 |
|---|---:|---:|---:|
| step-1000 dev token-weighted NLL | 0.885276 | 3.059478 | 比值 3.45596 |
| step-1000 masked-token accuracy | 0.781747 | 0.479731 | 差值 -0.302016 |
| final shuffled-minus-aligned E3FP ΔNLL | 不适用 | 0.0013676 | 仅保留初始 sensitivity 的 0.7455% |
| clip rate | 0.682 | 1.000 | M1 每步均触发 clip |
| 1,000 updates wall time | 968.81 s | 977.10 s | 同量级 |

三项预注册 gate 全部失败。因此正式裁决是：

```text
不进入 PF-2B / T3MI，也不加入 teacher 或 MSE；
先独立测试 F-Gate，定位固定 0.5 几何注入是否是主要破坏源。
```

这说明该固定融合在当前 CE 任务上既伤害语言恢复，又没有维持 E3FP 敏感性；它不能外推为“motif 3D”或 E3FP 本身无效。配对结果在 `tmp/pf2a_pair_decision_8ca607f.json`，两格完整轨迹在 `tmp/pf2a_{m0,m1}_manifest_8ca607f.json`。

## 2. `train_members % microbatch == 0` 是人为限制，应删除

原 reader 的底层分批本来就会自然返回最后一个短批；真正造成拒绝的是上层额外加入的整除检查和 cursor 的“每批必须等长”断言。LMDB、T5、PyTorch 均不要求训练集必须整除 batch。

采用以下最小语义：

```text
drop_last = False
最后一个 microbatch 可短但不可为空
一个 optimizer update 仍固定消费 grad_acc 个 microbatch
update 可以跨 epoch；每个 microbatch 使用自己的 epoch corruption key
loss 按该 update 的全部 supervised target tokens 精确加权
checkpoint 只提交已完成 optimizer update 后的 cursor
```

以 `N=30,240, microbatch=64, accumulation=2, updates=1,000` 为例：

| 方案 | 总成员曝光 | 科研问题 |
|---|---:|---|
| 可变尾批（采用） | 127,872 | 只有 4 个 epoch 边界 update 为 96 members，其余为 128；无人被排除 |
| 固定顺序 `drop_last=True` | 128,000 | 每个完整 epoch 永久丢最后 32 条 |
| 跨 epoch 拼满一个 microbatch | 128,000 | 同一 microbatch 混合两个 corruption epoch，需重写逐记录 epoch 合同 |
| 历史 PF-2A `63×2` | 126,000 | 正确但少 1.56% nominal 曝光；只保留为已完成实验的历史协议 |

3D-MolT5 与 CAMT5 的正式 DataLoader 都采用 `drop_last=False`；FineMolTex 的主预训练脚本采用 `shuffle=True + drop_last=True`，但它通过 shuffle 改变每轮尾部，和本项目冻结顺序不是同一统计合同。PyTorch 与 Hugging Face 的默认也都是 `drop_last=False`：

- [PyTorch DataLoader](https://docs.pytorch.org/docs/stable/data.html)
- [Hugging Face Trainer arguments](https://huggingface.co/docs/transformers/main_classes/trainer)
- [3D-MolT5 官方仓库](https://github.com/QizhiPei/3D-MolT5)
- [CAMT5 官方仓库](https://github.com/Songhyeontae/CAMT5)
- [FineMolTex 官方仓库](https://github.com/liushiliushi/FineMolTex)

当前实现同时记录 nominal effective batch、短批数、microbatch/update 的成员数最小值、最大值、均值和总曝光。P1+P2 无卡回归为 `135/135 PASS`。这项修改只影响未来运行，不改变 commit `8ca607f` 下已经完成的 PF-2A 证据。

## 3. motif 序列确实更长，但首先是 codec 问题

对 run3 完整 33,600 条、同一 union tokenizer、未腐化且无 padding 的输入做了全量比较：

| 指标 | AtomSELFIES | motif identity + GraphPorts | M−A |
|---|---:|---:|---:|
| 均值 | 23.32 | 49.80 | +26.48 |
| 中位数 | 24 | 49 | +26 |
| P95 | 30 | 79 | +55 |
| P99 | 32 | 91 | +67 |
| 最大值 | 40 | 137 | +106 |

- `M>A`：32,455 / 33,600 = 96.5923%；
- `M=A`：173；
- `M<A`：972；
- 平均 `M/A=2.183`；以长度平方作粗略 attention 上界时，平均比值为 5.332。

当前 33,600 条都低于 512，因此这不是 PF-1 的截断正确性错误；它是扩到 311 万规模前必须处理的吞吐与表示效率问题。

### 3.1 长度来自哪里

当前精确公式对 33,600 / 33,600 条都成立：

\[
L_M=2+L_{identity}+(4+5E),
\]

其中 `E` 是跨 motif 边数。平均组成：

- motif identity：12.82 tokens；
- graph stream：34.98 tokens，占 motif 输入均值约 70.24%；
- 外层分子边界：2 tokens。

每条边重复编码 A marker、edge id、endpoint A、endpoint B、B marker。全量 208,194 条跨 motif bond 都是 single；最大 motif 数 16，最大单 motif port 数 8；全部 33,600 条 contracted motif graph 都是 forest。长度差与 motif/edge 数的 Pearson 相关约 0.693，而与 atom 数仅约 0.024。即使 27,357 条全走 macro，95.81% 仍比 SELFIES 长，说明 raw-byte fallback 只是长尾放大器，不是主因。

完整结果：

- `tmp/motif_length_audit_run3_results.json`
- `tmp/pf1_run3_sequence_length_audit_20260808.json`
- `tmp/audit_pf1_run3_sequence_lengths.py`

### 3.2 不先改变 motif 划分

motif 划分决定语义单元、3D pooling 和编辑位置；graph codec 决定如何传输连接。为了长度而先合并 motif，会同时改变科学粒度和序列化成本，无法判断改善来自哪里。

参考模型也给出同样的职责分离边界：

- CAMT5 用 DFS 与 fragment 内 anchor 隐式表达连接，没有本项目的五 token/edge 外部表，但使用更大的 fragment vocabulary；它证明“连接可以更紧凑”，不证明我们的无损 codec 应直接复制它。[CAMT5, Findings EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.1221/)
- FineMolTex 保留 atom graph、motif graph 与 atom-to-motif map，在结构分支中聚合 motif；它支持 motif 作为逻辑位置，但不是可逆 T5 线性化先例。[FineMolTex, KDD 2025](https://doi.org/10.1145/3711896.3736834)
- 3D-MolT5 将 E3FP 作为与 SELFIES 对齐的 side channel，不增加顺序位置；论文还报告把 E3FP 直接拼接进序列会使训练时间超过 embedding summation 的 1.5 倍。[3D-MolT5, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/3d4dc72d715bd6415d356293079adf3d-Paper-Conference.pdf)
- Group SELFIES 和 SPE 分别支持“语义 group token”与“由数据学习常见分子子串”的方向，但都不能替代本项目的端口、连接和立体化学回环验证。[Group SELFIES](https://pubs.rsc.org/en/content/articlelanding/2023/dd/d3dd00012e)、[SMILES Pair Encoding](https://pubs.acs.org/doi/10.1021/acs.jcim.0c01127)

因此当前 v1 保留为严格可逆基线，但不再称为 sequence-compact。

### 3.3 唯一优先 v2

先只删除冗余 edge id 与 A/B marker；canonical edge order 决定 connection id，交替的两个自定界 endpoint uvarint 决定 A/B。保留 graph begin/end 与 port-radix header，通用 lossless 公式变为：

\[
L_M^{v2}=2+L_{identity}+(4+2E).
\]

在 run3 上估计均值约 31.22，较 49.80 下降约 37.3%，不改变 motif partition、macro/fallback、3D carrier 或目标。另一个仅适用于当前 `motif<16, port<16` 域的 nibble endpoint 版本可到均值约 29.22，但它需要显式 escape 合同；首轮不采用该额外复杂度。

v2 的放行条件不是长度更短，而是：全 33,600 条 encode→decode→strict-isomeric reconstruction、source-bound re-encode、atom-renumber、R/S、E/Z、radical/charge/isotope 全部通过。随后只比较 v1 与 v2 两格，报告 molecules/s、non-padding tokens/s、峰值显存与 CE；不同时改变 motif 划分。

## 4. GPU 锯齿不是只把 worker 数调大即可

当前热路径是：

1. LMDB 顺序取 bytes；
2. 每个 epoch 重做 canonical JSON decode/重编码、完整 dataclass/语义校验和部分哈希；
3. 单个 `ThreadPoolExecutor(max_workers=1)` 做 decode+collate，depth=2 只是一个生产线程预取两个 update；
4. 主线程从 Python tuple 直接构造 CUDA tensor，没有 pinned CPU tensor/non-blocking H2D；
5. dev eval 完全在主线程同步执行。

既有实测显示：30,240 条完整 validated decode 约 94.35 s，即 PF-2 每 126 条 update 约有 0.393 s decode；depth 0→2 的 50-update canary 只由 43.72 s 降到 41.35 s（1.057×）。无卡 0.5-vCPU 的分段探针也显示 raw LMDB get 几乎可忽略，主要时间在 JSON 与完整重验，不是磁盘。

此外 GPU 锯齿还有独立同步源：每 microbatch 的 loss `.item()`、gradient clip，以及 AdamWScale 对每个参数执行 RMS `.item()`，都会产生 GPU→CPU 同步。3D-MolT5/CAMT5 的参考 optimizer 也有类似实现；“有上游先例”不等于它的吞吐最优。

### 4.1 下一轮只做可归因优化

先做 50–100 updates 的分段 profiler，再按以下顺序：

1. **当前 1% screen：**一次验证后缓存 33,600 条 immutable record；epoch mask 与 collate 继续在线生成。
2. **可扩展路径：**map-style dataset，先试 `num_workers=4, prefetch_factor=2, persistent_workers=True`；每 worker 独立延迟打开只读 LMDB。
3. worker 产出 CPU tensor；正确 pin memory 后由主线程 `non_blocking=True` 搬运。
4. 固定 2/4/8 workers 做短 canary，到 queue-wait 平台即停止；不以“16 vCPU 就必须 16 workers”为目标。
5. 若数据等待显著下降后仍锯齿，再单独测试 GPU-resident AdamWScale；不能把 optimizer 数值变化混入 loader 对比。

PyTorch 官方明确区分多进程 DataLoader、prefetch、persistent workers 和 pinned memory；Python 密集处理不能靠提高 OMP 线程自动并行：[PyTorch DataLoader](https://docs.pytorch.org/docs/stable/data.html)、[performance tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)、[pinned memory/non-blocking guide](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)。3D-MolT5 先用 Dataset `map(num_proc=8)` 预处理再以 8 workers/pin memory 训练；CAMT5 常用 8 workers、prefetch 3 或 5。这些是合理起点，不是本项目应机械照搬的最优值。

必须保持的公平性门：record IDs、mask arrays、CPU tensors、主进程 torch RNG、成功 update 后 cursor、参数轨迹与同步实现逐项相同。预取到但尚未完成 optimizer update 的数据不得进入 checkpoint cursor。

## 5. 后续顺序与资源

### 当前无卡阶段

1. 提交可变尾批修复与本文件；
2. 独立实现并全量验证 GraphPorts endpoint-pair v2；
3. 准备 immutable-cache / 4-worker loader canary 与分段 profiler；
4. 实现 F-Gate，但不把 codec v2、loader 或 optimizer 改动混进同一科学对比。

### 下一次启用 GPU

1. 先跑 50–100 update loader/profiler canary，选择 0/2/4/8 workers 中达到平台的最小值；
2. 在当前 v1 codec 上跑 fresh `M0-G/M1-G` 独立 gate pair；
3. 只有 F-Gate 满足 non-degradation 且仍缺 geometry sensitivity，才重新考虑 PF-2B/T3MI；
4. motif codec v2 的效果单独比较，不与 F-Gate 合并。

下一轮 1×4090 足够完成 loader canary 与顺序 F-Gate；若单格 runner 和合并合同准备完毕，2×4090 可一格一卡缩短墙钟。当前不需要 4/8 卡。
