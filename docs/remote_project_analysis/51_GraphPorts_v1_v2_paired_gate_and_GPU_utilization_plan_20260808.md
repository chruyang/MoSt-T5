# GraphPorts v1/v2 配对门与 GPU 利用率优化计划

> 日期：2026-08-08  
> 当前结论：GraphPorts v2 派生 release 与完整 CPU 配对门均已通过；下一次启用 GPU 时只运行 M0-v1/M0-v2 codec gate。数据供给已先行优化，优化器同步优化必须作为独立工程实验，不能与 codec 比较同时改变。

## 1. 三个问题的裁决

### 1.1 train member 不整除并不是模型或库的限制

此前要求 `train_members % micro_batch_size == 0` 是项目自定义 reader/cursor 的人为限制，不是 T5、PyTorch DataLoader 或 LMDB 的要求。当前正式语义已改为标准 `drop_last=False`：

- 每个 epoch 的最后一个 microbatch 可以较短但不得为空；
- 一个 optimizer update 仍消费固定数量的 microbatch，必要时跨 epoch；
- loss 继续按该 update 内全部监督 token 加权；
- 只有 optimizer update 成功后才提交 cursor，prefetch 不得提前移动 checkpoint 边界；
- 不删除尾部成员，也不为整除而改变数据顺序。

在 `N=30,240, microbatch=64, accumulation=2, updates=1,000` 下，总曝光为 127,872 个 member；只有 4 个跨 epoch 更新为 96 members，其余为 128，所有成员均可被访问。静态 `drop_last=True` 虽可维持每步 128，却会让固定尾部 32 条在每个完整 epoch 都被丢弃，因此不采用。

上游代码对照：3D-MolT5 的两个 DataLoader 明确使用 `drop_last=False`；CAMT5 的 pretrain/continual-pretrain/mol2text/finetune 同样使用 `drop_last=False`；FineMolTex pretrain 选择 `shuffle=True, drop_last=True`，属于不同的数据曝光设计，不能机械照搬。PyTorch DataLoader 的公开 API 本身也将 `drop_last` 作为可选策略，而非整除前提。[3D-MolT5](https://github.com/QizhiPei/3D-MolT5)、[CAMT5](https://github.com/Songhyeontae/CAMT5)、[FineMolTex](https://doi.org/10.1145/3711896.3736834)、[PyTorch DataLoader](https://docs.pytorch.org/docs/stable/data.html)。

### 1.2 motif 序列确实常长于 AtomSELFIES，但问题首先在 graph serialization

对 PF-1 run3 的 33,600 条未腐化记录全量统计：

| 指标 | AtomSELFIES | GraphPorts v1 M | GraphPorts v2 M |
|---|---:|---:|---:|
| mean | 23.321 | 49.804 | 31.215 |
| median | 24 | 49 | 28 |
| P95 | 30 | 79 | 55 |
| P99 | 32 | 91 | 70 |
| max | 40 | 137 | 103 |
| M 长于 A 的分子比例 | — | 96.59% | 71.26% |

v1 graph stream 平均 34.981 tokens，占 M 序列约 70.24%；其长度在当前数据域精确为 `4+5E`。因此“motif 本身导致冗长”并不准确，主要冗余来自每条跨 motif 边重复写 edge id、A/B marker 与两个 endpoint。

v2 不改变 motif partition、identity、ports、stereo、connection metadata 或可逆重构，只把边表改为按规范连接顺序排列的两个自定界 endpoint。全量 33,600 条均满足 decode equality、双 owner 和长度公式；graph stream 平均降至 16.393，完整 M 平均降低 37.32%。这说明当前优先动作应是压缩连接语法，而不是立即重划 motif。

但 v2 仍有 71.26% 的分子长于 AtomSELFIES，所以不能宣称“motif 一定更短”。motif 的预期优势是语义单元、局部编辑和结构级遮蔽，不是无条件压缩。CAMT5 通过 fragment/anchor 隐式承载部分连接，FineMolTex 同时保留 atom graph、motif graph 与 atom-to-motif map，Group SELFIES 和 SPE 也说明 group/substructure token 可以有语义价值；这些工作支持研究 motif 粒度，却不证明本项目的线性化必然更短。[CAMT5 paper](https://aclanthology.org/2025.findings-emnlp.1221/)、[Group SELFIES](https://pubs.rsc.org/doi/d3dd00012e)、[SPE](https://pubs.acs.org/doi/10.1021/acs.jcim.0c01127)。

### 1.3 增加 worker 有用，但“吃满 CPU”不是科学目标

旧路径每个 epoch 都重复执行 LMDB get、canonical JSON decode/re-encode、完整 dataclass/语义 gate。分段实测表明瓶颈主要是严格 wire decode，而非磁盘读取。公平的 2,048-record 基准为：

| 模式 | records/s | 相对顺序解码 |
|---|---:|---:|
| sequential | 519 | 1.00× |
| 4 threads | 623 | 1.20× |
| 2 processes | 916 | 1.77× |
| 4 processes | 1,450 | 2.79× |
| 8 processes | 1,647 | 3.17× |

正式 1% screen 已采用 4 个 `spawn` worker 一次严格解码，随后保存在仅当前进程有效的 immutable cache；全 33,600 条 warmup 为 21.82 s，冻结顺序全回放约 0.049 s，约占 0.64 GiB。8 worker 相对 4 worker 只再提高约 14%，所以单条件采用 4 是合理平衡；如果四张卡各跑一个进程，则 4×4=16 workers 正好覆盖 16 vCPU。

这一步可消除数据解码造成的空洞，但不能保证 GPU 曲线恒定。训练主线程仍有独立同步点：每个 microbatch 的 loss `.item()`、gradient norm/clip，以及 AdamWScale 对大量参数逐个执行 RMS `.item()`。下一次 GPU 必须先分段计时，再决定是否改 optimizer；不能在同一 codec gate 中同时替换 loader 和 optimizer，否则无法归因。[PyTorch performance tuning](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)、[pin memory/non-blocking transfer](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)。

## 2. v2 派生 release 的全量事实

最终本地派生 release：`tmp/pf1_run3_local_audit/pf1-paired-release-v2-derived-run3`。

- scheduled / paired：33,600 / 33,600；
- train / dev：30,240 / 3,360；
- reject：0；
- strict source decode / strict target replay：33,600 / 33,600；
- source SDF 再读：否；
- E3FP、AtomSELFIES、motif partition、macro、tokenizer 再拟合：均否；
- 新 token ID：0；
- LMDB map：512 MiB，逻辑使用 328,785,920 bytes；
- v1 graph mean 34.981，v2 graph mean 16.393，降低 53.14%；
- v1 M mean 49.804，v2 M mean 31.215，降低 37.32%。

派生器把已发布且 source-bound 的 v1 wire 作为唯一输入，只重写 motif training document 的 graph suffix、连接 owner indices、相应 binding 和计数，然后再次走权威 paired-wire loader；不是从 SDF 重新生成第二套化学身份。

## 3. 完整跨版本腐化配对门

正式报告：`tmp/pf1_graphports_v1_v2_codec_pair_gate_run3_v2.json`。无 `_v2` 后缀的首版报告已被该补强版本取代，不作为长期证据。

验证域：

- train 30,240 条，每条重放实际会到达的 corruption epoch 0–4；
- dev 3,360 条，重放固定 seed=1 / epoch=0；
- 总计 33,600 records、154,560 corruption views；
- 另从两侧 LMDB raw wire 各独立解码 33,600 个 graph stream；67,200 次 graph decode 均回到同一 208,194 条 `cross_motif_bonds`；
- 每条要求 schedule/source row、Atom record、pair receipt、motif identity、E3FP、atom mapping 相同；
- 每个视图要求 selected motif IDs、mask decision、CE labels、identity carrier 与 geometry mapping 完全相同；
- 只允许 connection input token surface 不同，且 v2 不得更长。

门同时把 `64×2×1,000 updates` 与 30,240 个 train members 推导为 473 microbatches/epoch、实际到达 epoch 0–4；不接受任意 batch 配置套用该结论。报告绑定源/目标 manifest 与当前 v2 codec 版本，合法 connection byte 即使保持长度与 role 不变，只要不再解码到同一 bond table 就会被拒绝。

结果全部通过。train 的腐化后输入由 7,255,705 降至 4,442,620 tokens，减少 38.77%；dev 由 165,287 降至 103,322，减少 37.49%。train 的 983,483 个 target tokens 与 dev 的 23,065 个 target tokens 逐条相等。

这足以授权一个 M0-v1/M0-v2 codec screen；它不授权自动将 v2 升为主线，也不授权跨 AtomSELFIES/Motif 比较 raw CE。

## 4. 唯一下一项 GPU 实验：G-Codec Gate

仅比较：

- `G-v1 = M0 + GraphPorts v1`；
- `G-v2 = M0 + GraphPorts v2`。

共同固定：union-init、train/dev membership、member 顺序、mask、CE targets、optimizer、batch/exposure、BF16、evaluation step 与 1,000-update 预算。不得使用旧 M0 数字作为 v1 对照，因为尾批和输入 cache 合同已经改变；v1/v2 均须由当前 runner 重跑。

预注册报告：step 0/250/500/750/1000 dev NLL 与 accuracy；final 与 750–1000 走势；member/target-token/cursor parity；members/s、nonpadding tokens/member、tokens/s、wall time、peak VRAM、clip rate、gradient norm 和任何失败。

v2 晋级条件全部满足才接受：

1. final NLL 不高于 v1 的 1.02 倍；
2. final accuracy 下降不超过 1 percentage point；
3. 750→1000 无明显反向恶化；
4. mean encoder tokens 不高于 v1 的 70%；
5. members/s 不低于 v1 的 95%；
6. peak GPU memory 不高于 v1 的 105%；
7. 0 data/model/checkpoint failure。

若 NLL 恶化超过 5%、accuracy 下降超过 2 points、在 750/1000 持续劣于 v1，或出现合同失败，则保留 v1。2–5% / 1–2 points 的灰区只允许增加一个配对 seed，不继续堆更多 codec。

资源：一张 RTX 4090 顺序运行两格预计约 35–60 分钟，已足够；两张卡可并行各跑一格，但不是必要条件。四张或八张卡对此门没有科学收益。若并行，必须为每个进程显式设置不同 `CUDA_VISIBLE_DEVICES`；runner 内部始终使用该进程可见的 `cuda:0`。

## 5. GPU 锯齿的后续隔离顺序

1. 先用当前 AdamWScale + validated cache，在 G-v1/G-v2 中记录 data wait、CPU-to-tensor/H2D、forward/backward、clip、optimizer 的 wall time；
2. 若 data wait 已接近 0 而 GPU 仍长时间空闲，再单独做 50–100 update 的 optimizer equivalence canary；
3. candidate 可以是 GPU-resident/foreach RMS 与 step，但必须先证明同输入、同 seed 下参数更新数值闭合；
4. optimizer 变化不得混入 v1/v2 codec 结论；正式模型若采用新 optimizer 实现，比较架构必须从同一初始化重新训练；
5. 扩到约 311 万条时不复制 1% 的全量内存 cache，应改为 map-style Dataset、每 worker 延迟打开只读 LMDB、CPU tensors、bounded prefetch、pin memory 与 non-blocking H2D。

因此当前不需要恢复 GPU。先完成独立审查、提交 v2 release builder/配对门，并把两格运行命令与输出路径在无卡环境中 dry-run；届时只需把 nmb1 切回 1×4090 即可立即开始。
