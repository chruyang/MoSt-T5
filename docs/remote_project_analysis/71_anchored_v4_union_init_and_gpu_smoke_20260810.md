# Anchored V4 union-init 与 GPU smoke（2026-08-10）

## 已闭合的初始化边界

Stage 3 的共享 tokenizer snapshot 同时兼容两种 macro 语义计划，因此 snapshot tree SHA 只能证明 token surface，不能独立证明 token 对应的 motif identity。`load_verified_anchored_candidate_tokenizer` 现在要求调用方显式提供 `semantic_plan_sha256`，并同时验证：

- 候选 manifest 状态；
- semantic plan 位于 snapshot 声明的兼容计划集合且唯一；
- tokenizer 文件树及 tree SHA；
- base vocabulary ID 不变；
- 941 个 added token 构成连续区间；
- added token 均为 ordinary token；
- T5 sentinel、pad、EOS、UNK 以及自然文本回归不变。

本轮主线选择：

- semantic plan SHA-256：`aaa00a6a3d5f498085499f6123ccfa3047155800e0a025a3b85fcdf524fa9759`；
- tokenizer snapshot tree SHA-256：`1cc2e8c6ca3987d15d3558648e535f1e7157f4192cb4e2511ba2ad8ac47bc6a0`；
- base/final vocabulary：32,100 / 33,041；
- macro policy：`pretrain_train_only`。

`build_anchored_union_init_checkpoint_v1.py` 复用既有严格 raw-T5 初始化规则：精确 resize 到 33,041，并从 base tokenizer 长度 32,100 开始重新初始化全部新语义行，包括原 checkpoint 中 tokenizer 不可达的 32,100--32,127。input embedding 与 untied LM head 使用独立私有 RNG 流；非 vocabulary 参数逐值保持不变。远端发布目录：

`/root/autodl-tmp/anchored-union-init-v1-b0693bb`

## V4 初始化修正

旧构造链曾先实例化 V1/V2 adapter 再覆盖成 V4，虽然前向功能正确，但会额外消耗私有 RNG。现在 `FactorizedMotifT5V1._bind_t5_boundary` 将无随机的 T5 合同验证与 adapter 构造分开；V4 只实例化一次 V4 adapter。初始化合同显式绑定：

- anchored tokenizer semantic plan；
- shell fusion mode；
- adapter seed；
- E3FP domain 与各隐藏维度；
- L0 仅作为 atom-level 2D identity/context，不作为 3D 证据；
- higher shells 才承载局部环境状态的候选语义。

远端 V1--V4 factorized model/init 联合回归 20/20 通过。

## 真实单卡 smoke

数据来自完整 33,600-record anchored mmap cache；四种候选严格使用相同的前 128 条 train records、相同动态 `m_plus_g` view、774 target tokens 和 1,680 anchor endpoint tokens。每种模式均完成 BF16 forward/backward，不做 optimizer step，不保存训练权重。

### 16 x 8

| shell mode | wall s | peak reserved GiB | adapter grad | T5 grad |
|---|---:|---:|---:|---:|
| `l12_mean` | 1.158 | 4.09 | nonzero | nonzero |
| `l0_l12_mean` | 0.805 | 4.60 | nonzero | nonzero |
| `l0123_mean` | 0.839 | 4.77 | nonzero | nonzero |
| `l0_shell_attention_l123` | 0.805 | 4.75 | nonzero | nonzero |

### 64 x 2

| shell mode | wall s | peak reserved GiB | shell-attention grad |
|---|---:|---:|---|
| `l12_mean` | 0.702 | 8.98 | inactive as designed |
| `l0_l12_mean` | 0.344 | 10.33 | inactive as designed |
| `l0123_mean` | 0.339 | 10.36 | inactive as designed |
| `l0_shell_attention_l123` | 0.359 | 10.35 | nonzero |

64 x 2 在同一 128-member exposure 下明显减少 Python/launch 开销，且给 24 GB RTX 4090 留有足够空间容纳 optimizer state。下一轮短程训练默认采用 64 x 2、DataLoader workers=0；只有 profiler 证明主进程数据等待时才升到 2 workers。全量 cache 的 CPU 基准已经证明在当前短序列上 0 workers 快于 2/4 workers，因此不能以 CPU 占用率本身作为增加 workers 的理由。

## 解释边界

本次 loss 来自未训练的新增 vocabulary/adapter，并受 microbatch dropout 分组影响，不能横向选择 shell mode。该 smoke 只证明：

- anchored phrase-end carrier 与 attachment endpoint 均进入真实 T5；
- 四层 E3FP 候选的数据路径、梯度路径和 checkpoint 合同可执行；
- 64 x 2 是当前单卡更合适的资源配置；
- 当前 GPU 热路径不是数据 cache 瓶颈。

shell mode 的科学选择仍需同初始化、同 exposure 的短程训练，以及 aligned/zero/matched-shuffle 和 B2D 对照；不得从本 smoke 的初始 CE 数值判断优劣或宣称 3D 增益。
