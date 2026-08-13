# 最终 PF1 atom encoder 与 level embedding 单因素筛选

日期：2026-08-11

## 裁决

Anchored V4 是历史复杂度上界，不直接冻结为正式 atom encoder。正式进入 PF10 前只做
最后一次 PF1（33,600 members，约1%）淘汰筛选，随后停止新增 reducer 变体。

本轮仅比较三种已有实现：

| candidate | 实现 | 唯一结构差异 |
|---|---|---|
| `reference_fixed_four_mean` | V5 | 共享E3FP表、缺失shell为零、L0--L3固定四槽均值 |
| `l0_high_minimal_phi` | V8 | L0单独、可用L1--L3均值、最小两层phi |
| `l0_high_level_aware_phi` | V9 | 与V8完全相同，只在每个shell embedding上增加level embedding |

三者均不把 `atom_is_attachment` 作为学习特征，不显式加入presence bits；attachment只用于
ordered-anchor endpoint路由和验证。carrier/endpoint、tokenizer、cache、T5初始化、
动态corruption、优化器及数据顺序完全相同。

## 为什么level embedding仍需单独裁决

E3FP原始shell identifier的哈希header已包含迭代level，因此level embedding不增加新的
化学或三维观测。它可能有价值的原因是4096域folding会产生跨level碰撞，随后shell均值
也可能削弱层级可辨识性。故其正式解释仅限于：补偿离散folding/聚合后的level context。

历史V9在标准QM9上改善B2D但没有令F3D优于配对B2D，尚不能证明level embedding增强了
3D状态使用。本轮V8/V9严格单因素配对用于给出最终裁决。

## 六格矩阵

三个candidate各运行B2D与F3D，共六格。PF1只承担淘汰：

- 明显不稳定或难以优化的候选淘汰；
- F3D持续劣于同结构B2D的候选不得因绝对CE较低晋级；
- aligned相对matched-donor/zero没有敏感性的候选不能声称使用对应3D状态；
- 简单候选与复杂候选差异落在筛选噪声内时保留简单候选。

PF1不授权最终架构结论。仅两个候选进入PF10；PF10预训练、冻结QM9迁移和扰动门共同决定
正式架构，完整语料只训练唯一胜者。

## 实现与运行边界

- 单格入口：`most_t5_next.p2.run_anchored_atom_encoder_screen_v1`
- 六格入口：`most_t5_next.p2.launch_anchored_atom_encoder_screen_v1`
- 单格manifest：`atom_encoder_screen_manifest.json`
- 总状态：`launcher_status.json`
- updates：沿用PF1冻结协议1,000 updates、64×2；
- 默认DataLoader workers=0、prefetch factor=4。完整PF1 mmap cache实测0 workers快于2/4；
  worker参数仍可覆盖，但不得仅为提高CPU占用而更改。
- matched overlay应在存在时传入，最终评估同时报告aligned、carrier-only、endpoint-only、
  zero及matched donor。

定向与相邻回归共21项通过；V5/V8/V9初始化合同另4项通过。

## PF1 实测结果

远端产物：
`/root/autodl-tmp/anchored-atom-encoder-screen-v1-b0693bb`。六格全部 PASS，
总 wall 约34.7分钟，无OOM或traceback；各格均为同一PF1数据、1,000 updates、64x2、
同一T5/adapter seed和同一0.5几何注入比例。

下表报告update 1000的完整dev结果。`zero delta = NLL(zero)-NLL(aligned)`；
`matched delta = NLL(same-identity donor)-NLL(aligned)`，正值才表示破坏对应状态后变差。

| candidate | cell | aligned NLL | accuracy | zero delta | matched delta |
|---|---:|---:|---:|---:|---:|
| fixed four mean | B2D | 1.184280 | 0.709801 | +0.005192 | +0.000271 |
| fixed four mean | F3D | 1.057848 | 0.742699 | -0.024033 | -0.000701 |
| L0/high minimal phi | B2D | 1.060397 | 0.749681 | +0.087781 | -0.000878 |
| L0/high minimal phi | F3D | 0.812438 | 0.796131 | +0.224587 | -0.000042 |
| L0/high level-aware phi | B2D | 0.502954 | 0.867266 | +1.444167 | +0.000526 |
| L0/high level-aware phi | F3D | 0.683133 | 0.824284 | +1.382391 | -0.000009 |

## PF1 裁决

1. `reference_fixed_four_mean` 淘汰。它在F3D中置零反而改善0.024 NLL，且matched donor
   也没有带来惩罚；因此本轮没有证据表明固定四槽均值在使用对应三维状态。
2. `l0_high_minimal_phi` 晋级PF10。相对固定均值，F3D NLL下降0.245，zero消融恶化
   0.225 NLL；L0与高层状态分离是有价值的结构候选。
3. `l0_high_level_aware_phi` 晋级PF10，但不在PF1冻结为胜者。相对无level版本，
   F3D NLL再下降0.129，zero消融惩罚由0.225增至1.382；level context显著提高通道
   可用性。可是matched donor差值仍约为零，尚未证明它提高对应构象状态敏感性。

因此PF10只比较V8与V9，不再新增atom reducer。PF10需沿用相同B2D/F3D配对并保留
aligned/zero/matched-donor门，同时做冻结QM9迁移；只有在更大样本和独立3D敏感任务上
仍成立时，才允许把level embedding写入正式架构。PF1结果只能支持“L0/high分流与level
context值得扩大验证”，不能支持“模型已学习构象特异3D状态”。

## 正式预训练前仍待裁决：motif词表与E3FP参数

以下参数不得由PF1候选或历史实现默认带入正式预训练：

### Motif词表

- 512只是PF1 occurrence覆盖率与序列长度曲线上的candidate，不是文献规定值；
- 保留“高频whole-motif macro + 无损chemical lexer fallback”的开放词表方法；
- 在完整Phase-I train corpus上比较至少512/2,048/4,096/8,192；
- 同时报告occurrence coverage与type coverage，不能用高频occurrence覆盖替代化学语义广度；
- 报告seen/unseen motif、输入/目标长度、每个macro训练频次、词表参数/softmax成本、吞吐，
  以及ChEBI-20等文本语义任务的motif级结果；
- 最终预算、registry和token IDs必须在第一次optimizer update前一次冻结，下游不得扩词。

### Motif结构等价表面与参数绑定

当前每个已注册macro在lexer域都存在一个可逆展开，因此同一pure-motif identity原则上有两种
等价表面：一个whole-motif macro，或`chemical_tokens+ + fallback_suffix`。canonical encoder
目前对已注册identity总是选择macro，训练数据不会主动混用两种表面；但decoder/生成模型仍
可能产生非canonical alias。正式预训练前须完成以下裁决：

- 对完整corpus做graph/identity等价census，排除不同SMILES遍历、环编号、分支顺序等造成的
  重复macro类型；
- 明确生成与评估是严格canonical surface，还是允许等价surface解码后统一canonicalize；
- 比较仅保留唯一canonical surface、macro由完整fallback phrase初始化、macro与完整fallback
  phrase表示对齐，以及受限hard tying；
- 报告alias是否分散autoregressive输出概率，以及seen/unseen motif上的影响。

单原子例子必须单独审查。whole-motif `[C]` macro表达“一个完整motif identity”，而fallback
中的lexer原子`C`是可出现在`[C]`、`[CC]`、`[C(=O)N]`等多种phrase内部的组合词素；二者相关
但语义粒度不同，不能仅因字符相同就默认共享同一embedding。候选包括：

1. 不绑定，只由canonical macro surface训练；
2. 用fallback `[C]`完整phrase（包括边界/carrier）的池化表示初始化或正则macro；
3. 只在单原子motif的语义等价经实验成立时做受限parameter tying；
4. 不为单原子identity建立macro，统一走组合路径，作为长度/语义对照。

这里要检验的是whole-motif与**完整fallback phrase**的等价，而不是把whole-motif token与其中
任意一个atom lexeme机械相等。该问题与macro budget共同裁决；当前不冻结绑定策略。

### E3FP状态域与特殊行

两边实际E3FP哈希域都为4096。3D-MolT5使用4097行embedding：外部`-1`先加一映射到
padding row 0，真实ID `0..4095`映射到`1..4096`。当前adapter使用4098行：真实ID
`0..4095`保持不变，内部padding row为4096，state-mask row为4097。因此多出的一行只服务
显式E3FP状态corruption，不代表额外的E3FP类别或更高3D容量。

正式任务若不再执行离散state-mask输入，须比较/裁决是否删除mask row并收敛为
“4096 real + 1 padding”；若保留任何E3FP遮蔽/geometry-required view，则4098可保留，但必须
在model/cache/checkpoint合同中分别命名`real_state_count=4096`、`padding_id=4096`、
`mask_id=4097`，禁止把4098表述为E3FP vocabulary size。

正式冻结前保留两个显式候选，当前不预先裁决：

| 候选 | embedding行数 | ID域 | 适用条件 |
|---|---:|---|---|
| `e3fp_real_plus_padding` | 4097 | real=`0..4095`，padding=`4096` | E3FP只作为已观测输入，不执行离散state masking |
| `e3fp_real_plus_padding_mask` | 4098 | real=`0..4095`，padding=`4096`，mask=`4097` | 正式训练仍包含“存在但被主动隐藏”的E3FP状态输入 |

因此待解决问题是“是否保留第4098行”，不是“mask是否等于4098”；零基ID下第4098行的
`mask_id`为4097。裁决必须随Phase-I/II任务定义一起完成，不能只因历史checkpoint使用过
mask row而默认保留。

padding row无论是否保留mask都需要存在。训练tensor必须把不同原子数的分子补成统一
`[B,A,4]`形状，而且部分原子的高阶shell可能不存在；这些位置在外部记录中用`-1`表示，
进入embedding前映射到padding ID。padding必须与真实E3FP ID 0分离，其embedding固定为零/
不接收梯度，并由atom/shell valid mask再次排除，避免批处理补位或缺失shell被模型误解为
一个真实的哈希状态。它只表达“这里没有观测值”，不表达化学或三维状态。

### E3FP模型参数

正式预训练前还须统一裁决并冻结：

- fingerprint bits：4096是否继续与3D-MolT5对齐；
- maximum level：L0--L3及缺失shell政策；
- consumed levels：L0身份与L1--L3空间环境的组合方式；
- state embedding dim：当前64与3D-MolT5的768维直接embedding之间的容量差；
- 是否使用level embedding；
- atom reducer：V8或V9，以及高层mean/直接拼接等候选；
- motif carrier与anchor endpoint的注入比例、是否共享投影；
- padding/mask特殊行是否参与梯度、初始化与checkpoint兼容验证。

这些变量需通过PF10与独立3D敏感任务裁决；在此之前，4096、64维、V8/V9和额外mask row都
属于candidate configuration，而不是已确定的3D-motif架构常数。
