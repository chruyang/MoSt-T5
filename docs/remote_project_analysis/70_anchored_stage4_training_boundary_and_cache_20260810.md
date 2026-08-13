# Anchored motif Stage 4 训练边界与缓存（2026-08-10）

## 冻结接口

`most_t5_next/p2/anchored_training_record_v1.py` 固定以下数据所有权：

- anchored surface 唯一拥有 model-facing token axis；
- 历史 paired record 只提供已验证的 atom axis、E3FP、attachment role 和来源哈希；
- GraphPorts input IDs 不进入新记录；
- 每个 anchor occurrence 映射到其精确 attachment atom；
- macro carrier 是其单 token，fallback carrier 是 `<MOST:FALLBACK_END>`；
- identity 被选中时整个 identity phrase 替换为 sentinel，carrier 临时变为 sentinel；
  未选中时仍使用 phrase 末端 carrier。

为此，`production_bridge.collate_production_motif_record` 不再假定 carrier 永远是
identity span 的第一个 token，而是读取记录中已验证的 `logical_to_carrier`。旧
GraphPorts 记录的 carrier 本来就在 span 首位，因此旧路径数值行为不变。

## 真实 PF1 绑定证据

- selection 0：旧 GraphPorts 91 tokens，新 anchored 43 tokens；15 motifs、28 anchors；
  atom ownership 与15行E3FP逐值不变；
- 首个 fallback（selection 2）：4 motifs、1 fallback；suffix ID=32100且确为
  fallback carrier；动态corruption后1个被选motif使用sentinel carrier，3个未选
  motif均保持phrase末端carrier，所有anchors保留；
- 8-worker有界decode+bind前128条耗时2.5975秒（49.28 records/s），输出顺序与
  split顺序一致。

## 训练 mmap cache

cache 继续复用既有 flat arrays + offsets ABI。随机 epoch mask、sentinel target、
padding、view选择和dropout均不提前固化。

- `derive_anchored_donor_atom_maps_v1.py` 从既有PF10 canonical-local donor sidecar按
  PF1 storage key派生33,600条顺序一致子集，不重算化学/几何；artifact SHA-256为
  `fa2ef9029ff578aac21c95f449d7fd88541e9ed9f5dac556230e43a4970d02e3`；
- `build_anchored_training_tensor_cache_v1.py` 把anchored token轴、anchor地址、E3FP、
  Morgan对照和canonical-local atom地址写入同一mmap cache，并在manifest绑定surface、
  macro registry、tokenizer manifest和donor sidecar哈希。

真实128条cache smoke：64 train + 64 dev、3238 tokens、946 motifs、1818 atoms；8个
decode workers下构建4.866秒，完整mmap重载和动态F3D collate通过。

100 batches×16 members稳态CPU测试中，0/4/8 DataLoader workers分别约
964/471/503 members/s，且三种设置的成员顺序、动态mask形状和21,050个anchor endpoint
计数完全一致。当前小序列collate已经不是CPU瓶颈，多worker的IPC反而更慢；正式GPU
训练应从0--2 workers起测，而离线canonical decode/build可使用8--12 workers。不得
为了提高CPU占用率而选择更慢的热路径配置。

完整PF1 candidate cache随后用12个decode workers完成：33,600 records、822,737
tokens、241,799 motifs、474,586 atoms；构建耗时61.64秒（545.12 records/s），GPU
保持0 MiB。正式目录为远端
`/root/autodl-tmp/anchored-training-cache-pf1-full-v1-b0693bb`。这证明CPU并行应集中在
一次性严格decode/编译阶段，训练时直接读取mmap。

完整cache上100 batches×32 members的复测为：0/2/4 workers约
1176/758/944 members/s；39,874个endpoint、成员顺序和动态batch形状逐项一致。
因此PF1规模下默认先采用0 worker；若GPU实测仍出现data wait，再测1--2个persistent
workers，而不是直接使用8--16个worker。

## 科研边界

以上仍是PF1 candidate证据；完整Phase-I corpus重算macro registry并完成全部注册训练
语料anchor-domain census之前，不构成正式预训练准入。cache替代的是训练热路径，不
替代权威canonical release或其审计证据。

## Level-explicit atom memory 候选

`motif_geometry_adapter_v4.py` 与 `factorized_motif_t5_v4.py` 已把文档67的四个候选落为
同一参数拓扑：

1. `l12_mean`：仅L1/L2均值；
2. `l0_l12_mean`：显式L0 identity slot + L1/L2均值；
3. `l0123_mean`：四层共同均值；
4. `l0_shell_attention_l123`：显式L0 identity slot + 对L1--L3的可学习shell attention。

四种模式始终向同一个atom encoder提供两个固定宽度slot、两个presence bit和相同role
embedding，因此比较不会暗中改变参数量。checkpoint extra state绑定mode并拒绝跨mode
加载。L0只解释为原子级2D identity/context补充；任何3D主张仍必须来自higher shell
并通过B2D及aligned/zero/matched-shuffle对照。远端CPU上adapter V1--V4、anchored
reader/cache联合44项回归通过，另V3/V4 tiny-T5前向/反向5项通过。

## 2026-08-11 状态修订

本节四模式是用于回答 shell/L0/L3 问题的**配对消融脚手架**，不再作为正式 atom encoder
的复杂度下界。文档77确认 V4 路径可用后，正式候选改为3D-MolT5参考语义：共享E3FP
embedding、缺失shell固定零、固定四槽均值。role只保留为endpoint路由合同。cache无需
重建，因为它已经保存原始 `[atom,4]` IDs、atom ownership与endpoint address。见文档78。
