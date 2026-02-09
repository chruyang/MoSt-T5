import torch
import torch.nn as nn
from transformers import T5EncoderModel, T5ForConditionalGeneration
from torch_scatter import scatter_sum
from .configuration import MoStT5Config


class GSMATEmbeddings(nn.Module):
    """
    双模态 Embedding 层：
    1. Motif Branch: 2D 语义嵌入，复用 T5 权重。
    2. E3FP Branch: 3D 结构嵌入，采用极小值初始化以防 NaN。
    """

    def __init__(self, config: MoStT5Config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.d_model)

        # 3D 结构分支：多层级 E3FP 指纹嵌入
        self.e3fp_embeddings = nn.ModuleList([
            nn.Embedding(config.e3fp_vocab_size + 1, config.d_model, padding_idx=0)
            for _ in range(config.e3fp_num_levels)
        ])

        # ✅ 初始化优化 (参考 3D-MolT5):
        # 将 3D Embedding 标准差设为极小值，确保训练初期由 T5 预训练语义主导
        for emb in self.e3fp_embeddings:
            nn.init.normal_(emb.weight, std=1e-5)

    def forward(self, motif_ids, e3fp_ids):
        # 2D 分支
        motif_embeds = self.word_embeddings(motif_ids)

        # 3D 分支：处理 Padding (ID + 1)
        e3fp_ids_shifted = e3fp_ids + 1
        e3fp_embeds = 0
        for i, layer in enumerate(self.e3fp_embeddings):
            e3fp_embeds += layer(e3fp_ids_shifted[:, :, i])

        return motif_embeds, e3fp_embeds


class GeoSemanticFusion(nn.Module):
    """
    方案B：特征级自适应门控融合 (Feature-wise Adaptive Gating)
    通过学习一个门控向量，在每个特征维度上自动调节 2D 和 3D 的比例。
    """

    def __init__(self, config: MoStT5Config):
        super().__init__()
        self.hidden_dim = config.d_model

        # 门控网络：生成 Feature-wise 权重向量 [B, N, D]
        self.gate_proj = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid()
        )

        # ✅ 初始化技巧 (Initialization Trick):
        # 将 Bias 设为 0，使 Sigmoid 初期输出在 0.5 附近，实现均衡融合
        nn.init.zeros_(self.gate_proj[-2].bias)

        self.projector = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.norm = nn.LayerNorm(self.hidden_dim, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, motif_emb, e3fp_emb, atom_to_motif_map, atom_mask):
        batch_size, n_motifs, dim = motif_emb.shape

        # 1. Micro-to-Macro 聚合 (参考 3D-MolT5 均值聚合逻辑)
        masked_e3fp = e3fp_emb * atom_mask.unsqueeze(-1)
        offset = torch.arange(batch_size, device=motif_emb.device).view(-1, 1) * n_motifs
        flat_map = (atom_to_motif_map + offset).view(-1)

        sum_features = scatter_sum(masked_e3fp.view(-1, dim), flat_map, dim=0, dim_size=batch_size * n_motifs)
        count_atoms = scatter_sum(atom_mask.view(-1, 1).float(), flat_map, dim=0, dim_size=batch_size * n_motifs)

        pooled_3d = sum_features / (count_atoms + 1e-9)
        pooled_3d = pooled_3d.view(batch_size, n_motifs, dim)

        # 2. 特征级门控融合 (方案B 核心)
        concat_feat = torch.cat([motif_emb, pooled_3d], dim=-1)
        alpha = self.gate_proj(concat_feat)  # [B, N, D]

        # 融合公式：Fused = (1 - alpha) * 2D + alpha * 3D
        fused_emb = (1 - alpha) * motif_emb + alpha * pooled_3d

        # 3. 后处理与残差
        fused_emb = self.projector(fused_emb)
        return self.norm(motif_emb + self.dropout(fused_emb))


class MoStT5Encoder(T5EncoderModel):
    def __init__(self, config: MoStT5Config):
        super().__init__(config)
        self.gsm_embeddings = GSMATEmbeddings(config)
        self.fusion_layer = GeoSemanticFusion(config)

        # ✅ 数值加固 (参考 3D-MolT5): PAD Token 权重置零
        self.gsm_embeddings.word_embeddings.weight.data[0] = torch.zeros(config.d_model)

        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # 提取自定义 3D 参数
        e3fp_ids = kwargs.pop('e3fp_ids', None)
        atom_to_motif_map = kwargs.pop('atom_to_motif_map', None)
        atom_attention_mask = kwargs.pop('atom_attention_mask', None)

        motif_emb, e3fp_emb = self.gsm_embeddings(input_ids, e3fp_ids)
        inputs_embeds = self.fusion_layer(motif_emb, e3fp_emb, atom_to_motif_map, atom_attention_mask)

        # 参数白名单过滤，确保向下传递时不含非法参数
        valid_keys = ['output_attentions', 'output_hidden_states', 'return_dict', 'head_mask']
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}

        return super().forward(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **filtered_kwargs
        )


class MoStT5ForConditionalGeneration(T5ForConditionalGeneration):
    def __init__(self, config: MoStT5Config):
        super().__init__(config)
        self.encoder = MoStT5Encoder(config)
        self.post_init()

    def forward(self, motif_ids=None, motif_attention_mask=None, **kwargs):
        # 处理 generate() 时的 attention_mask 命名冲突
        if motif_attention_mask is None:
            motif_attention_mask = kwargs.pop("attention_mask", None)

        if "encoder_outputs" not in kwargs:
            kwargs["encoder_outputs"] = self.encoder(
                input_ids=motif_ids,
                attention_mask=motif_attention_mask,
                **kwargs
            )

        # 彻底清理自定义参数，防止传给 Decoder 引起 TypeError
        for k in ['e3fp_ids', 'atom_to_motif_map', 'atom_attention_mask']:
            kwargs.pop(k, None)

        return super().forward(**kwargs)