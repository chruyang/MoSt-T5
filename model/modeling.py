import torch
import torch.nn as nn
from transformers import T5EncoderModel, T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from torch_scatter import scatter_mean
from .configuration_most_t5 import MoStT5Config


class GSMATEmbeddings(nn.Module):
    """
    双模态 Embedding 层:
    1. Motif Branch: 标准 T5 Embedding
    2. Atom Branch: E3FP Multi-level Embedding
    """

    def __init__(self, config: MoStT5Config):
        super().__init__()
        self.d_model = config.d_model

        # 1. Motif Embedding (复用 T5 的词表，包含 Text + Motif)
        self.word_embeddings = nn.Embedding(config.vocab_size, config.d_model)

        # 2. E3FP Embedding (多层级求和)
        # 注意: padding_idx=0, 词表大小 = bits + 1 (为 padding 留位)
        self.e3fp_embeddings = nn.ModuleList([
            nn.Embedding(config.e3fp_vocab_size + 1, config.d_model, padding_idx=0)
            for _ in range(config.e3fp_num_levels)
        ])

        self.LayerNorm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, motif_ids, e3fp_ids):
        # motif_ids: [B, Seq_Len]
        # e3fp_ids:  [B, Atom_Num, Levels] (Dataset 中 Pad 为 -1)

        # --- 2D Branch ---
        motif_embeds = self.word_embeddings(motif_ids)

        # --- 3D Branch ---
        # 关键: Dataset 中 Pad 是 -1，这里 +1 变为 0，对应 padding_idx=0
        e3fp_ids_shifted = e3fp_ids + 1

        e3fp_embeds = 0
        for i, layer in enumerate(self.e3fp_embeddings):
            # 累加所有 Level 的特征
            e3fp_embeds += layer(e3fp_ids_shifted[:, :, i])

        return motif_embeds, e3fp_embeds


class GeoSemanticFusion(nn.Module):
    def __init__(self, config: MoStT5Config):
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
        # 投影层 (上一轮建议加上的)
        self.projector = nn.Linear(config.d_model, config.d_model)

    def forward(self, motif_embeds, e3fp_embeds, atom_to_motif_map, atom_mask):
        """
        motif_embeds: [B, N_Motifs, D]
        e3fp_embeds: [B, N_Atoms, D]
        atom_to_motif_map: [B, N_Atoms]
        atom_mask: [B, N_Atoms] (1 for valid, 0 for pad)
        """
        batch_size, n_motifs, dim = motif_embeds.shape
        n_atoms = e3fp_embeds.shape[1]

        # =========================================================
        # [修复 1] 安全检查 (您建议的)
        # =========================================================
        if atom_to_motif_map.max() >= n_motifs:
            raise ValueError(f"Map index {atom_to_motif_map.max()} out of bounds for {n_motifs} motifs")

        # =========================================================
        # [修复 2] 解决 Padding 污染 Motif 0 的问题
        # =========================================================

        # 1. 准备特征：Mask 掉 Padding 原子的特征 (变为0)
        # atom_mask: [B, N] -> [B, N, 1]
        masked_e3fp = e3fp_embeds * atom_mask.unsqueeze(-1)

        # 2. 准备索引：加上 Batch 偏移
        offset = torch.arange(batch_size, device=motif_embeds.device) * n_motifs
        offset = offset.view(-1, 1)

        # 展平
        flat_map = (atom_to_motif_map + offset).view(-1)  # [B*N_Atoms]
        flat_e3fp = masked_e3fp.view(-1, dim)  # [B*N_Atoms, D]
        flat_mask = atom_mask.view(-1, 1)  # [B*N_Atoms, 1]

        # 3. 聚合计算 (手动实现 Mean 以避免 Padding 干扰)
        # scatter_mean 会把 Padding 原子的 0 值也算入分母，导致 Motif 0 被稀释
        # 所以我们用 Sum / Count 的方式

        # 分子：特征求和 (Padding 原子特征为 0，不影响 Sum)
        from torch_scatter import scatter_sum
        sum_features = scatter_sum(
            flat_e3fp,
            flat_map,
            dim=0,
            dim_size=batch_size * n_motifs
        )

        # 分母：有效原子计数 (Padding 原子 Mask 为 0，不增加 Count)
        # 注意：这里我们 scatter mask 本身，得到每个 Motif 有多少个真实原子
        count_atoms = scatter_sum(
            flat_mask.float(),  # 必须转 float
            flat_map,
            dim=0,
            dim_size=batch_size * n_motifs
        )

        # 避免除以 0 (加上 epsilon)
        count_atoms = count_atoms + 1e-9

        # 计算真正的 Mean
        pooled_3d = sum_features / count_atoms

        # =========================================================

        # 4. 恢复形状与注入
        pooled_3d = pooled_3d.view(batch_size, n_motifs, dim)
        pooled_3d = self.projector(pooled_3d)

        fused_embeds = motif_embeds + self.dropout(pooled_3d)
        fused_embeds = self.norm(fused_embeds)

        return fused_embeds


class MoStT5Encoder(T5EncoderModel):
    """
    自定义 Encoder: 替换 T5 原生的 Embedding 层，加入 Geo-Semantic Fusion
    """

    def __init__(self, config: MoStT5Config):
        super().__init__(config)
        self.gsm_embeddings = GSMATEmbeddings(config)
        self.fusion_layer = GeoSemanticFusion(config)

        # 初始化权重
        self.post_init()
        # 绑定权重，确保 shared embedding 正常工作
        self.shared = self.gsm_embeddings.word_embeddings

    def forward(self,
                input_ids=None,  # 这里传入 motif_ids
                attention_mask=None,  # 这里传入 motif_attention_mask
                e3fp_ids=None,  # 新增
                atom_to_motif_map=None,  # 新增
                atom_attention_mask=None,  # 新增
                **kwargs):
        # 1. 获取 Embedding
        motif_emb, e3fp_emb = self.gsm_embeddings(input_ids, e3fp_ids)

        # 2. 执行融合
        inputs_embeds = self.fusion_layer(
            motif_emb,
            e3fp_emb,
            atom_to_motif_map,
            atom_attention_mask
        )

        # 3. 调用 T5 Encoder (传入 inputs_embeds 而非 input_ids)
        return super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )


class MoStT5ForConditionalGeneration(T5ForConditionalGeneration):
    """
    完整的生成模型 (Encoder-Decoder)
    Encoder: MoStT5Encoder (带 3D 融合)
    Decoder: 标准 T5 Decoder (生成文本)
    """

    def __init__(self, config: MoStT5Config):
        # 初始化父类，这会创建 self.encoder 和 self.decoder
        super().__init__(config)

        # 核心：用我们的自定义 Encoder 替换掉 T5 原生的 Encoder
        self.encoder = MoStT5Encoder(config)

        self.post_init()

    def forward(self,
                motif_ids=None,
                motif_attention_mask=None,
                e3fp_ids=None,
                atom_to_motif_map=None,
                atom_attention_mask=None,
                labels=None,
                **kwargs):
        # 这里的参数名必须与 Collator 输出的 Key 一致
        # 或者在 Trainer 中会通过 **inputs 传入

        # 1. Encoder 前向传播
        encoder_outputs = self.encoder(
            input_ids=motif_ids,
            attention_mask=motif_attention_mask,
            e3fp_ids=e3fp_ids,
            atom_to_motif_map=atom_to_motif_map,
            atom_attention_mask=atom_attention_mask,
            **kwargs
        )

        # 2. Decoder 前向传播 (标准 T5 流程)
        return super().forward(
            encoder_outputs=encoder_outputs,
            labels=labels,  # 文本生成的 Labels
            **kwargs
        )