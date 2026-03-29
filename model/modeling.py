import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Stack
from .configuration import MoStT5Config


# =========================================================================
# 1. GSMATEmbeddings
# =========================================================================
class GSMATEmbeddings(nn.Module):
    def __init__(self, config: MoStT5Config, word_embeddings=None):
        super().__init__()
        if word_embeddings is not None:
            self.word_embeddings = word_embeddings
        else:
            self.word_embeddings = nn.Embedding(config.vocab_size, config.d_model)

        self.e3fp_embeddings = nn.ModuleList([
            nn.Embedding(config.e3fp_vocab_size + 1, config.d_model, padding_idx=0)
            for _ in range(config.e3fp_num_levels)
        ])

    def set_word_embeddings(self, new_embeddings):
        self.word_embeddings = new_embeddings

    def forward(self, input_ids, e3fp_ids):
        motif_embeds = self.word_embeddings(input_ids)
        e3fp_ids_shifted = e3fp_ids + 1
        e3fp_embeds = 0
        for i, layer in enumerate(self.e3fp_embeddings):
            e3fp_embeds += layer(e3fp_ids_shifted[:, :, i])
        return motif_embeds, e3fp_embeds


# =========================================================================
# 2. GeoSemanticFusion
# =========================================================================
class GeoSemanticFusion(nn.Module):
    """
    几何语义融合层 (GeoSemanticFusion) - SOTA 终极重构版
    """

    def __init__(self, config):
        super().__init__()
        self.dim = config.d_model

        # 🚀 1. 引入 3D 几何注意力映射 (Attention Pooling)
        # 替代原始粗暴的均值池化，赋予模型空间拓扑的敏感度
        self.q_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim)
        self.v_proj = nn.Linear(self.dim, self.dim)

        # 🚀 2. 门控投影网络
        self.projector = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim)
        )

        self.gate_proj = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(self.dim)
        self.dropout = nn.Dropout(config.dropout_rate)

    def compute_pooled_3d(self, motif_emb, e3fp_emb, atom_to_motif_map, atom_mask):
        Q = self.q_proj(motif_emb)  # [B, L_m, D]
        K = self.k_proj(e3fp_emb)  # [B, L_a, D]
        V = self.v_proj(e3fp_emb)  # [B, L_a, D]

        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.dim ** 0.5)

        device = motif_emb.device
        n_motifs = motif_emb.size(1)
        motif_indices = torch.arange(n_motifs, device=device).view(1, n_motifs, 1)
        atom_map_expanded = atom_to_motif_map.unsqueeze(1)

        valid_connection = (atom_map_expanded == motif_indices) & atom_mask.unsqueeze(1).bool()
        scores.masked_fill_(~valid_connection, float('-inf'))

        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        return torch.bmm(attn_weights, V)  # 返回高空间分辨率的 pooled_3d

    def forward(self, motif_emb, e3fp_emb, atom_to_motif_map, atom_mask):
        # 1. 获取高分辨率 3D 融合特征
        pooled_3d = self.compute_pooled_3d(motif_emb, e3fp_emb, atom_to_motif_map, atom_mask)

        # 2. 自适应门控融合
        projected_3d = self.projector(pooled_3d)
        concat_feat = torch.cat([motif_emb, projected_3d], dim=-1)
        gate = self.gate_proj(concat_feat)

        device = motif_emb.device
        n_motifs = motif_emb.size(1)
        motif_indices = torch.arange(n_motifs, device=device).view(1, n_motifs, 1)
        atom_map_expanded = atom_to_motif_map.unsqueeze(1)
        valid_connection = (atom_map_expanded == motif_indices) & atom_mask.unsqueeze(1).bool()

        has_atoms = valid_connection.any(dim=-1, keepdim=True).float()
        gate = gate * has_atoms

        # 🚀 3. 纯粹的凸组合，彻底斩断残差泄漏！
        fused_emb = (1.0 - gate) * motif_emb + gate * projected_3d

        return self.norm(self.dropout(fused_emb))


# =========================================================================
# 3. MoStT5Encoder
# =========================================================================
class MoStT5Encoder(T5Stack):
    def __init__(self, config: MoStT5Config, embed_tokens=None):
        super().__init__(config, embed_tokens=embed_tokens)
        self.gsm_embeddings = GSMATEmbeddings(config, word_embeddings=embed_tokens)
        self.fusion_layer = GeoSemanticFusion(config)

    def set_input_embeddings(self, new_embeddings):
        super().set_input_embeddings(new_embeddings)
        self.gsm_embeddings.set_word_embeddings(new_embeddings)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        e3fp_ids = kwargs.pop('e3fp_ids', None)
        atom_to_motif_map = kwargs.pop('atom_to_motif_map', None)
        atom_attention_mask = kwargs.pop('atom_attention_mask', None)

        motif_emb, e3fp_emb = self.gsm_embeddings(input_ids, e3fp_ids)
        inputs_embeds = self.fusion_layer(motif_emb, e3fp_emb, atom_to_motif_map, atom_attention_mask)

        valid_keys = ['output_attentions', 'output_hidden_states', 'return_dict', 'head_mask']
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}

        return super().forward(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            **filtered_kwargs
        )


# =========================================================================
# 4. MoStT5ForConditionalGeneration
# =========================================================================
class MoStT5ForConditionalGeneration(T5ForConditionalGeneration):
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]
    _keys_to_ignore_on_load_unexpected = [
        "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    def __init__(self, config: MoStT5Config):
        super().__init__(config)
        self.encoder = MoStT5Encoder(config, embed_tokens=self.shared)

        # 🔴 Geometric Head (添加正确的权重初始化)
        self.geometric_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon),
            nn.Linear(config.d_model, config.d_model)
        )

        # 🔴 关键修复：对 Geometric Head 进行正交初始化，防止梯度爆炸
        for module in self.geometric_head:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.1)  # 使用较小的正交初始化
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.post_init()

    def _init_weights(self, module):
        factor = self.config.initializer_factor
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.002)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if input_ids is None and "motif_ids" in kwargs:
            input_ids = kwargs.pop("motif_ids")

        e3fp_ids = kwargs.pop("e3fp_ids", None)
        atom_to_motif_map = kwargs.pop("atom_to_motif_map", None)
        atom_attention_mask = kwargs.pop("atom_attention_mask", None)
        kwargs.pop("motif_attention_mask", None)

        mask_positions = kwargs.pop("mask_positions", None)
        unmasked_e3fp_ids = kwargs.pop("unmasked_e3fp_ids", None)

        target_3d_pooled = None
        if mask_positions is not None and mask_positions.any():
            target_e3fp_ids = unmasked_e3fp_ids if unmasked_e3fp_ids is not None else e3fp_ids
            if target_e3fp_ids is not None:
                with torch.no_grad():
                    # 提取基础 Embedding
                    _, e3fp_emb_target = self.encoder.gsm_embeddings(input_ids, target_e3fp_ids)
                    motif_emb_target = self.encoder.gsm_embeddings.word_embeddings(input_ids)

                    unmasked_atom_mask = (target_e3fp_ids[:, :, 0] != -1).long()

                    # 🚀 接口同步升级：传入 motif_emb_target 作为 Attention Pooling 的 Query
                    target_3d_pooled = self.encoder.fusion_layer.compute_pooled_3d(
                        motif_emb=motif_emb_target,
                        e3fp_emb=e3fp_emb_target,
                        atom_to_motif_map=atom_to_motif_map,
                        atom_mask=unmasked_atom_mask
                    ).detach()

        if kwargs.get("encoder_outputs") is None:
            kwargs["encoder_outputs"] = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                e3fp_ids=e3fp_ids,
                atom_to_motif_map=atom_to_motif_map,
                atom_attention_mask=atom_attention_mask,
                **kwargs
            )

        outputs = super().forward(input_ids=None, attention_mask=attention_mask, labels=labels, **kwargs)

        # 🔴 核心升级三：防坍缩 SmoothL1Loss 计算 (彻底解除归一化封印)
        if outputs.loss is not None and target_3d_pooled is not None:
            encoder_hidden_states = kwargs["encoder_outputs"][0]
            predicted_3d = self.geometric_head(encoder_hidden_states)

            pred_masked = predicted_3d[mask_positions]
            target_masked = target_3d_pooled[mask_positions]

            if pred_masked.numel() > 0:
                # 1. 清除可能的 NaN 和 Inf，保留原始幅度和方向！
                pred_masked = torch.nan_to_num(pred_masked, nan=0.0, posinf=0.0, neginf=0.0)
                target_masked = torch.nan_to_num(target_masked, nan=0.0, posinf=0.0, neginf=0.0)

                # 2. 暴力回归真实三维特征：采用对异常值鲁棒的 SmoothL1Loss (Huber Loss)
                loss_fct_3d = nn.SmoothL1Loss(reduction='mean')
                loss_3d = loss_fct_3d(pred_masked, target_masked)

                # 3. 最终保险：如果 loss_3d 异常大，说明出现了数值不稳定，直接丢弃该项
                if loss_3d.isnan() or loss_3d.isinf():
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"⚠️ 3D Loss 出现 NaN/Inf，已自动丢弃该批次。")
                    loss_3d = torch.tensor(0.0, device=loss_3d.device)

                lambda_3d = getattr(self.config, 'lambda_3d', 0.1)
                outputs.loss = outputs.loss + lambda_3d * loss_3d

        return outputs