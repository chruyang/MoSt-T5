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
    def __init__(self, config: MoStT5Config):
        super().__init__()
        self.hidden_dim = config.d_model
        self.gate_proj = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid()
        )
        nn.init.zeros_(self.gate_proj[-2].bias)
        self.projector = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.norm = nn.LayerNorm(self.hidden_dim, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def compute_pooled_3d(self, e3fp_emb, atom_to_motif_map, atom_mask, batch_size, n_motifs):
        """将底层原子 3D 特征池化到 2D Motif 级别"""
        dim = e3fp_emb.shape[-1]

        # 1. 🚀 建立有效映射掩码 (排除 -1 的干扰)
        valid_map_mask = (atom_to_motif_map >= 0).long()
        # 双重保险：必须是真实原子 (atom_mask) 且有合法映射
        final_mask = atom_mask * valid_map_mask

        masked_e3fp = e3fp_emb * final_mask.unsqueeze(-1)

        # 2. 🚀 安全截断：把 -1 变成 0 避免计算出错
        # 因为 final_mask 已经是 0，所以该位置特征不会被计入 sum
        safe_map = torch.clamp(atom_to_motif_map, min=0)

        # 3. 计算展平索引
        flat_map = safe_map + (torch.arange(batch_size, device=e3fp_emb.device).view(-1, 1) * n_motifs)
        flat_map = flat_map.view(-1)

        # 初始化统计张量
        sum_features = torch.zeros(batch_size * n_motifs, dim, device=e3fp_emb.device, dtype=e3fp_emb.dtype)
        count_atoms = torch.zeros(batch_size * n_motifs, 1, device=e3fp_emb.device, dtype=e3fp_emb.dtype)

        # 4. 执行聚合
        sum_features.index_add_(0, flat_map, masked_e3fp.view(-1, dim))
        count_atoms.index_add_(0, flat_map, final_mask.view(-1, 1).float())

        return (sum_features / (count_atoms + 1e-9)).view(batch_size, n_motifs, dim)

    def forward(self, motif_emb, e3fp_emb, atom_to_motif_map, atom_mask):
        batch_size, n_motifs, dim = motif_emb.shape
        pooled_3d = self.compute_pooled_3d(e3fp_emb, atom_to_motif_map, atom_mask, batch_size, n_motifs)

        concat_feat = torch.cat([motif_emb, pooled_3d], dim=-1)
        alpha = self.gate_proj(concat_feat)
        fused_emb = (1 - alpha) * motif_emb + alpha * pooled_3d
        fused_emb = self.projector(fused_emb)

        return self.norm(motif_emb + self.dropout(fused_emb))


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
                    _, e3fp_emb_target = self.encoder.gsm_embeddings(input_ids, target_e3fp_ids)
                    batch_size, n_motifs = input_ids.shape
                    unmasked_atom_mask = (target_e3fp_ids[:, :, 0] != -1).long()

                    target_3d_pooled = self.encoder.fusion_layer.compute_pooled_3d(
                        e3fp_emb_target, atom_to_motif_map, unmasked_atom_mask, batch_size, n_motifs
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

        # 🔴 核心升级三：防坍缩 Cosine Loss 计算
        if outputs.loss is not None and target_3d_pooled is not None:
            encoder_hidden_states = kwargs["encoder_outputs"][0]
            predicted_3d = self.geometric_head(encoder_hidden_states)

            pred_masked = predicted_3d[mask_positions]
            target_masked = target_3d_pooled[mask_positions]

            if pred_masked.numel() > 0:
                # 🔴 关键修复：添加数值稳定性保护
                # 1. 强制 L2 归一化，学习结构方向而非幅度
                pred_masked_norm = torch.nn.functional.normalize(pred_masked, p=2, dim=-1)
                target_masked_norm = torch.nn.functional.normalize(target_masked, p=2, dim=-1)

                # 2. 清除可能的 NaN 和 Inf
                pred_masked_norm = torch.nan_to_num(pred_masked_norm, nan=0.0, posinf=0.0, neginf=0.0)
                target_masked_norm = torch.nan_to_num(target_masked_norm, nan=0.0, posinf=0.0, neginf=0.0)

                # 3. 限制余弦相似度的数值范围，避免极端值
                pred_masked_norm = torch.clamp(pred_masked_norm, min=-1.0, max=1.0)
                target_masked_norm = torch.clamp(target_masked_norm, min=-1.0, max=1.0)

                loss_fct_3d = nn.CosineEmbeddingLoss(reduction='mean')
                # 目标设为 1，希望向量方向完全一致
                target_labels = torch.ones(pred_masked_norm.size(0), device=pred_masked_norm.device)

                loss_3d = loss_fct_3d(pred_masked_norm, target_masked_norm, target_labels)

                # 🔴 最终保险：如果 loss_3d 异常大，说明出现了数值不稳定，直接丢弃该项
                if loss_3d.isnan() or loss_3d.isinf():
                    logger.warning(f"⚠️ 3D Loss 出现 NaN/Inf，已自动丢弃该批次。Loss: {loss_3d.item()}")
                    loss_3d = torch.tensor(0.0, device=loss_3d.device)

                lambda_3d = getattr(self.config, 'lambda_3d', 0.1)
                outputs.loss = outputs.loss + lambda_3d * loss_3d

        return outputs