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
        # 统一使用 input_ids (对应 motif_ids)
        motif_embeds = self.word_embeddings(input_ids)
        e3fp_ids_shifted = e3fp_ids + 1
        e3fp_embeds = 0
        for i, layer in enumerate(self.e3fp_embeddings):
            e3fp_embeds += layer(e3fp_ids_shifted[:, :, i])
        return motif_embeds, e3fp_embeds


# =========================================================================
# 2. GeoSemanticFusion (纯 PyTorch 实现，无 torch_scatter)
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
        """
        🚀 代码重构：将底层原子 3D 特征池化到 2D Motif 级别的逻辑单独抽离，
        方便预训练在无需完整前向传播的情况下，直接生成 3D MSE Loss 的 Target！
        """
        dim = e3fp_emb.shape[-1]
        masked_e3fp = e3fp_emb * atom_mask.unsqueeze(-1)

        # 计算打平的索引
        flat_map = atom_to_motif_map + (torch.arange(batch_size, device=e3fp_emb.device).view(-1, 1) * n_motifs)
        flat_map = flat_map.view(-1)

        # 索引保护
        max_index = batch_size * n_motifs - 1
        flat_map = torch.clamp(flat_map, min=0, max=max_index)

        # 原生 PyTorch 聚合 (替代 scatter_sum)
        sum_features = torch.zeros(batch_size * n_motifs, dim, device=e3fp_emb.device, dtype=e3fp_emb.dtype)
        count_atoms = torch.zeros(batch_size * n_motifs, 1, device=e3fp_emb.device, dtype=e3fp_emb.dtype)

        sum_features.index_add_(0, flat_map, masked_e3fp.view(-1, dim))
        count_atoms.index_add_(0, flat_map, atom_mask.view(-1, 1).float())

        # 平均池化
        pooled_3d = sum_features / (count_atoms + 1e-9)
        return pooled_3d.view(batch_size, n_motifs, dim)

    def forward(self, motif_emb, e3fp_emb, atom_to_motif_map, atom_mask):
        batch_size, n_motifs, dim = motif_emb.shape

        # 1. 获取池化后的 3D 状态 (复用抽离的函数)
        pooled_3d = self.compute_pooled_3d(e3fp_emb, atom_to_motif_map, atom_mask, batch_size, n_motifs)

        # 2. 门控融合
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

        # 🚀 核心新增：Geometric Head (几何头)，用于预训练 3D 状态重构
        self.geometric_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon),
            nn.Linear(config.d_model, config.d_model)
        )

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
        # 1. 兼容性接口：将 motif_ids 自动转为 input_ids
        if input_ids is None and "motif_ids" in kwargs:
            input_ids = kwargs.pop("motif_ids")

        # 2. 提取附加参数
        e3fp_ids = kwargs.pop("e3fp_ids", None)
        atom_to_motif_map = kwargs.pop("atom_to_motif_map", None)
        atom_attention_mask = kwargs.pop("atom_attention_mask", None)
        kwargs.pop("motif_attention_mask", None)

        # 🚀 3. 提取预训练控制参数
        mask_positions = kwargs.pop("mask_positions", None)  # Collator 传来的 2D 掩码位置
        unmasked_e3fp_ids = kwargs.pop("unmasked_e3fp_ids", None)  # Collator 传来的未破坏的原始 3D ID

        # 🚀 4. 计算 3D 几何损失的 Target (仅在预训练掩码阶段执行)
        target_3d_pooled = None
        if mask_positions is not None and mask_positions.any():
            # 优先使用完整的未破坏 3D 信息计算重构目标，兜底使用当前的 e3fp_ids
            target_e3fp_ids = unmasked_e3fp_ids if unmasked_e3fp_ids is not None else e3fp_ids
            if target_e3fp_ids is not None:
                with torch.no_grad():
                    # 利用 Embedding 层独立算出无掩码的 3D E3FP 向量
                    _, e3fp_emb_target = self.encoder.gsm_embeddings(input_ids, target_e3fp_ids)
                    batch_size, n_motifs = input_ids.shape

                    # 动态生成无掩码的 atom_mask (假设 e3fp_pad_id = -1)
                    unmasked_atom_mask = (target_e3fp_ids[:, :, 0] != -1).long()

                    # 复用刚才抽离的池化函数，完美获取真实的 3D Target
                    target_3d_pooled = self.encoder.fusion_layer.compute_pooled_3d(
                        e3fp_emb_target, atom_to_motif_map, unmasked_atom_mask, batch_size, n_motifs
                    ).detach()

        # 5. 正常经过 Encoder 前向传播
        if kwargs.get("encoder_outputs") is None:
            kwargs["encoder_outputs"] = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                e3fp_ids=e3fp_ids,
                atom_to_motif_map=atom_to_motif_map,
                atom_attention_mask=atom_attention_mask,
                **kwargs
            )

        # 6. 正常经过 T5 主干，计算 2D 拓扑分类 Loss (Cross Entropy)
        outputs = super().forward(input_ids=None, attention_mask=attention_mask, labels=labels, **kwargs)

        # 🚀 7. 施加 Geometric Head 的 3D 回归 Loss
        if outputs.loss is not None and target_3d_pooled is not None:
            # 提取 Encoder 最顶层的隐藏状态
            encoder_hidden_states = kwargs["encoder_outputs"][0]

            # 使用几何头进行 3D 状态重构
            predicted_3d = self.geometric_head(encoder_hidden_states)

            # 仅在被 Mask 掉的位置强制要求 3D 重构逼近真实状态
            pred_masked = predicted_3d[mask_positions]
            target_masked = target_3d_pooled[mask_positions]

            if pred_masked.numel() > 0:
                loss_fct = nn.MSELoss()
                loss_3d = loss_fct(pred_masked, target_masked)

                # 双轨 Loss 加权融合 (预训练初期 lambda_3d 设为 0.1 最佳)
                lambda_3d = getattr(self.config, 'lambda_3d', 0.1)
                outputs.loss = outputs.loss + lambda_3d * loss_3d

        return outputs