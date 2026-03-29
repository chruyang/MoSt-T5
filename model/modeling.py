import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Stack
from .configuration import MoStT5Config
import logging

from dataclasses import dataclass
from typing import Optional
from transformers.modeling_outputs import Seq2SeqLMOutput

logger = logging.getLogger(__name__)

# =========================================================================
# 🚀 新增：定义 DDP 兼容的自定义输出类
# =========================================================================
@dataclass
class MoStT5Output(Seq2SeqLMOutput):
    main_lm_loss: Optional[torch.FloatTensor] = None
    geom_3d_loss: Optional[torch.FloatTensor] = None


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
    几何语义融合层 (GeoSemanticFusion) - SOTA 防弹重构版
    """

    def __init__(self, config):
        super().__init__()
        self.dim = config.d_model

        self.q_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim)
        self.v_proj = nn.Linear(self.dim, self.dim)

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

        self.dropout = nn.Dropout(config.dropout_rate)

    def compute_pooled_3d(self, motif_emb, e3fp_emb, atom_to_motif_map, atom_mask):
        Q = self.q_proj(motif_emb)  # [B, L_m, D]
        K = self.k_proj(e3fp_emb)  # [B, L_a, D]
        V = self.v_proj(e3fp_emb)  # [B, L_a, D]

        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.dim ** 0.5)

        device = motif_emb.device
        B, L_m, _ = motif_emb.shape
        L_a = e3fp_emb.shape[1]

        motif_indices = torch.arange(L_m, device=device).view(1, L_m, 1)
        atom_map_expanded = atom_to_motif_map.unsqueeze(1)

        valid_connection = (atom_map_expanded == motif_indices) & atom_mask.unsqueeze(1).bool()

        # 🚀 终极防爆修复：张量广播级安全防护盾
        valid_rows = valid_connection.any(dim=-1, keepdim=True)  # [B, L_m, 1]

        # 如果有一行没有任何连接，我们在它的第 0 位打个标记（假装它连接了第 0 个原子）
        # 先创建一个 [1, 1, L_a] 的标记，只有第一列为 True
        dummy_mask = torch.zeros((1, 1, L_a), dtype=torch.bool, device=device)
        if L_a > 0:
            dummy_mask[0, 0, 0] = True

        # 将空行替换为 dummy_mask
        safe_valid_connection = torch.where(
            valid_rows,
            valid_connection,  # 有原子的行，保持原样
            dummy_mask  # 没有原子的行，用假标记
        )

        scores.masked_fill_(~safe_valid_connection, float('-inf'))
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)

        # 安全计算完 Softmax 后，把真正空行的权重强制清零！
        attn_weights = attn_weights * valid_rows.float()

        return torch.bmm(attn_weights, V)

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

        # 3. 纯粹的凸组合
        fused_emb = (1.0 - gate) * motif_emb + gate * projected_3d

        return self.dropout(fused_emb)


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
    # 🚀 修复核心：显式列出所有权重共享路径，包含嵌套在子模块中的路径
    _tied_weights_keys = [
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
        "shared.weight",
        "encoder.gsm_embeddings.word_embeddings.weight"
    ]

    _keys_to_ignore_on_load_unexpected = [
        "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    def __init__(self, config: MoStT5Config):
        super().__init__(config)
        self.encoder = MoStT5Encoder(config, embed_tokens=self.shared)

        self.geometric_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model)
        )

        # 🚀 步骤 1：先执行通用的 post_init
        self.post_init()

        # 🚀 步骤 2：执行“覆盖式”特殊初始化
        self.encoder.fusion_layer.apply(self._init_weights)
        self.encoder.gsm_embeddings.e3fp_embeddings.apply(self._init_weights)

        for module in self.geometric_head:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

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

        # 1. 获取 Encoder 内部特征
        if mask_positions is not None and mask_positions.any():
            target_e3fp_ids = unmasked_e3fp_ids if unmasked_e3fp_ids is not None else e3fp_ids
            if target_e3fp_ids is not None:
                with torch.no_grad():
                    _, e3fp_emb_target = self.encoder.gsm_embeddings(input_ids, target_e3fp_ids)
                    motif_emb_target = self.encoder.gsm_embeddings.word_embeddings(input_ids)
                    unmasked_atom_mask = (target_e3fp_ids[:, :, 0] != -1).long()

                    target_3d_pooled = self.encoder.fusion_layer.compute_pooled_3d(
                        motif_emb=motif_emb_target,
                        e3fp_emb=e3fp_emb_target,
                        atom_to_motif_map=atom_to_motif_map,
                        atom_mask=unmasked_atom_mask
                    ).detach()

        # 2. 让大模型自己走一遍完整前向
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

        # 3. 计算 3D 几何 Loss
        lambda_3d = getattr(self.config, 'lambda_3d', 0.1)

        if outputs.loss is not None:
            # 🚀 提取原始文本 Loss
            main_lm_loss_val = outputs.loss.detach().clone()

            # ！！！已彻底移除 NaN 抑制逻辑，允许真实报错！！！

            encoder_hidden_states = kwargs["encoder_outputs"][0]
            predicted_3d_full = self.geometric_head(encoder_hidden_states)
            dummy_loss_3d = predicted_3d_full.sum() * 0.0

            valid_3d_loss = None
            if target_3d_pooled is not None and mask_positions is not None and mask_positions.any():
                pred_masked = predicted_3d_full[mask_positions]
                target_masked = target_3d_pooled[mask_positions]

                if pred_masked.numel() > 0:
                    pred_masked = torch.nan_to_num(pred_masked, nan=0.0, posinf=0.0, neginf=0.0)
                    target_masked = torch.nan_to_num(target_masked, nan=0.0, posinf=0.0, neginf=0.0)

                    loss_fct_3d = nn.SmoothL1Loss(reduction='mean')
                    current_loss = loss_fct_3d(pred_masked, target_masked)

                    if not (current_loss.isnan() or current_loss.isinf()):
                        valid_3d_loss = current_loss

            if valid_3d_loss is not None:
                final_3d_loss = valid_3d_loss + dummy_loss_3d
            else:
                final_3d_loss = dummy_loss_3d

            # 更新总 loss 用于梯度的反向传播
            new_total_loss = outputs.loss + lambda_3d * final_3d_loss

            # 🚀 封装成 MoStT5Output 返回，完美兼容 DDP 并透传子 Loss
            return MoStT5Output(
                loss=new_total_loss,
                logits=outputs.logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
                main_lm_loss=main_lm_loss_val,
                geom_3d_loss=final_3d_loss.detach().clone()
            )

        return outputs