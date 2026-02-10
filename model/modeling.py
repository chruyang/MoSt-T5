import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Stack
from torch_scatter import scatter_sum
from .configuration import MoStT5Config


# =========================================================================
# 1. GSMATEmbeddings: 支持动态更新 Embedding 引用
# =========================================================================
class GSMATEmbeddings(nn.Module):
    def __init__(self, config: MoStT5Config, word_embeddings=None):
        super().__init__()
        # 如果传入了 word_embeddings，直接引用（实现共享）
        if word_embeddings is not None:
            self.word_embeddings = word_embeddings
        else:
            self.word_embeddings = nn.Embedding(config.vocab_size, config.d_model)

        self.e3fp_embeddings = nn.ModuleList([
            nn.Embedding(config.e3fp_vocab_size + 1, config.d_model, padding_idx=0)
            for _ in range(config.e3fp_num_levels)
        ])

    def set_word_embeddings(self, new_embeddings):
        """✅ 关键修复：允许外部更新 word_embeddings 引用"""
        self.word_embeddings = new_embeddings

    def forward(self, motif_ids, e3fp_ids):
        motif_embeds = self.word_embeddings(motif_ids)
        e3fp_ids_shifted = e3fp_ids + 1
        e3fp_embeds = 0
        for i, layer in enumerate(self.e3fp_embeddings):
            e3fp_embeds += layer(e3fp_ids_shifted[:, :, i])
        return motif_embeds, e3fp_embeds


# =========================================================================
# 2. Fusion Layer (无变化)
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

    def forward(self, motif_emb, e3fp_emb, atom_to_motif_map, atom_mask):
        batch_size, n_motifs, dim = motif_emb.shape
        masked_e3fp = e3fp_emb * atom_mask.unsqueeze(-1)
        offset = torch.arange(batch_size, device=motif_emb.device).view(-1, 1) * n_motifs
        flat_map = (atom_to_motif_map + offset).view(-1)

        # 索引保护
        max_index = batch_size * n_motifs - 1
        flat_map = torch.clamp(flat_map, min=0, max=max_index)

        sum_features = scatter_sum(masked_e3fp.view(-1, dim), flat_map, dim=0, dim_size=batch_size * n_motifs)
        count_atoms = scatter_sum(atom_mask.view(-1, 1).float(), flat_map, dim=0, dim_size=batch_size * n_motifs)
        pooled_3d = sum_features / (count_atoms + 1e-9)
        pooled_3d = pooled_3d.view(batch_size, n_motifs, dim)

        concat_feat = torch.cat([motif_emb, pooled_3d], dim=-1)
        alpha = self.gate_proj(concat_feat)
        fused_emb = (1 - alpha) * motif_emb + alpha * pooled_3d
        fused_emb = self.projector(fused_emb)
        return self.norm(motif_emb + self.dropout(fused_emb))


# =========================================================================
# 3. MoStT5Encoder: 修复 resize_token_embeddings 导致的引用断裂
# =========================================================================
class MoStT5Encoder(T5Stack):
    def __init__(self, config: MoStT5Config, embed_tokens=None):
        super().__init__(config, embed_tokens=embed_tokens)
        self.gsm_embeddings = GSMATEmbeddings(config, word_embeddings=embed_tokens)
        self.fusion_layer = GeoSemanticFusion(config)

    def set_input_embeddings(self, new_embeddings):
        """✅ 关键修复：当 T5Stack 更新 embed_tokens 时，同步更新 gsm_embeddings"""
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

        # 显式 use_cache=False
        return super().forward(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            **filtered_kwargs
        )


# =========================================================================
# 4. MoStT5ForConditionalGeneration: 完善初始化与权重绑定
# =========================================================================
class MoStT5ForConditionalGeneration(T5ForConditionalGeneration):
    # ✅ 1. 参考 3D-MolT5: 显式声明绑定键，帮助 HF 识别共享权重
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    def __init__(self, config: MoStT5Config):
        super().__init__(config)

        # 共享权重
        self.encoder = MoStT5Encoder(config, embed_tokens=self.shared)

        # 初始化
        self.post_init()

    # ✅ 2. 核心修复：重写系统级初始化逻辑
    # 任何时候调用 init_weights (包括 from_pretrained 内部)，都会强制使用 std=0.002
    def _init_weights(self, module):
        factor = self.config.initializer_factor  # T5默认是 1.0

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            # ⚠️ 强制所有 Embedding 使用极小方差 (0.002)
            # 这比在 __init__ 里手动设置更安全，因为它会覆盖所有场景
            module.weight.data.normal_(mean=0.0, std=0.002)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def forward(self, motif_ids=None, motif_attention_mask=None, e3fp_ids=None, atom_to_motif_map=None,
                atom_attention_mask=None, labels=None, **kwargs):
        if motif_attention_mask is None:
            motif_attention_mask = kwargs.pop("attention_mask", None)

        if kwargs.get("encoder_outputs") is None:
            kwargs["encoder_outputs"] = self.encoder(
                input_ids=motif_ids,
                attention_mask=motif_attention_mask,
                e3fp_ids=e3fp_ids,
                atom_to_motif_map=atom_to_motif_map,
                atom_attention_mask=atom_attention_mask,
                **kwargs
            )

        for k in ['motif_ids', 'motif_attention_mask', 'e3fp_ids', 'atom_to_motif_map', 'atom_attention_mask']:
            kwargs.pop(k, None)

        return super().forward(labels=labels, **kwargs)