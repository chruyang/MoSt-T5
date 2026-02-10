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

    def forward(self, motif_emb, e3fp_emb, atom_to_motif_map, atom_mask):
        batch_size, n_motifs, dim = motif_emb.shape

        # 1. 准备数据
        masked_e3fp = e3fp_emb * atom_mask.unsqueeze(-1)
        # 计算打平的索引
        flat_map = atom_to_motif_map + (torch.arange(batch_size, device=motif_emb.device).view(-1, 1) * n_motifs)
        flat_map = flat_map.view(-1)

        # 索引保护
        max_index = batch_size * n_motifs - 1
        flat_map = torch.clamp(flat_map, min=0, max=max_index)

        # 2. 原生 PyTorch 聚合 (替代 scatter_sum)
        sum_features = torch.zeros(batch_size * n_motifs, dim, device=motif_emb.device, dtype=motif_emb.dtype)
        count_atoms = torch.zeros(batch_size * n_motifs, 1, device=motif_emb.device, dtype=motif_emb.dtype)

        # 使用 index_add_
        sum_features.index_add_(0, flat_map, masked_e3fp.view(-1, dim))
        count_atoms.index_add_(0, flat_map, atom_mask.view(-1, 1).float())

        # 3. 平均池化
        pooled_3d = sum_features / (count_atoms + 1e-9)
        pooled_3d = pooled_3d.view(batch_size, n_motifs, dim)

        # 4. 融合
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

        # 这里的 input_ids 就是标准的输入
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
    # ✅ 1. 显式声明绑定键 (对齐 3D-MolT5)
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    # ✅ 2. 忽略无关警告
    _keys_to_ignore_on_load_unexpected = [
        "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    def __init__(self, config: MoStT5Config):
        super().__init__(config)
        self.encoder = MoStT5Encoder(config, embed_tokens=self.shared)
        self.post_init()

    # ✅ 3. 系统级初始化 (std=0.002)
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
        # ✅ 4. 兼容性接口：将 motif_ids 自动转为 input_ids
        if input_ids is None and "motif_ids" in kwargs:
            input_ids = kwargs.pop("motif_ids")

        # 提取并移除自定义参数
        e3fp_ids = kwargs.pop("e3fp_ids", None)
        atom_to_motif_map = kwargs.pop("atom_to_motif_map", None)
        atom_attention_mask = kwargs.pop("atom_attention_mask", None)
        kwargs.pop("motif_attention_mask", None)  # 清理旧名

        if kwargs.get("encoder_outputs") is None:
            kwargs["encoder_outputs"] = self.encoder(
                input_ids=input_ids,  # 统一使用 input_ids
                attention_mask=attention_mask,
                e3fp_ids=e3fp_ids,
                atom_to_motif_map=atom_to_motif_map,
                atom_attention_mask=atom_attention_mask,
                **kwargs
            )

        return super().forward(input_ids=None, attention_mask=attention_mask, labels=labels, **kwargs)