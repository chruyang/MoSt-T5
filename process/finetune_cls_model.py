import torch
import torch.nn as nn


class MoStT5ForSequenceClassification(nn.Module):
    def __init__(self, pretrained_model, num_labels=2, dropout_rate=0.1):
        super().__init__()
        # 1. 继承预训练好的 Encoder
        self.encoder = pretrained_model.encoder
        hidden_size = pretrained_model.config.d_model

        # 2. 仿照常规图模型的分类头设计
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_labels)
        )

    def forward(self, input_ids, attention_mask, e3fp_ids, atom_to_motif_map, atom_attention_mask, labels=None,
                **kwargs):
        # 3. 获取包含 3D 信息的 Encoder 表征
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            e3fp_ids=e3fp_ids,
            atom_to_motif_map=atom_to_motif_map,
            atom_attention_mask=atom_attention_mask
        )
        hidden_states = encoder_outputs[0]  # [batch_size, seq_len, hidden_size]

        # 4. Mean Pooling (忽略 Padding 部分)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask  # [batch_size, hidden_size]

        # 5. 计算 Logits
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}