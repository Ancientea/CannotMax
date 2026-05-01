"""
Transformer model for Arknights battle prediction.
"""
import torch
import torch.nn as nn

from ..config import MONSTER_COUNT, FIELD_FEATURE_COUNT


class UnitAwareTransformer(nn.Module):
    """
    Transformer model for predicting battle outcomes.
    
    Args:
        num_units: Total number of units (monsters + field features)
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
    """
    
    def __init__(self, num_units: int, embed_dim: int = 128, num_heads: int = 8, num_layers: int = 4):
        super().__init__()
        self.num_units = num_units
        self.monster_count = MONSTER_COUNT
        self.field_count = FIELD_FEATURE_COUNT
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        self.unit_embed = nn.Embedding(num_units, embed_dim)
        nn.init.normal_(self.unit_embed.weight, mean=0.0, std=0.02)

        self.value_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        self.enemy_attentions = nn.ModuleList()
        self.friend_attentions = nn.ModuleList()
        self.enemy_ffn = nn.ModuleList()
        self.friend_ffn = nn.ModuleList()
        self.norm = nn.ModuleList()

        for _ in range(num_layers):
            self.enemy_attentions.append(
                nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.2)
            )
            self.enemy_ffn.append(
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(embed_dim * 2, embed_dim),
                )
            )

            self.friend_attentions.append(
                nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.2)
            )
            self.friend_ffn.append(
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(embed_dim * 2, embed_dim),
                )
            )

            nn.init.xavier_uniform_(self.enemy_attentions[-1].in_proj_weight)
            nn.init.xavier_uniform_(self.friend_attentions[-1].in_proj_weight)
            self.norm.append(nn.LayerNorm(embed_dim))

        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, left_signs, left_counts, right_signs, right_counts):
        """
        Args:
            left_signs: (batch, num_units)
            left_counts: (batch, num_units)
            right_signs: (batch, num_units)
            right_counts: (batch, num_units)
        Returns:
            (batch, 1) probability
        """
        batch_size = left_counts.size(0)

        # 提取 TopK 特征（TopK 天然返回有效索引 0~num_units-1）
        k = min(8, left_counts.shape[1])
        left_values, left_indices = torch.topk(left_counts, k=k, dim=1)
        right_values, right_indices = torch.topk(right_counts, k=k, dim=1)

        # 嵌入
        left_feat = self.unit_embed(left_indices)
        right_feat = self.unit_embed(right_indices)

        embed_dim = self.embed_dim

        # 后半维度乘以数量值
        left_feat = torch.cat(
            [
                left_feat[..., : embed_dim // 2],
                left_feat[..., embed_dim // 2:] * left_values.unsqueeze(-1),
            ],
            dim=-1,
        )
        right_feat = torch.cat(
            [
                right_feat[..., : embed_dim // 2],
                right_feat[..., embed_dim // 2:] * right_values.unsqueeze(-1),
            ],
            dim=-1,
        )

        # FFN
        left_feat = left_feat + self.value_ffn(left_feat)
        right_feat = right_feat + self.value_ffn(right_feat)

        # mask (B, k)
        left_mask = left_values > 0.1
        right_mask = right_values > 0.1

        for i in range(self.num_layers):
            # 敌方注意力
            delta_left, _ = self.enemy_attentions[i](
                query=left_feat,
                key=right_feat,
                value=right_feat,
                key_padding_mask=~right_mask,
                need_weights=False,
            )
            delta_right, _ = self.enemy_attentions[i](
                query=right_feat,
                key=left_feat,
                value=left_feat,
                key_padding_mask=~left_mask,
                need_weights=False,
            )

            left_feat = left_feat + delta_left
            right_feat = right_feat + delta_right

            left_feat = left_feat + self.enemy_ffn[i](left_feat)
            right_feat = right_feat + self.enemy_ffn[i](right_feat)

            # 友方注意力
            delta_left, _ = self.friend_attentions[i](
                query=left_feat,
                key=left_feat,
                value=left_feat,
                key_padding_mask=~left_mask,
                need_weights=False,
            )
            delta_right, _ = self.friend_attentions[i](
                query=right_feat,
                key=right_feat,
                value=right_feat,
                key_padding_mask=~right_mask,
                need_weights=False,
            )

            left_feat = left_feat + delta_left
            right_feat = right_feat + delta_right

            left_feat = left_feat + self.friend_ffn[i](left_feat)
            right_feat = right_feat + self.friend_ffn[i](right_feat)

            left_feat = self.norm[i](left_feat)
            right_feat = self.norm[i](right_feat)

        L = self.fc(left_feat).squeeze(-1) * left_mask
        R = self.fc(right_feat).squeeze(-1) * right_mask

        output = torch.sigmoid(R.sum(1) - L.sum(1))
        return output
