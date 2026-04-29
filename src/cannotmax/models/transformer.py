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

        # 构建索引 (怪物索引 + 场地索引)
        monster_indices = torch.arange(self.monster_count, device=left_counts.device)
        field_indices = torch.arange(self.monster_count, self.monster_count + self.field_count, device=left_counts.device)

        # 提取怪物和场地计数
        left_monster_counts = left_counts[:, :self.monster_count]
        left_field_counts = left_counts[:, self.monster_count:]
        right_monster_counts = right_counts[:, :self.monster_count]
        right_field_counts = right_counts[:, self.monster_count:]

        # 计算非零单位掩码
        left_monster_mask = left_monster_counts > 0
        right_monster_mask = right_monster_counts > 0

        # 生成单位索引
        left_unit_indices = torch.where(
            left_monster_mask,
            monster_indices.expand(batch_size, -1),
            torch.full((batch_size, self.monster_count), -1, device=left_counts.device, dtype=torch.long)
        )
        left_unit_indices = torch.cat([left_unit_indices, field_indices.unsqueeze(0).expand(batch_size, -1)], dim=1)

        right_unit_indices = torch.where(
            right_monster_mask,
            monster_indices.expand(batch_size, -1),
            torch.full((batch_size, self.monster_count), -1, device=left_counts.device, dtype=torch.long)
        )
        right_unit_indices = torch.cat([right_unit_indices, field_indices.unsqueeze(0).expand(batch_size, -1)], dim=1)

        # 获取嵌入
        left_embeddings = self.unit_embed(left_unit_indices)
        right_embeddings = self.unit_embed(right_unit_indices)

        # 价值嵌入
        left_values = self.value_ffn(left_embeddings)
        right_values = self.value_ffn(right_embeddings)

        # 逐层处理
        for i in range(self.num_layers):
            # 友方注意力
            left_friend_attn, _ = self.friend_attentions[i](left_values, left_values, left_values)
            left_values = self.norm[i](left_values + left_friend_attn)
            left_values = left_values + self.friend_ffn[i](left_values)

            right_friend_attn, _ = self.friend_attentions[i](right_values, right_values, right_values)
            right_values = self.norm[i](right_values + right_friend_attn)
            right_values = right_values + self.friend_ffn[i](right_values)

            # 敌方注意力
            left_enemy_attn, _ = self.enemy_attentions[i](left_values, right_values, right_values)
            left_values = self.norm[i](left_values + left_enemy_attn)
            left_values = left_values + self.enemy_ffn[i](left_values)

            right_enemy_attn, _ = self.enemy_attentions[i](right_values, left_values, left_values)
            right_values = self.norm[i](right_values + right_enemy_attn)
            right_values = right_values + self.enemy_ffn[i](right_values)

        # 聚合
        left_agg = left_values.mean(dim=1)
        right_agg = right_values.mean(dim=1)
        combined = torch.cat([left_agg, right_agg], dim=1)

        # 输出
        out = self.fc(combined)
        return torch.sigmoid(out)
