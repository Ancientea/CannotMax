import torch
import torch.nn as nn
from src.core.config import FIELD_FEATURE_COUNT, MONSTER_COUNT


class UnitAwareTransformer(nn.Module):
    def __init__(
        self, num_units, embed_dim=256, num_heads=4, num_layers=4, dropout=0.3
    ):
        super().__init__()
        # num_units，包括怪物种类和场地特征种类
        # 怪物特征 + 场地特征 = 总特征数量
        self.num_units = num_units
        self.monster_count = MONSTER_COUNT
        self.field_count = FIELD_FEATURE_COUNT
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # 嵌入层
        self.unit_embed = nn.Embedding(num_units, embed_dim)
        nn.init.normal_(self.unit_embed.weight, mean=0.0, std=0.02)

        self.value_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        # 注意力层与FFN
        self.enemy_attentions = nn.ModuleList()
        self.friend_attentions = nn.ModuleList()
        self.enemy_ffn = nn.ModuleList()
        self.friend_ffn = nn.ModuleList()
        self.norm = nn.ModuleList()

        for _ in range(num_layers):
            # 敌方注意力层
            self.enemy_attentions.append(
                nn.MultiheadAttention(
                    embed_dim, num_heads, batch_first=True, dropout=dropout
                )
            )
            self.enemy_ffn.append(
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim * 2, embed_dim),
                )
            )

            # 友方注意力层
            self.friend_attentions.append(
                nn.MultiheadAttention(
                    embed_dim, num_heads, batch_first=True, dropout=dropout
                )
            )
            self.friend_ffn.append(
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim * 2, embed_dim),
                )
            )

            # 初始化注意力层参数
            nn.init.xavier_uniform_(self.enemy_attentions[-1].in_proj_weight)
            nn.init.xavier_uniform_(self.friend_attentions[-1].in_proj_weight)
            self.norm.append(nn.LayerNorm(embed_dim))

        # 全连接输出层
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, 1),
        )

    def forward(self, left_signs, left_counts, right_signs, right_counts):
        # 提取TopK特征（怪物 + 场地）
        # 由于现在包含场地特征，需要增加k值以确保重要特征不被遗漏
        # k=8 可以保证包含主要怪物和所有场地特征
        k = min(4, left_counts.shape[1])  # 确保k不超过实际特征数
        left_values, left_indices = torch.topk(left_counts, k=k, dim=1)
        right_values, right_indices = torch.topk(right_counts, k=k, dim=1)

        # 嵌入
        left_feat = self.unit_embed(left_indices)  # (B, k, 128)
        right_feat = self.unit_embed(right_indices)  # (B, k, 128)

        embed_dim = self.embed_dim

        # 前x维不变，后y维 *= 数量，但使用缩放后的值
        left_feat = torch.cat(
            [
                left_feat[..., : embed_dim // 2],  # 前x维
                left_feat[..., embed_dim // 2 :]
                * left_values.unsqueeze(-1),  # 后y维乘数量
            ],
            dim=-1,
        )
        right_feat = torch.cat(
            [
                right_feat[..., : embed_dim // 2],
                right_feat[..., embed_dim // 2 :] * right_values.unsqueeze(-1),
            ],
            dim=-1,
        )

        # FFN
        left_feat = left_feat + self.value_ffn(left_feat)
        right_feat = right_feat + self.value_ffn(right_feat)

        # 生成mask (B, k) 0.1防一手可能的浮点误差
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

            # 残差连接
            left_feat = left_feat + delta_left
            right_feat = right_feat + delta_right

            # FFN
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

            # 残差连接
            left_feat = left_feat + delta_left
            right_feat = right_feat + delta_right

            # FFN
            left_feat = left_feat + self.friend_ffn[i](left_feat)
            right_feat = right_feat + self.friend_ffn[i](right_feat)
            left_feat = self.norm[i](left_feat)
            right_feat = self.norm[i](right_feat)

        # 输出战斗力
        L = self.fc(left_feat).squeeze(-1) * left_mask
        R = self.fc(right_feat).squeeze(-1) * right_mask

        # 计算战斗力差输出概率，'L': 0, 'R': 1，R大于L时输出大于0.5
        output = torch.sigmoid(R.sum(1) - L.sum(1))

        return output
