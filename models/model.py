import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import FIELD_FEATURE_COUNT, MONSTER_COUNT


class PhasePreservingLinearMHA(nn.Module):
    """
    线性注意力：
    线性注意力使用 φ(x)=elu(x)+1 作为核特征映射，将 O(L²d) 降为 O(Ld²)，
    引入强低秩平滑先验，在小数据下防止过拟合到少数关键交互。
    相位保持：
    相位保持将注意力输出投影回 query 方向，只保留沿 query 的分量，
    强制模型以查询为中心聚合信息，是一种额外的几何正则化。
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % (2 * num_heads) == 0, "embed_dim 必须能被 2 * num_heads 整除（相位保持要求）"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.half_head_dim = self.head_dim // 2

        # Q/K/V 投影及输出投影
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        for p in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            if p.bias is not None:
                nn.init.zeros_(p.bias)

    @staticmethod
    def feature_map(x):
        """
        φ(x) = relu(x) + 1，非负，数值稳定
        此外，这里用 relu 而非常规的 elu，尽可能保证映射不变形
        """
        return F.relu(x) + 1.0

    def forward(self, query, key, value, key_padding_mask=None):
        B, Lq, _ = query.shape
        Lk = key.shape[1]

        # 1. 线性投影 + 多头重塑
        q = self.q_proj(query).reshape(B, Lq, self.num_heads, self.head_dim)
        k = self.k_proj(key).reshape(B, Lk, self.num_heads, self.head_dim)
        v = self.v_proj(value).reshape(B, Lk, self.num_heads, self.head_dim)

        q = q.permute(0, 2, 1, 3)   # (B, H, Lq, d)
        k = k.permute(0, 2, 1, 3)   # (B, H, Lk, d)
        v = v.permute(0, 2, 1, 3)

        # 2. 特征映射到非负空间
        q_prime = self.feature_map(q)
        k_prime = self.feature_map(k)

        # 3. 处理 mask：无效位置（如填充怪物）的 value 和 k 特征置 0，使其不参与注意力聚合
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(-1).to(v.dtype)  # (B, 1, Lk, 1)
            v = v * mask
            k_prime = k_prime * mask
        # 若无 mask，则无需特殊处理

        # 4. 线性注意力核心计算
        #    attn_out = φ(q) (φ(k)^T v) / (φ(q) (φ(k)^T 1) + eps)
        k_trans = k_prime.transpose(-2, -1)              # (B, H, d, Lk)
        kv = torch.matmul(k_trans, v)                    # (B, H, d, d)
        k_sum = k_trans.sum(dim=-1, keepdim=True)        # (B, H, d, 1) 即 φ(k)^T 1

        numerator = torch.matmul(q_prime, kv)            # (B, H, Lq, d)
        denominator = torch.matmul(q_prime, k_sum) + 1e-8 # (B, H, Lq, 1)

        attn_out = numerator / denominator
        # 防御性处理：应对极端数值（如全 mask 行或 fp16 下溢）
        attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=1e4, neginf=-1e4)
        attn_out = self.dropout(attn_out)

        # 5. 相位保持：将注意力输出投影到 query 的复数域方向上
        #    复数表示允许模型在“幅度”和“相位”两个维度上保留信息，
        #    而相位保持强制输出只取 query 方向的分量，等价于一种归一化且避免正交噪声。
        attn_out_c = attn_out.reshape(B, self.num_heads, Lq, self.half_head_dim, 2)
        q_c = q.reshape(B, self.num_heads, Lq, self.half_head_dim, 2)

        # 计算 attn_out 在 q 上的投影系数
        dot = (attn_out_c * q_c).sum(dim=(-2, -1))         # (B, H, Lq)
        q_norm_sq = (q_c * q_c).sum(dim=(-2, -1)) + 1e-8
        q_norm_sq = torch.clamp(q_norm_sq, min=1e-8)         # 防除零
        coef = dot / q_norm_sq
        coef = torch.clamp(coef, min=-10.0, max=10.0)       # 限制投影系数范围，稳定训练

        # 重建沿 q 方向的输出
        out_c = coef.unsqueeze(-1).unsqueeze(-1) * q_c   # (B, H, Lq, hd, 2)
        out_c = out_c.permute(0, 2, 1, 3, 4).reshape(B, Lq, self.num_heads * self.half_head_dim, 2)
        out = out_c.reshape(B, Lq, self.embed_dim)

        return self.out_proj(out)


class ResidualFFN(nn.Module):
    """
    标准残差前馈网络：Linear → ReLU → Linear → Dropout，然后加上残差连接。
    提供非线性特征变换能力，同时通过残差保持梯度流动。
    """
    def __init__(self, embed_dim, dropout=0.1, hidden_factor=2):
        super().__init__()
        hidden_dim = embed_dim * hidden_factor
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.linear1.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.linear2.weight, a=math.sqrt(5))
        if self.linear1.bias is not None:
            nn.init.zeros_(self.linear1.bias)
        if self.linear2.bias is not None:
            nn.init.zeros_(self.linear2.bias)

    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.activation(out)
        out = self.linear2(out)
        out = self.dropout(out)
        return residual + out


class UnitAwareTransformer(nn.Module):
    def __init__(self, num_units, embed_dim=256, num_heads=4, num_layers=3, dropout=0.3):
        super().__init__()
        self.num_units = num_units
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # 嵌入层：为每种怪物和场地特征学习一个稠密向量
        self.unit_embed = nn.Embedding(num_units, embed_dim)
        nn.init.normal_(self.unit_embed.weight, mean=0.0, std=0.02)

        # 输入侧的残差 FFN：为每个单位提供非线性微扰，
        # 使基础战斗力不再只是数量的线性函数，可以捕捉规模非线性（如边际递减效应）。
        self.value_ffn = ResidualFFN(embed_dim, dropout)

        # 敌方/友方注意力模块和对应的 FFN（逐层独立参数）
        self.enemy_attentions = nn.ModuleList()
        self.friend_attentions = nn.ModuleList()
        self.enemy_ffn = nn.ModuleList()
        self.friend_ffn = nn.ModuleList()

        for _ in range(num_layers):
            self.enemy_attentions.append(PhasePreservingLinearMHA(embed_dim, num_heads, dropout))
            self.enemy_ffn.append(ResidualFFN(embed_dim, dropout))
            self.friend_attentions.append(PhasePreservingLinearMHA(embed_dim, num_heads, dropout))
            self.friend_ffn.append(ResidualFFN(embed_dim, dropout))

    def forward(self, left_signs, left_counts, right_signs, right_counts):
        # 提取 TopK 特征（怪物 + 场地）
        # 每方最多 3 个怪物 + 1 个场地特征，k = 4 可覆盖全部
        k = min(4, left_counts.shape[1])  # 确保 k 不超过实际特征数
        left_values, left_indices = torch.topk(left_counts, k=k, dim=1)
        right_values, right_indices = torch.topk(right_counts, k=k, dim=1)

        # 嵌入层，base 保留单体的原始特征
        # AI 可解释性结果表明，怪物单体的原始特征对模型预测很重要
        left_base = self.unit_embed(left_indices)   # (B, k, embed_dim)
        right_base = self.unit_embed(right_indices)

        # 直接用模长表示战斗力
        # 早期尝试 PINN 结构时，线性项权重占比过高且泛化性差。
        # 因此改用线性 + 微扰设计：基础战斗力由模长表示，微扰来自 FFN
        left_feat = left_base * left_values.unsqueeze(-1)
        right_feat = right_base * right_values.unsqueeze(-1)

        # FFN 提供非线性微扰，使模型能捕捉非线性的战斗力增长
        left_feat = self.value_ffn(left_feat)
        right_feat = self.value_ffn(right_feat)

        # 生成 mask (B, k)，使用 0.1 阈值防止浮点误差
        left_mask = left_values > 0.1
        right_mask = right_values > 0.1

        # 动态获取 Batch Size，供后续 unflatten 使用
        B = left_feat.size(0)

        for i in range(self.num_layers):
            # 敌方注意力
            # 设计 4 组交互：(left_feat, right_base)、(left_base, right_feat)、
            # (right_feat, left_base)、(right_base, left_feat)。实验表明这种交叉交互优于全量组合
            # 利用批处理，合并左右之间的交互，从而加速运算
            # 旧代码的 1×2 组交互，也可通过批处理合并计算以加速
            # l 和 r 互看，base 和 feat 互看
            q_enemy = torch.cat([left_feat, left_base, right_feat, right_base], dim=0)
            k_enemy = torch.cat([right_base, right_feat, left_base, left_feat], dim=0)
            # mask 重复 2 次
            mask_enemy = torch.cat([right_mask.repeat(2, 1), left_mask.repeat(2, 1)], dim=0)

            out_enemy = self.enemy_attentions[i](
                query=q_enemy, key=k_enemy, value=k_enemy,
                key_padding_mask=mask_enemy,
            )
            # 拆分回 4 组结果，对应地加到各自的特征上
            out_enemy = out_enemy.unflatten(0, (4, B))
            # 等效于：left_feat += attn(left_base, right_feat) + attn(left_feat, right_base)
            left_feat = left_feat + out_enemy[:2].sum(dim=0)
            right_feat = right_feat + out_enemy[2:].sum(dim=0)

            # FFN 后处理
            left_feat = self.enemy_ffn[i](left_feat)
            right_feat = self.enemy_ffn[i](right_feat)

            # ---- 友方注意力 ----
            # 同阵营内 base 与 feat 的交互，捕捉兵种间的协同或互补关系
            q_friend = torch.cat([left_feat, left_base, right_feat, right_base], dim=0)
            k_friend = torch.cat([left_base, left_feat, right_base, right_feat], dim=0)
            mask_friend = torch.cat([left_mask.repeat(2, 1), right_mask.repeat(2, 1)], dim=0)

            out_friend = self.friend_attentions[i](
                query=q_friend, key=k_friend, value=k_friend,
                key_padding_mask=mask_friend,
            )
            out_friend = out_friend.unflatten(0, (4, B))
            left_feat = left_feat + out_friend[:2].sum(dim=0)
            right_feat = right_feat + out_friend[2:].sum(dim=0)

            left_feat = self.friend_ffn[i](left_feat)
            right_feat = self.friend_ffn[i](right_feat)

            # 注意：此处不进行 LayerNorm，以保留嵌入向量的模长作为战斗力度量。
            # 对向量进行归一化会破坏模长，而模长直接与数量关联，是重要的线性基础信息。
            # left_feat = self.norm[i](left_feat)
            # right_feat = self.norm[i](right_feat)

        # 战斗力评估：先计算各特征的 L2 范数，再求和，避免先求和再取模（会丢失非线性信息）
        # 注意力模块已处理了特征间非线性交互
        # 这里不进行如下的全连接输出，会过参数化降低泛化性能
        # L = self.output_ffn(left_feat).squeeze(-1) * left_mask
        # R = self.output_ffn(right_feat).squeeze(-1) * right_mask
        L_norms = torch.norm(left_feat, p=2, dim=-1)
        R_norms = torch.norm(right_feat, p=2, dim=-1)
        L = (L_norms * left_mask).sum(dim=1)
        R = (R_norms * right_mask).sum(dim=1)

        return torch.sigmoid(R - L)
