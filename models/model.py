import torch
import torch.nn as nn
import torch.nn.functional as F
from config import FIELD_FEATURE_COUNT, MONSTER_COUNT


class UnitAwareTransformer(nn.Module):
    def __init__(self, num_units, embed_dim=256, num_heads=4, num_layers=3, dropout=0.3):
        super().__init__()
        # num_units = 怪物种类数 + 场地特征种类数
        self.num_units = num_units
        self.monster_count = MONSTER_COUNT
        self.field_count = FIELD_FEATURE_COUNT
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout

        # 嵌入层
        self.unit_embed = nn.Embedding(num_units, embed_dim)
        nn.init.normal_(self.unit_embed.weight, mean=0.0, std=0.02)

        self.value_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        # 注意力层与 FFN
        self.enemy_attentions = nn.ModuleList()
        self.friend_attentions = nn.ModuleList()
        self.enemy_ffn = nn.ModuleList()
        self.friend_ffn = nn.ModuleList()

        # 门控投影层
        self.enemy_gate_proj_q = nn.ModuleList()
        self.enemy_gate_proj_k = nn.ModuleList()
        self.friend_gate_proj_q = nn.ModuleList()
        self.friend_gate_proj_k = nn.ModuleList()

        for _ in range(num_layers):
            # 敌方注意力层
            self.enemy_attentions.append(
                nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
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
                nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
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

            # 门控机制：通过减轻不重要交互的干扰，使模型更专注于克制与互补关系。
            # 实验表明该设计略微降低了测试集损失。
            # 初始门控权重为零，使 sigmoid 输出接近 0.5，避免早期训练对注意力施加过大压力。
            gate_q = nn.Linear(embed_dim, 1, bias=False)
            gate_k = nn.Linear(embed_dim, 1, bias=False)
            nn.init.zeros_(gate_q.weight)
            nn.init.zeros_(gate_k.weight)
            self.enemy_gate_proj_q.append(gate_q)
            self.enemy_gate_proj_k.append(gate_k)

            gate_q = nn.Linear(embed_dim, 1, bias=False)
            gate_k = nn.Linear(embed_dim, 1, bias=False)
            nn.init.zeros_(gate_q.weight)
            nn.init.zeros_(gate_k.weight)
            self.friend_gate_proj_q.append(gate_q)
            self.friend_gate_proj_k.append(gate_k)

    def _gated_attention(self, attention_module, q, k, v, key_padding_mask, gate_q_proj, gate_k_proj):
        embed_dim = attention_module.embed_dim
        num_heads = attention_module.num_heads
        head_dim = embed_dim // num_heads
        dropout_p = attention_module.dropout

        w = attention_module.in_proj_weight
        b = attention_module.in_proj_bias

        if b is not None:
            q_proj = F.linear(q, w[:embed_dim], b[:embed_dim])
            k_proj = F.linear(k, w[embed_dim:2*embed_dim], b[embed_dim:2*embed_dim])
            v_proj = F.linear(v, w[2*embed_dim:], b[2*embed_dim:])
        else:
            q_proj = F.linear(q, w[:embed_dim])
            k_proj = F.linear(k, w[embed_dim:2*embed_dim])
            v_proj = F.linear(v, w[2*embed_dim:])

        N, k_len, _ = q.shape
        q_mh = q_proj.view(N, k_len, num_heads, head_dim).transpose(1, 2)
        k_mh = k_proj.view(N, k_len, num_heads, head_dim).transpose(1, 2)
        v_mh = v_proj.view(N, k_len, num_heads, head_dim).transpose(1, 2)

        scale = head_dim ** -0.5
        attn_logits = torch.matmul(q_mh, k_mh.transpose(-2, -1)) * scale

        # 门控
        log_gate = self._apply_gate(q, k, gate_q_proj, gate_k_proj)  # (N, k, k)
        log_gate = log_gate.unsqueeze(1)  # (N, 1, k, k)
        attn_logits = attn_logits + log_gate

        # key_padding_mask
        if key_padding_mask is not None:
            expand_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (N, 1, 1, k)
            # 如果一行全为 False，softmax 会 NaN，用一个小值填充而非 -inf
            # 但先确保 -inf 不变，然后后处理检测全 mask 行
            attn_logits = attn_logits.masked_fill(~expand_mask, float('-inf'))

        # 防止整行全为 -inf 导致 softmax 输出 NaN
        # 计算每行最大值，若为 -inf，则说明全 mask，此时将 logits 置为 0，使权重均匀
        row_max = attn_logits.max(dim=-1, keepdim=True).values
        full_masked = row_max == float('-inf')
        if full_masked.any():
            attn_logits = attn_logits.masked_fill(full_masked, 0.0)  # 让 softmax 输出均匀分布

        # 数值稳定的 softmax
        attn_weights = torch.softmax(attn_logits, dim=-1)
        # 如果原来全 mask 的行，再次强制设为零（防止 softmax 后非零）
        if full_masked.any():
            attn_weights = attn_weights.masked_fill(full_masked, 0.0)

        attn_weights = F.dropout(attn_weights, p=dropout_p, training=self.training)
        attn_output = torch.matmul(attn_weights, v_mh)
        attn_output = attn_output.transpose(1, 2).contiguous().view(N, k_len, embed_dim)

        if attention_module.out_proj is not None:
            attn_output = attention_module.out_proj(attn_output)

        return attn_output

    def _apply_gate(self, q, k, gate_q_proj, gate_k_proj):
        # 采用 2N 自由度的低秩分解，当查询和键的特征匹配时，门控被激活
        # 可解释性示例：群攻属性匹配数量属性、法伤匹配法抗等场景下门控被激活
        s = gate_q_proj(q).squeeze(-1)      # (N, k)
        t = gate_k_proj(k).squeeze(-1)      # (N, k)
        raw_sum = s.unsqueeze(-1) + t.unsqueeze(-2)   # (N, k, k)
        # 裁剪和到合理范围，避免 sigmoid 饱和
        # raw_sum = torch.clamp(raw_sum, min=-10.0, max=10.0)
        gate = torch.sigmoid(raw_sum)
        log_gate = torch.log(gate + 1e-8)
        # log_gate = torch.clamp(log_gate, min=-20.0, max=0.0)   # 防止极端的负值
        return log_gate

    def forward(self, left_signs, left_counts, right_signs, right_counts):
        # 提取 TopK 特征（怪物 + 场地）
        # 每方最多 3 个怪物 + 1 个场地特征，k = 4 可覆盖全部
        k = min(4, left_counts.shape[1])  # 确保 k 不超过实际特征数
        left_values, left_indices = torch.topk(left_counts, k=k, dim=1)
        right_values, right_indices = torch.topk(right_counts, k=k, dim=1)

        # 嵌入层，base 保留单体的原始特征
        # AI 可解释性结果表明，怪物单体的原始特征对模型预测很重要
        left_base = self.unit_embed(left_indices)  # (B, k, embed_dim)
        right_base = self.unit_embed(right_indices)  # (B, k, embed_dim)

        # 直接用模长表示战斗力
        # 早期尝试 PINN 结构时，线性项权重占比过高且泛化性差。
        # 因此改用线性 + 微扰设计：基础战斗力由模长表示，微扰来自 FFN
        left_feat = left_base * left_values.unsqueeze(-1)
        right_feat = right_base * right_values.unsqueeze(-1)

        # FFN 提供非线性微扰，使模型能捕捉非线性的战斗力增长
        left_feat = left_feat + self.value_ffn(left_feat)
        right_feat = right_feat + self.value_ffn(right_feat)

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

            out_enemy = self._gated_attention(
                self.enemy_attentions[i], q_enemy, k_enemy, k_enemy,
                mask_enemy,
                self.enemy_gate_proj_q[i], self.enemy_gate_proj_k[i]
            )

            # 拆分为 4 个 (B, k, d) 张量，分别对应四组交互
            out_enemy = out_enemy.unflatten(0, (4, B))
            # 等效于：left_feat += attn(left_base, right_feat) + attn(left_feat, right_base)
            left_feat = left_feat + out_enemy[:2].sum(dim=0)
            right_feat = right_feat + out_enemy[2:].sum(dim=0)
            left_feat = left_feat + self.enemy_ffn[i](left_feat)
            right_feat = right_feat + self.enemy_ffn[i](right_feat)

            # 友方注意力
            q_friend = torch.cat([left_feat, left_base, right_feat, right_base], dim=0)
            k_friend = torch.cat([left_base, left_feat, right_base, right_feat], dim=0)
            mask_friend = torch.cat([left_mask.repeat(2, 1), right_mask.repeat(2, 1)], dim=0)

            out_friend = self._gated_attention(
                self.friend_attentions[i], q_friend, k_friend, k_friend,
                mask_friend,
                self.friend_gate_proj_q[i], self.friend_gate_proj_k[i]
            )

            out_friend = out_friend.unflatten(0, (4, B))
            left_feat = left_feat + out_friend[:2].sum(dim=0)
            right_feat = right_feat + out_friend[2:].sum(dim=0)
            left_feat = left_feat + self.friend_ffn[i](left_feat)
            right_feat = right_feat + self.friend_ffn[i](right_feat)
            # 不进行 LayerNorm：保留嵌入向量的模长信息作为战斗力度量，归一化会破坏模长的物理含义
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

        output = torch.sigmoid(R - L)
        return output
