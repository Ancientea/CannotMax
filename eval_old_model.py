"""旧版多模态模型加载 & 评估 — best_model_loss.pth"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# ── 模型定义 ──
class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(5, 192, kernel_size=16, stride=16)
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, 192))

class TransformerWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=192, nhead=6, dim_feedforward=768, batch_first=True, norm_first=True)
            for _ in range(6)
        ])

class ImageViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.transformer = TransformerWrapper()
        self.norm = nn.LayerNorm(192)
        self.head = nn.Sequential(nn.Linear(192, 128), nn.LayerNorm(128))

    def forward(self, x):
        x = self.patch_embed.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.patch_embed.pos_embed
        for layer in self.transformer.layers:
            x = layer(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)

class OldModel(nn.Module):
    """旧版多模态模型 — ViT图像 + 怪物嵌入 + 跨模态注意力"""
    def __init__(self):
        super().__init__()
        E = 128
        self.unit_embed = nn.Embedding(60, E)
        self.image_encoder = ImageViT()
        self.value_ffn = nn.Sequential(nn.Linear(E, 256), nn.ReLU(), nn.Linear(256, E))
        self.img_norm1 = nn.LayerNorm(E)
        self.img_norm2 = nn.LayerNorm(E)
        self.norm1 = nn.ModuleList([nn.LayerNorm(E) for _ in range(3)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(E) for _ in range(3)])
        self.norm3 = nn.ModuleList([nn.LayerNorm(E) for _ in range(3)])
        self.norm4 = nn.ModuleList([nn.LayerNorm(E) for _ in range(3)])
        self.img_cross_attn = nn.MultiheadAttention(E, num_heads=4, batch_first=True)
        self.enemy_attentions = nn.ModuleList([nn.MultiheadAttention(E, 4, batch_first=True) for _ in range(3)])
        self.friend_attentions = nn.ModuleList([nn.MultiheadAttention(E, 4, batch_first=True) for _ in range(3)])

        def make_ffn():
            return nn.Sequential(nn.Linear(E, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, E))
        self.enemy_ffn = nn.ModuleList([make_ffn() for _ in range(3)])
        self.friend_ffn = nn.ModuleList([make_ffn() for _ in range(3)])
        self.final_norm = nn.LayerNorm(E)
        self.fc = nn.Sequential(nn.Linear(E, 256), nn.ReLU(), nn.Linear(256, 1), nn.Sigmoid())

    def forward(self, img, enemy_ids, friend_ids):
        # 图像编码
        img_feat = self.image_encoder(img)          # [B, 128]
        img_feat = self.img_norm1(img_feat)

        # 怪物嵌入
        e = self.unit_embed(enemy_ids)               # [B, Ne, 128]
        f = self.unit_embed(friend_ids)              # [B, Nf, 128]

        # 值变换
        e = self.value_ffn(e)
        f = self.value_ffn(f)

        # 跨模态注意力
        img_q = img_feat.unsqueeze(1)                # [B, 1, 128]
        img_out, _ = self.img_cross_attn(img_q, e, e)
        img_feat = img_out.squeeze(1)
        img_feat = self.img_norm2(img_feat)

        # 3 层交替处理
        for i in range(3):
            # Enemy
            e = e + self.enemy_attentions[i](self.norm1[i](e), self.norm1[i](e), self.norm1[i](e))[0]
            e = e + self.enemy_ffn[i](self.norm2[i](e))
            # Friend
            f = f + self.friend_attentions[i](self.norm3[i](f), self.norm3[i](f), self.norm3[i](f))[0]
            f = f + self.friend_ffn[i](self.norm4[i](f))

        # 聚合
        out = e.mean(dim=1) + f.mean(dim=1) + img_feat
        out = self.final_norm(out)
        return self.fc(out).squeeze(-1)


if __name__ == "__main__":
    import pandas as pd
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 加载模型
    sd = torch.load("models/best_model_loss.pth", map_location=device, weights_only=False)
    model = OldModel().to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    print("模型加载成功")

    # 加载数据
    csv_path = "arknights_filtered.csv" if len(sys.argv) < 2 else sys.argv[1]
    print(f"数据: {csv_path}")
    df = pd.read_csv(csv_path, header=None)
    data = df.values  # numpy array
    n_rows, n_cols = data.shape
    n_monsters = (n_cols - 1) // 2  # 左右各 n 只怪物 + 1 标签
    print(f"样本: {n_rows}, 怪物数: {n_monsters}")

    # 只用前 60 只怪物 (旧模型是 60 怪)
    n_use = min(n_monsters, 60)
    if n_monsters > 60:
        print(f"注意: 数据有 {n_monsters} 怪, 只用前 {n_use} 只")

    # 提取特征和标签
    left_counts = data[:, :n_use].astype(np.int64)   # [N, 60]
    right_counts = data[:, n_monsters:n_monsters+n_use].astype(np.int64)
    labels = (data[:, -1] == 'R').astype(np.float32)  # R=1, L=0

    @torch.no_grad()
    def evaluate_batch(batch_left, batch_right, batch_labels):
        """将counts转换为ID序列并评估"""
        B = batch_left.shape[0]
        # 展开为 ID 序列 (重复每个monster id count次)
        enemy_ids_list = []
        friend_ids_list = []
        for i in range(B):
            e_ids = np.repeat(np.arange(n_use), batch_left[i])
            f_ids = np.repeat(np.arange(n_use), batch_right[i])
            # 截断到合理长度
            enemy_ids_list.append(torch.from_numpy(e_ids[:200]).to(device))
            friend_ids_list.append(torch.from_numpy(f_ids[:200]).to(device))
        
        # Pad to same length
        max_e = max(len(x) for x in enemy_ids_list)
        max_f = max(len(x) for x in friend_ids_list)
        if max_e == 0: max_e = 1
        if max_f == 0: max_f = 1
        
        e_padded = torch.zeros(B, max_e, dtype=torch.long, device=device)
        f_padded = torch.zeros(B, max_f, dtype=torch.long, device=device)
        for i in range(B):
            e_padded[i, :len(enemy_ids_list[i])] = enemy_ids_list[i]
            f_padded[i, :len(friend_ids_list[i])] = friend_ids_list[i]
        
        # 零图输入
        img = torch.zeros(B, 5, 128, 128, device=device)
        
        outputs = model(img, e_padded, f_padded)
        preds = (outputs > 0.5).float()
        correct = (preds == batch_labels.to(device)).sum().item()
        return correct, B

    # 逐批评估
    batch_size = 128
    total_correct = 0
    total_samples = 0
    pbar = tqdm(range(0, n_rows, batch_size), desc="评估中")
    for start in pbar:
        end = min(start + batch_size, n_rows)
        bl = left_counts[start:end]
        br = right_counts[start:end]
        blab = labels[start:end]
        c, b = evaluate_batch(torch.from_numpy(bl), torch.from_numpy(br), torch.from_numpy(blab))
        total_correct += c
        total_samples += b
        pbar.set_postfix(acc=f"{total_correct/total_samples*100:.1f}%")

    acc = total_correct / total_samples * 100
    print(f"\n准确率: {acc:.2f}% ({total_correct}/{total_samples})")
    print(f"注意: 图像输入为零图, 实际准确率仅供参考")
