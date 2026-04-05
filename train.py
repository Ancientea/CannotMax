from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import torch.optim as optim
class ArknightsDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file, header=None)
        features = data.iloc[:, :-1].values.astype(np.float32)
        labels = data.iloc[:, -1].map({'L': 0, 'R': 1}).values
        labels = np.where((labels != 0) & (labels != 1), 0, labels).astype(np.float32)

        # 分离左右双方并保留符号信息
        self.left_signs = np.sign(features[:, :26])
        self.right_signs = np.sign(features[:, 26:])
        self.left_counts = np.abs(features[:, :26])
        self.right_counts = np.abs(features[:, 26:])
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.left_signs[idx]),
            torch.tensor(self.left_counts[idx]),
            torch.tensor(self.right_signs[idx]),
            torch.tensor(self.right_counts[idx]),
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )


class UnitAwareTransformer(nn.Module):
    def __init__(self, num_units=27, embed_dim=128, num_heads=8, num_layers=3):
        super().__init__()
        self.num_units = num_units
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # 嵌入层
        self.unit_embed = nn.Embedding(num_units, embed_dim, padding_idx=0)
        nn.init.normal_(self.unit_embed.weight, mean=0.0, std=0.02)

        self.value_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

        # 注意力层与FFN
        self.enemy_attentions = nn.ModuleList()
        self.friend_attentions = nn.ModuleList()
        self.enemy_ffn = nn.ModuleList()
        self.friend_ffn = nn.ModuleList()

        self.enemy_norm1 = nn.ModuleList()
        self.friend_norm1 = nn.ModuleList()
        self.enemy_norm2 = nn.ModuleList()
        self.friend_norm2 = nn.ModuleList()

        for _ in range(num_layers):
            # 敌方注意力层
            self.enemy_attentions.append(
                nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.2)
            )
            self.enemy_ffn.append(nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(embed_dim * 4, embed_dim)
            ))

            # 友方注意力层
            self.friend_attentions.append(
                nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.2)
            )
            self.friend_ffn.append(nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(embed_dim * 4, embed_dim)
            ))

            # 初始化注意力层参数
            nn.init.xavier_uniform_(self.enemy_attentions[-1].in_proj_weight)
            nn.init.xavier_uniform_(self.friend_attentions[-1].in_proj_weight)

        # 全连接输出层
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, 1)
        )

    def forward(self, left_sign, left_count, right_sign, right_count):
        # 提取Top3兵种特征
        left_values, left_indices = torch.topk(left_count, k=3, dim=1)
        right_values, right_indices = torch.topk(right_count, k=3, dim=1)

        # 嵌入
        left_feat = self.unit_embed(left_indices)  # (B, 3, 128)
        right_feat = self.unit_embed(right_indices)  # (B, 3, 128)

        embed_dim = self.embed_dim

        # 前x维不变，后y维 *= 数量
        left_feat = torch.cat([
            left_feat[..., :embed_dim // 2],  # 前x维
            left_feat[..., embed_dim // 2:] * left_values.unsqueeze(-1)  # 后y维乘数量，后y维可直接代替统计量，同时避免引入额外统计量参数
        ], dim=-1)
        right_feat = torch.cat([
            right_feat[..., :embed_dim // 2],
            right_feat[..., embed_dim // 2:] * right_values.unsqueeze(-1)
        ], dim=-1)

        # FFN
        left_feat = left_feat + self.value_ffn(left_feat)
        right_feat = right_feat + self.value_ffn(right_feat)

        # 生成mask (B, 3)
        left_mask = (left_values > 0)
        right_mask = (right_values > 0)

        for i in range(self.num_layers):
            # 敌方注意力
            delta_left, _ = self.enemy_attentions[i](
                query=left_feat,
                key=right_feat,
                value=right_feat,
                key_padding_mask=~right_mask,
                need_weights=False
            )
            delta_right, _ = self.enemy_attentions[i](
                query=right_feat,
                key=left_feat,
                value=left_feat,
                key_padding_mask=~left_mask,
                need_weights=False
            )
            # 残差连接（现在可以归一化了，但没什么意义，反而训练更慢）
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
                need_weights=False
            )
            delta_right, _ = self.friend_attentions[i](
                query=right_feat,
                key=right_feat,
                value=right_feat,
                key_padding_mask=~right_mask,
                need_weights=False
            )
            # 残差连接
            left_feat = left_feat + delta_left
            right_feat = right_feat + delta_right

            # FFN
            left_feat = left_feat + self.friend_ffn[i](left_feat)
            right_feat = right_feat + self.friend_ffn[i](right_feat)

        # 输出战斗力
        L = self.fc(left_feat).squeeze(-1) * left_mask
        R = self.fc(right_feat).squeeze(-1) * right_mask
        # 计算战斗力差输出概率，'L': 0, 'R': 1，R大于L时输出大于0.5
        output = torch.sigmoid(R.sum(1) - L.sum(1))

        return output

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for ls, lc, rs, rc, labels in train_loader:
        ls, lc, rs, rc, labels = [x.to(device) for x in (ls, lc, rs, rc, labels)]

        optimizer.zero_grad()
        outputs = model(ls, lc, rs, rc).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        preds = (outputs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(train_loader), 100 * correct / total


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for ls, lc, rs, rc, labels in data_loader:
            ls, lc, rs, rc, labels = [x.to(device) for x in (ls, lc, rs, rc, labels)]
            outputs = model(ls, lc, rs, rc).squeeze()

            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(data_loader), 100 * correct / total

def main():
    config = {
        'batch_size': 128,
        'embed_dim': 128,
        'n_layers': 4,
        'lr': 3e-4,
        'epochs': 150,
        'seed': 42
    }

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = ArknightsDataset('arknights.csv')
    train_indices, val_indices = train_test_split(
        range(len(dataset)), test_size=0.1, random_state=config['seed']
    )

    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_indices),
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=32
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, val_indices),
        batch_size=config['batch_size'],
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=32
    )

    model = UnitAwareTransformer(
        num_units=26,
        embed_dim=config['embed_dim'],
        num_heads=8,
        num_layers=config['n_layers']
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    best_acc = 0
    for epoch in range(config['epochs']):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device)
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            # 保存完整模型（方便部署）
            torch.save(model, 'best_model_full.pth')

        print(f"Epoch {epoch + 1}/{config['epochs']}")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
        print("-" * 40)

if __name__ == "__main__":
    main()