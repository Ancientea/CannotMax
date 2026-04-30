import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

import lightning as L

from config import FIELD_FEATURE_COUNT, MONSTER_COUNT

TOTAL_FEATURE_COUNT = (MONSTER_COUNT + FIELD_FEATURE_COUNT) * 2


def preprocess_data(csv_file):
    print(f"预处理数据文件: {csv_file}")

    data = pd.read_csv(csv_file, header=None, skiprows=1)
    print(f"原始数据形状: {data.shape}")

    expected_columns = TOTAL_FEATURE_COUNT + 2
    if data.shape[1] != expected_columns:
        print(f"数据列数不符！期望 {expected_columns} 列，实际 {data.shape[1]} 列")
        print(
            f"期望格式: {MONSTER_COUNT}(怪物L) + {FIELD_FEATURE_COUNT}(场地L) + {MONSTER_COUNT}(怪物R) + {FIELD_FEATURE_COUNT}(场地R) + 1(Result) + 1(ImgPath)"
        )
        raise Exception("数据格式不符")

    data = data.iloc[:, 0 : TOTAL_FEATURE_COUNT + 1]

    features = data.iloc[:, :-1]
    labels = data.iloc[:, -1]

    extreme_values = (np.abs(features) > 20).sum().sum()
    if extreme_values > 0:
        print(f"发现 {extreme_values} 个绝对值大于20的特征值")

    invalid_labels = labels.apply(lambda x: x not in ["L", "R"]).sum()
    if invalid_labels > 0:
        print(f"发现 {invalid_labels} 个无效标签")

    feature_min = features.min().min()
    feature_max = features.max().max()
    feature_mean = features.mean().mean()
    feature_std = features.std().mean()

    print(f"特征值范围: [{feature_min}, {feature_max}]")
    print(f"特征值平均值: {feature_mean:.4f}, 标准差: {feature_std:.4f}")

    return data.shape[1]


class ArknightsDataset(Dataset):
    def __init__(self, csv_file, max_value=None):
        data = pd.read_csv(csv_file, header=None, skiprows=1)
        expected_columns = TOTAL_FEATURE_COUNT + 2
        if data.shape[1] != expected_columns:
            print(f"数据列数不符！期望 {expected_columns} 列，实际 {data.shape[1]} 列")
            raise Exception("数据格式不符")
        data = data.iloc[:, 0 : TOTAL_FEATURE_COUNT + 1]
        features = data.iloc[:, :-1].values.astype(np.float32)
        labels = data.iloc[:, -1].map({"L": 0, "R": 1}).values
        labels = np.where((labels != 0) & (labels != 1), 0, labels).astype(np.float32)

        left_monster_end = MONSTER_COUNT
        left_field_end = MONSTER_COUNT + FIELD_FEATURE_COUNT
        right_monster_end = MONSTER_COUNT + FIELD_FEATURE_COUNT + MONSTER_COUNT
        right_field_end = (
            MONSTER_COUNT + FIELD_FEATURE_COUNT + MONSTER_COUNT + FIELD_FEATURE_COUNT
        )

        left_monster_features = features[:, :left_monster_end]
        left_field_features = features[:, left_monster_end:left_field_end]
        right_monster_features = features[:, left_field_end:right_monster_end]
        right_field_features = features[:, right_monster_end:right_field_end]

        left_counts = np.concatenate(
            [np.abs(left_monster_features), left_field_features], axis=1
        )
        right_counts = np.concatenate(
            [np.abs(right_monster_features), right_field_features], axis=1
        )
        left_signs = np.concatenate(
            [np.sign(left_monster_features), np.ones_like(left_field_features)], axis=1
        )
        right_signs = np.concatenate(
            [np.sign(right_monster_features), np.ones_like(right_field_features)],
            axis=1,
        )

        if max_value is not None:
            left_counts = np.clip(left_counts, 0, max_value)
            right_counts = np.clip(right_counts, 0, max_value)

        device = self._get_device()
        self.left_signs = torch.from_numpy(left_signs).to(device)
        self.right_signs = torch.from_numpy(right_signs).to(device)
        self.left_counts = torch.from_numpy(left_counts).to(device)
        self.right_counts = torch.from_numpy(right_counts).to(device)
        self.labels = torch.from_numpy(labels).float().to(device)

    @staticmethod
    def _get_device():
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        return torch.device("cpu")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.left_signs[idx],
            self.left_counts[idx],
            self.right_signs[idx],
            self.right_counts[idx],
            self.labels[idx],
        )


def stratified_random_split(dataset, test_size=0.1, seed=42):
    labels = dataset.labels
    device = ArknightsDataset._get_device()
    if str(device) != "cpu":
        labels = labels.cpu()
    labels = labels.numpy()

    indices = np.arange(len(labels))
    train_indices, val_indices = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=labels
    )
    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
    )


class ArknightsDataModule(L.LightningDataModule):
    def __init__(self, data_cfg: DictConfig, seed: int = 42):
        super().__init__()
        self.data_file = data_cfg.data_file
        self.batch_size = data_cfg.batch_size
        self.test_size = data_cfg.test_size
        self.max_feature_value = data_cfg.max_feature_value
        self.num_workers = data_cfg.num_workers
        self.seed = seed

        self.train_dataset = None
        self.val_dataset = None
        self._data_length = 0

    def prepare_data(self):
        preprocess_data(self.data_file)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            dataset = ArknightsDataset(
                self.data_file, max_value=self.max_feature_value
            )
            self._data_length = len(dataset)
            self.train_dataset, self.val_dataset = stratified_random_split(
                dataset, test_size=self.test_size, seed=self.seed
            )
            train_size = len(self.train_dataset)
            val_size = len(self.val_dataset)
            print(f"训练集大小: {train_size}, 验证集大小: {val_size}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
