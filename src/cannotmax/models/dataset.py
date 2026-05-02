"""
Dataset for Arknights battle data.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from cannotmax.config import FIELD_FEATURE_COUNT, MONSTER_COUNT

TOTAL_FEATURE_COUNT = (MONSTER_COUNT + FIELD_FEATURE_COUNT) * 2


class ArknightsDataset(Dataset):
    """Dataset for Arknights battle data."""

    def __init__(self, csv_file: str, max_value: int | None = None):
        data = pd.read_csv(csv_file, header=None, skiprows=1)

        expected_columns = TOTAL_FEATURE_COUNT + 2  # +2 for Result and ImgPath
        if data.shape[1] != expected_columns:
            raise ValueError(
                f"Column count mismatch! Expected {expected_columns}, got {data.shape[1]}"
            )

        data = data.iloc[:, 0 : TOTAL_FEATURE_COUNT + 1]
        features = data.iloc[:, :-1].values.astype(np.float32)
        labels = data.iloc[:, -1].map({"L": 0, "R": 1}).values
        labels = np.where((labels != 0) & (labels != 1), 0, labels).astype(np.float32)

        # 分割双方单位和场地特征
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

        self.left_signs = torch.from_numpy(left_signs)
        self.right_signs = torch.from_numpy(right_signs)
        self.left_counts = torch.from_numpy(left_counts)
        self.right_counts = torch.from_numpy(right_counts)
        self.labels = torch.from_numpy(labels).float()

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
