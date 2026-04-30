import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Sequence, cast
from omegaconf import DictConfig

import lightning as L

from config import FIELD_FEATURE_COUNT, MONSTER_COUNT
from models.model import UnitAwareTransformer
from models.muon import get_muon_lion_optimizers


class ArknightsLightningModule(L.LightningModule):
    def __init__(self, model_cfg: DictConfig, trainer_cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(ignore=["model_cfg"])
        self.automatic_optimization = False
        self.max_epochs = int(trainer_cfg.epochs)

        total_units = MONSTER_COUNT + FIELD_FEATURE_COUNT
        self.model = UnitAwareTransformer(
            num_units=total_units,
            embed_dim=model_cfg.embed_dim,
            num_heads=model_cfg.num_heads,
            num_layers=model_cfg.num_layers,
            dropout=float(model_cfg.get("dropout", 0.3)),
        )
        self.criterion = nn.MSELoss()
        self.lr = trainer_cfg.lr
        self.lion_lr = trainer_cfg.lion_lr
        self.weight_decay = trainer_cfg.weight_decay

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @staticmethod
    def _check_input_validity(ls, lc, rs, rc):
        if (
            torch.isnan(ls).any()
            or torch.isnan(lc).any()
            or torch.isnan(rs).any()
            or torch.isnan(rc).any()
        ):
            print("警告: 输入数据包含NaN，跳过该批次")
            return False
        if (
            torch.isinf(ls).any()
            or torch.isinf(lc).any()
            or torch.isinf(rs).any()
            or torch.isinf(rc).any()
        ):
            print("警告: 输入数据包含Inf，跳过该批次")
            return False
        return True

    @staticmethod
    def _check_output_validity(outputs):
        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            print("警告: 模型输出包含NaN或Inf，跳过该批次")
            return None
        if (outputs < 0).any() or (outputs > 1).any():
            print("警告: 模型输出不在[0,1]范围内，进行修正")
            outputs = torch.clamp(outputs, 1e-7, 1 - 1e-7)
        return outputs

    def training_step(self, batch, batch_idx, *args, **kwargs):
        del batch_idx, args, kwargs
        ls, lc, rs, rc, labels = batch

        if not self._check_input_validity(ls, lc, rs, rc):
            return None

        if (labels < 0).any() or (labels > 1).any():
            print("警告: 标签值不在[0,1]范围内，进行修正")
            labels = torch.clamp(labels, 0, 1)

        try:
            optimizers = cast(Sequence[Any], self.optimizers())
            muon_opt, lion_opt = optimizers
            muon_opt.zero_grad()
            lion_opt.zero_grad()

            outputs = self.model(ls, lc, rs, rc).squeeze()

            outputs = self._check_output_validity(outputs)
            if outputs is None:
                return None

            loss = self.criterion(outputs.float(), labels.float())

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"警告: 损失值为 {loss.item()}, 跳过该批次")
                return None

            preds = (outputs > 0.5).float()
            acc = (preds == labels).float().mean() * 100

            self.manual_backward(loss)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            muon_opt.step()
            lion_opt.step()

            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

            return loss

        except RuntimeError as e:
            print(f"警告: 训练过程中出错 - {str(e)}")
            return None

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        del batch_idx, args, kwargs
        ls, lc, rs, rc, labels = batch

        if not self._check_input_validity(ls, lc, rs, rc):
            return None

        if (labels < 0).any() or (labels > 1).any():
            labels = torch.clamp(labels, 0, 1)

        try:
            outputs = self.model(ls, lc, rs, rc).squeeze()

            outputs = self._check_output_validity(outputs)
            if outputs is None:
                return None

            loss = self.criterion(outputs.float(), labels.float())

            if torch.isnan(loss) or torch.isinf(loss):
                return None

            preds = (outputs > 0.5).float()
            acc = (preds == labels).float().mean() * 100

            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

            return loss

        except RuntimeError as e:
            print(f"警告: 评估过程中出错 - {str(e)}")
            return None

    def configure_optimizers(self):
        muon_opt, lion_opt = get_muon_lion_optimizers(
            self.model, muon_lr=self.lr, lion_lr=self.lion_lr, weight_decay=self.weight_decay
        )
        scheduler_muon = optim.lr_scheduler.CosineAnnealingLR(
            muon_opt, T_max=self.max_epochs
        )
        scheduler_lion = optim.lr_scheduler.CosineAnnealingLR(
            lion_opt, T_max=self.max_epochs
        )
        return [muon_opt, lion_opt], [scheduler_muon, scheduler_lion]
