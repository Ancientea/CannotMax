import time
from pathlib import Path

import torch

import lightning as L


class CompatibilityCheckpoint(L.Callback):
    def __init__(self, save_dir: str):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        metrics = trainer.callback_metrics
        val_acc = metrics.get("val_acc")
        val_loss = metrics.get("val_loss")

        if val_acc is not None:
            if not hasattr(self, "_best_acc") or val_acc > self._best_acc:
                self._best_acc = val_acc
                torch.save(
                    pl_module.model,
                    self.save_dir / "best_model_acc.pth",
                )

        if val_loss is not None:
            if not hasattr(self, "_best_loss") or val_loss < self._best_loss:
                self._best_loss = val_loss
                torch.save(
                    pl_module.model,
                    self.save_dir / "best_model_loss.pth",
                )

    def on_train_epoch_end(self, trainer, pl_module):
        torch.save(pl_module.model, self.save_dir / "best_model_full.pth")


class TrainingLogCallback(L.Callback):
    def __init__(self, total_epochs: int):
        super().__init__()
        self.total_epochs = total_epochs
        self._start_time = None
        self._epoch_start_time = None

    def on_train_start(self, trainer, pl_module):
        self._start_time = time.time()
        self._epoch_start_time = self._start_time

    def on_train_epoch_start(self, trainer, pl_module):
        current_epoch = trainer.current_epoch + 1
        print(f"Epoch {current_epoch}/{self.total_epochs}")

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        train_loss = metrics.get("train_loss", torch.tensor(0.0))
        train_acc = metrics.get("train_acc", torch.tensor(0.0))
        val_loss = metrics.get("val_loss", torch.tensor(float("inf")))
        val_acc = metrics.get("val_acc", torch.tensor(0.0))

        train_loss_val = train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss
        train_acc_val = train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc
        val_loss_val = val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss
        val_acc_val = val_acc.item() if isinstance(val_acc, torch.Tensor) else val_acc

        print(
            f"Train Loss: {train_loss_val:.4f} | Acc: {train_acc_val:.2f}% | "
            f"Val Loss: {val_loss_val:.4f} | Acc: {val_acc_val:.2f}%"
        )

        current_time = time.time()
        epoch_duration = current_time - self._epoch_start_time
        elapsed_time = current_time - self._start_time
        current_epoch = trainer.current_epoch + 1
        avg_epoch_time = elapsed_time / current_epoch
        estimated_total_time = avg_epoch_time * self.total_epochs
        remaining_time = estimated_total_time - elapsed_time

        print(
            f"Epoch Time: {epoch_duration:.2f}s | "
            f"Elapsed: {elapsed_time / 60:.2f}min | "
            f"Remaining: {remaining_time / 60:.2f}min | "
            f"Total: {estimated_total_time / 60:.2f}min"
        )

        self._epoch_start_time = current_time
