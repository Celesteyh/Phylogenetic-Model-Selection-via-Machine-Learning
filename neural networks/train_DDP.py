import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
import os
import argparse
import Encoder
import numpy as np
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
import early_stopping


def ddp_setup():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    print(f"Process {local_rank} assigned to GPU {local_rank}")
    init_process_group(backend="nccl")


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_data: DataLoader,
            vali_data: DataLoader,
            optimizer: torch.optim.Optimizer,
            loss_fn: torch.nn.Module,
            scaler: GradScaler,
            early_stopping: early_stopping.EarlyStopping,
            lr_scheduler: ReduceLROnPlateau,
            save_every: int,
            snapshot_path: str,
            best_model_path: str,
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.vali_data = vali_data
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scaler = scaler
        self.early_stopping = early_stopping
        self.lr_scheduler = lr_scheduler
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        self.best_model_path = best_model_path
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_train_batch(self, source, targets, batch_index, record, total_loss):
        self.optimizer.zero_grad()
        with autocast():
            pred = self.model(source)
            loss = self.loss_fn(pred, targets)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        total_loss += loss.item()
        if batch_index % 100 == 99 and self.gpu_id == 0:
            record.write(f"Loss: {total_loss / 100:.4f}\n")
            total_loss = 0

        _, pred_label = torch.max(pred, 1)
        n_correct = (pred_label == targets).long().sum().item()
        n_total = targets.size(0)
        return n_correct, n_total, total_loss

    def _run_vali_batch(self, source, targets):
        pred = self.model(source)
        _, pred_label = torch.max(pred, 1)
        n_correct = (pred_label == targets).long().sum().item()
        n_total = targets.size(0)
        return n_correct, n_total

    def _run_epoch(self, epoch, best_accy, record, lr_changed_flag, stop_flag):
        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']} in GPU {self.gpu_id}")

        if self.gpu_id == 0:
            record.write(f"Epoch {epoch + 1}\n")
            record.write(f"Learning rate: {self.optimizer.param_groups[0]['lr']}\n")

        n_correct_train, n_total_train, n_correct_vali, n_total_vali = 0, 0, 0, 0
        total_loss = 0  # training loss for 100 batches

        self.train_data.sampler.set_epoch(epoch)
        self.model.train()
        for batch_index, data in enumerate(self.train_data):
            source, targets = data
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            correct_train, total_train, total_loss = self._run_train_batch(source, targets, batch_index, record,
                                                                           total_loss)
            n_correct_train += correct_train
            n_total_train += total_train

        acc_tensor_train = torch.tensor([n_correct_train, n_total_train], device=self.gpu_id)
        dist.all_reduce(acc_tensor_train, op=dist.ReduceOp.SUM)
        global_correct_train, global_total_train = acc_tensor_train.tolist()
        global_accuracy_train = global_correct_train / global_total_train

        self.model.eval()
        with torch.no_grad():
            for source, targets in self.vali_data:
                source = source.to(self.gpu_id)
                targets = targets.to(self.gpu_id)
                correct_vali, total_vali = self._run_vali_batch(source, targets)
                n_correct_vali += correct_vali
                n_total_vali += total_vali

        acc_tensor_vali = torch.tensor([n_correct_vali, n_total_vali], device=self.gpu_id)
        dist.all_reduce(acc_tensor_vali, op=dist.ReduceOp.SUM)
        global_correct_vali, global_total_vali = acc_tensor_vali.tolist()
        global_accuracy_vali = global_correct_vali / global_total_vali

        if self.gpu_id == 0:
            record.write(f"Training Accuracy: {100 * global_accuracy_train}\n")
            record.write(f"Validation Accuracy: {100 * global_accuracy_vali}\n")
            record.write("-------------------------------------------------\n")
            record.flush()
        if global_accuracy_vali > best_accy:
            best_accy = global_accuracy_vali
            if self.gpu_id == 0:
                torch.save(self.model.module.state_dict(), self.best_model_path)

        old_lr = self.optimizer.param_groups[0]['lr']
        self.lr_scheduler.step(global_accuracy_vali)
        new_lr = self.optimizer.param_groups[0]['lr']
        if old_lr != new_lr and self.gpu_id == 0:
            lr_changed_flag += 1

        if self.early_stopping.step(global_accuracy_vali) and self.gpu_id == 0:
            stop_flag += 1

        dist.broadcast(lr_changed_flag, src=0)
        dist.broadcast(stop_flag, src=0)
        return lr_changed_flag, stop_flag, best_accy

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int, record):
        if self.gpu_id == 0:
            record.write('----------Training started----------\n')

        best_accy = 0.0
        lr_changed_flag = torch.tensor(0, device=self.gpu_id)
        stop_flag = torch.tensor(0, device=self.gpu_id)

        for epoch in range(self.epochs_run, max_epochs):
            lr_changed_flag, stop_flag, best_accy = self._run_epoch(epoch, best_accy, record, lr_changed_flag, stop_flag)
            print(f"lr_changed_flag: {lr_changed_flag.item()} | stop_flag: {stop_flag.item()} in GPU {self.gpu_id}")

            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)

            if lr_changed_flag.item() == 1:
                # reset the early stopping
                self.early_stopping.reset()
                stop_flag.zero_()
                lr_changed_flag.zero_()

            if stop_flag.item() == 1:
                if self.gpu_id == 0:
                    record.write(f"Stop at Epoch {epoch}\n")
                break

        if self.gpu_id == 0:
            record.write(f"Best accuracy: {best_accy * 100:.2f}%\n")
            record.write('----------Training finished----------\n')


class MyDataset(Dataset):
    def __init__(self, data_path_1, data_path_2):
        self.data_path_1 = data_path_1
        self.data_path_2 = data_path_2
        self.data_files = os.listdir(data_path_1)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        feature_name = self.data_files[index]
        initial_feature_1 = np.load(os.path.join(self.data_path_1, feature_name)).astype(np.float32)
        initial_feature_2 = np.load(os.path.join(self.data_path_2, feature_name)).astype(np.float32)
        initial_feature = np.concatenate((initial_feature_1, initial_feature_2), axis=1)
        feature = torch.from_numpy(initial_feature)
        label = torch.tensor(int(feature_name[1]), dtype=torch.long)
        return feature, label


def load_train_objs(train_data_path_1: str, val_data_path_1: str, train_data_path_2: str, val_data_path_2: str,
                    weight_decay: float = 0.0001, learning_rate: float = 0.001):
    train_set = MyDataset(train_data_path_1, train_data_path_2)
    vali_set = MyDataset(val_data_path_1, val_data_path_2)

    model = Encoder.EncoderBlock()
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=weight_decay, lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    scaler = GradScaler()
    es = early_stopping.EarlyStopping(mode='max', min_delta=0, patience=10, percentage=False)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5)

    return train_set, vali_set, model, optimizer, loss_fn, scaler, es, lr_scheduler


def prepare_dataloader(dataset: Dataset, batch_size: int, shuffle: bool):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset, shuffle=shuffle)
    )


def main(save_every: int, total_epochs: int, batch_size: int):
    ddp_setup()
    try:
        train_path_1 = '/scratch/dx61/yd7308/RHAS_I_embedding_train/'
        vali_path_1 = '/scratch/dx61/yd7308/RHAS_I_embedding_vali/'
        train_path_2 = '/scratch/dx61/yd7308/RHAS_I_feature_2features_transformer_train/'
        vali_path_2 = '/scratch/dx61/yd7308/RHAS_I_feature_2features_transformer_vali/'
        snapshot_path = "/scratch/dx61/yd7308/snapshot/snapshot_embedding_22_new.pth"
        best_model_path = "/scratch/dx61/yd7308/saved_models/best_model_RHAS_I_embedding_22_new.pth"
        record_path = "/scratch/dx61/yd7308/record_RHAS_I_embedding_22_new.txt"

        train_dataset, vali_dataset, model, optimizer, loss_fn, scaler, es, lr_scheduler = load_train_objs(
            train_path_1,
            vali_path_1,
            train_path_2,
            vali_path_2)
        train_data = prepare_dataloader(train_dataset, batch_size, True)
        vali_data = prepare_dataloader(vali_dataset, batch_size, False)

        with open(record_path, "w") as record:
            trainer = Trainer(model, train_data, vali_data, optimizer, loss_fn, scaler, es, lr_scheduler, save_every, snapshot_path, best_model_path)
            trainer.train(total_epochs, record)
            print("Finish training...")
    except Exception as e:
        print(f"An exception occurred: {e}")
    finally:
        destroy_process_group()
        print("Process group destroyed. Exiting.")


if __name__ == "__main__":
    main(5, 100, 64)
