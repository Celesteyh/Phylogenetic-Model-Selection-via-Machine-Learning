import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
import protFinder
import RHASFinder
import early_stopping
import time

start_time = time.time()


def parse_args():
    """
    Paths and hyperparameters
    """
    parser = argparse.ArgumentParser(description='Training neural networks')
    parser.add_argument('--train_data_path', type=str, required=True, help='Path to training data')
    parser.add_argument('--valid_data_path', type=str, required=True, help='Path to validation data')
    parser.add_argument('--log_dir', type=str, help='Directory for TensorBoard logs')
    parser.add_argument('--save_model_dir', type=str, required=True, help='Directory for saved networks')
    parser.add_argument('--record_file', type=str, help='File that records the training process')

    parser.add_argument('--network_id', type=int, default=0, help='0 for protFinder, 1 for RHASFinder')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loader')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay for optimizer')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')

    return parser.parse_args()


class MyDataset(Dataset):
    def __init__(self, data_path, network):
        self.data_path = data_path
        self.data_files = os.listdir(data_path)
        self.network = network

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        feature_name = self.data_files[index]
        initial_feature = np.load(os.path.join(self.data_path, feature_name))
        if self.network == 0:  # protFinder
            feature = torch.tensor(initial_feature, dtype=torch.float32).permute(2, 0, 1)
            label = torch.tensor(int(feature_name[0]), dtype=torch.long)
        else:  # RHASFinder
            feature = torch.tensor(initial_feature, dtype=torch.float32)
            label = torch.tensor(min(2, int(feature_name[1])), dtype=torch.long)  # R2, R3, R4 -> R
        return feature, label


def train_model(dataloader, model, loss_fn, optimizer, device, epoch, scaler, record):
    running_loss, total_loss, n_correct, n_total = 0.0, 0.0, 0, 0
    n_batches = len(dataloader)

    for batch_index, data in enumerate(dataloader):
        feature, label = data

        feature = feature.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        with autocast():
            pred = model(feature)
            loss = loss_fn(pred, label)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        total_loss += loss.item()

        if batch_index % 100 == 99:
            record.write('epoch {} loss: {:.4f}\n'.format(epoch, running_loss / 100) + '\n')
            running_loss = 0.0

        _, pred_label = torch.max(pred, 1)

        n_correct += (pred_label == label).long().sum().item()
        n_total += label.size(0)

    avg_loss = total_loss / n_batches
    accuracy = n_correct / n_total
    record.write('Epoch {} Accuracy: {:.2f}\n'.format(epoch, 100 * accuracy) + '\n')
    return avg_loss, accuracy


def test_model(dataloader, model, loss_fn, device, record):
    test_loss, n_correct, n_total = 0.0, 0, 0
    n_batches = len(dataloader)

    with torch.no_grad():
        for data in dataloader:
            feature, label = data

            feature = feature.to(device)
            label = label.to(device)

            pred = model(feature)

            loss = loss_fn(pred, label)
            test_loss += loss.item()

            _, pred_label = torch.max(pred, 1)
            n_correct += (pred_label == label).long().sum().item()
            n_total += label.size(0)

    test_loss /= n_batches
    accuracy = n_correct / n_total
    record.write("Test Error: \nAccuracy: {:.2f}, Average loss: {:.4f} \n".format(100 * accuracy, test_loss) + '\n')
    return test_loss, accuracy


def main():
    args = parse_args()

    train_data_path = args.train_data_path
    valid_data_path = args.valid_data_path
    log_dir = args.log_dir
    saved_model_dir = args.save_model_dir

    network_id = args.network_id
    batch_size = 64 if network_id == 0 else 128
    num_workers = args.num_workers
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    epochs = args.epochs
    device = torch.device('cuda')

    # initialize dataset and dataloader
    train_dataset = MyDataset(train_data_path, network_id)
    valid_dataset = MyDataset(valid_data_path, network_id)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    # initialize network
    model = protFinder.Conv_SEB() if network_id == 0 else RHASFinder.Conv2d()

    # loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=weight_decay, lr=learning_rate)

    # amp: automatic mixed precision
    scaler = GradScaler()

    # early stopping
    es = early_stopping.EarlyStopping(mode='max', min_delta=0, patience=10, percentage=False)

    # learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5)

    # save the tensorboard result
    writer = SummaryWriter(log_dir=log_dir)

    # train and evaluate the model
    model.to(device)
    model = nn.DataParallel(model)

    # record the training process
    record = open(args.record_file, 'w')

    best_accuracy = 0

    record.write('----------Training started----------\n')

    for epoch in range(1, epochs + 1):
        model.train()
        # average training loss & accuracy of this epoch
        train_loss, train_accuracy = train_model(train_loader, model, loss_fn, optimizer, device, epoch, scaler, record)

        # training loss vs epochs
        writer.add_scalar('train_loss', train_loss, epoch)
        # training accuracy vs epochs
        writer.add_scalar('train_accuracy', train_accuracy, epoch)

        model.eval()
        valid_loss, valid_accuracy = test_model(valid_loader, model, loss_fn, device, record)

        writer.add_scalar('valid_loss', valid_loss, epoch)
        writer.add_scalar('valid_accuracy', valid_accuracy, epoch)

        # save the model with best validation accuracy
        if valid_accuracy > best_accuracy:
            save_path = os.path.join(saved_model_dir, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            best_accuracy = valid_accuracy
        record.flush()

        lr_scheduler.step(valid_accuracy)

        if es.step(valid_accuracy):
            break

        record.write('learning rate: ' + str(optimizer.param_groups[0]["lr"]) + '\n')

    record.write('best accuracy: ' + str(best_accuracy * 100) + '\n')
    writer.close()
    record.write('----------Training finished----------\n')

    end_time = time.time()
    record.write(f"Time taken: {end_time - start_time} seconds")
    record.close()


if __name__ == '__main__':
    main()

