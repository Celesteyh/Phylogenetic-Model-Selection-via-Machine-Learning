import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import RHAS
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
import early_stopping_2
import time

start_time = time.time()

RHAS_map = {'G4': 1, 'R2': 2, 'R3': 3, 'R4': 4}


class MyDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data_files = os.listdir(data_path)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        feature_name = self.data_files[index]
        initial_feature = np.load(os.path.join(self.data_path, feature_name))
        feature = torch.tensor(initial_feature, dtype=torch.float32).permute(1, 0)  # (10000, 4) => (4, 10000)
        RHAS = feature_name[:-4].split('_')[0].split('+')[-1].split('{')[0]
        if RHAS in list(RHAS_map.keys()):
            label = torch.tensor(RHAS_map[RHAS], dtype=torch.long)
        else:
            label = torch.tensor(0, dtype=torch.long)
        return feature, label


def train_model(dataloader, model, loss_fn, optimizer, device, epoch):
    """ return the average loss and accuracy of this epoch """
    # running_loss: average loss every 100 batches
    # total_loss: average loss over this epoch
    # n_correct: number of correct predictions
    # n_total: number of all predictions
    running_loss, total_loss, n_correct, n_total = 0.0, 0.0, 0, 0
    n_batches = len(dataloader)
    # For one batch of data
    for batch_index, data in enumerate(dataloader):
        feature, label = data

        feature = feature.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        with autocast():
            pred = model(feature)  # (B, 7)
            loss = loss_fn(pred, label)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        total_loss += loss.item()
        # Report every 100 iterations(No.100, No.200, ...)
        if batch_index % 100 == 99:
            # Print with 4 decimal places
            # print(' epoch {} loss: {:.4f}'.format(epoch, running_loss / 100))
            record.write(' epoch {} loss: {:.4f}\n'.format(epoch, running_loss / 100) + '\n')
            running_loss = 0.0

        _, pred_label = torch.max(pred, 1)

        # pred_label == label is a boolean tensor
        n_correct += (pred_label == label).long().sum().item()
        # label is a 1D tensor and label.size(0) returns its length, it equals to batch size in most cases
        n_total += label.size(0)

    accuracy = n_correct / n_total
    # print('Epoch {} Accuracy: {:.2f}'.format(epoch, 100 * accuracy))
    record.write('Epoch {} Accuracy: {:.2f}\n'.format(epoch, 100 * accuracy) + '\n')
    return total_loss / n_batches, accuracy


def test_model(dataloader, model, loss_fn, device):
    """ return the average loss and accuracy"""
    # number of batches
    n_batches = len(dataloader)
    # test_loss: average test loss over all validation/test data
    # n_correct: number of correct predictions
    # n_total: number of all predictions
    test_loss, n_correct, n_total = 0.0, 0, 0
    # Disabled gradient calculation
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

    # Calculate average loss and accuracy
    test_loss /= n_batches
    accuracy = n_correct / n_total
    # print("Test Error: \nAccuracy: {:.2f}, Average loss: {:.4f} \n".format(100 * accuracy, test_loss))
    record.write("Test Error: \nAccuracy: {:.2f}, Average loss: {:.4f} \n".format(100 * accuracy, test_loss) + '\n')
    return test_loss, accuracy


# parameters
train_data_path = '/scratch/dx61/yd7308/simulated_train_RHAS_feature/'
valid_data_path = '/scratch/dx61/yd7308/simulated_vali_RHAS_feature/'
batch_size = 40
num_workers = 4
learning_rate = 0.001
weight_decay = 0.0001
epochs = 200
device = torch.device('cuda')

# create dataset and dataloader
train_dataset = MyDataset(train_data_path)
valid_dataset = MyDataset(valid_data_path)
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

# create model
model = RHAS.RHAS_classifier()
# loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam([{'params': conv_weights, 'weight_decay': weight_decay},
#                               {'params': non_conv_weights, 'weight_decay': 0.0}], lr=learning_rate)
optimizer = torch.optim.AdamW(model.parameters(), weight_decay=weight_decay, lr=learning_rate)
# amp: automatic mixed precision
scaler = GradScaler()
# early stopping
es = early_stopping_2.EarlyStopping(mode='max', min_delta=0, patience=8, percentage=False)
# learning rate scheduler
lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3)

# Save the tensorboard result to path
writer = SummaryWriter(log_dir='/scratch/dx61/yd7308/summary/RHAS')

# train and evaluate the model
model.to(device)
model = nn.DataParallel(model)

record = open('record_RHAS_10000_simulated.txt', 'w')

best_accuracy = 0
# print('----------Training started----------')
record.write('----------Training started----------\n')
for epoch in range(1, epochs + 1):
    # Train the model using training set
    # Set the model in training mode
    model.train()
    # Get the average training loss of this epoch
    train_loss, train_accuracy = train_model(train_loader, model, loss_fn, optimizer, device, epoch)
    # Save the point to tensorboard
    # Training loss vs epochs
    writer.add_scalar('train_loss', train_loss, epoch)
    # Training accuracy vs epochs
    writer.add_scalar('train_accuracy', train_accuracy, epoch)
    # save model weights
    # save_path = '/scratch/dx61/yd7308/saved_models/train_{:03d}.pth'.format(epoch)
    # torch.save(model.state_dict(), save_path)

    # Evaluate the model using validation set
    # Set the model in evaluation mode
    model.eval()
    valid_loss, valid_accuracy = test_model(valid_loader, model, loss_fn, device)
    # Save the point to tensorboard
    # Validation loss vs epochs
    writer.add_scalar('valid_loss', valid_loss, epoch)
    # Validation accuracy vs epochs
    writer.add_scalar('valid_accuracy', valid_accuracy, epoch)
    # Save the model with best validation accuracy
    if valid_accuracy > best_accuracy:
        save_path = '/scratch/dx61/yd7308/saved_models/best_model_RHAS_10000_simulated.pth'
        torch.save(model.state_dict(), save_path)
        best_accuracy = valid_accuracy
    record.flush()
    # learning rate scheduler
    lr_scheduler.step(valid_accuracy)
    # early stopping
    if es.step(valid_accuracy):
        break
    record.write('learning rate: ' + str(optimizer.param_groups[0]["lr"]) + '\n')
# print('best accuracy: ' + best_accuracy)
record.write('best accuracy: ' + str(best_accuracy * 100) + '\n')
writer.close()
# print('----------Training finished----------')
record.write('----------Training finished----------\n')
end_time = time.time()
record.write(f"Time taken: {end_time - start_time} seconds")
record.close()

