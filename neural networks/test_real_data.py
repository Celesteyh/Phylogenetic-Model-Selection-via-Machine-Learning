import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import numpy as np
import resnet_nolstm_20_440
import conv_attention
import random
import conv
import pandas as pd



class MyDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data_files = os.listdir(data_path)
        # randomly select 1638 files
        # self.data_files = random.sample(os.listdir(data_path), 1638)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        feature_name = self.data_files[index]
        initial_feature = np.load(os.path.join(self.data_path, feature_name))
        np.random.seed(723)
        indices = np.random.choice(initial_feature.shape[0], size=625, replace=False)
        feature = torch.tensor(initial_feature[indices, :440].reshape((25, 25, 440)), dtype=torch.float32).permute(2, 0, 1)
        feature = torch.tensor(initial_feature.reshape((25, 25, 440)), dtype=torch.float32).permute(2, 0, 1)
        label = torch.tensor(int(feature_name[0]), dtype=torch.long)  # extract the label from the file name
        return feature, label


# create dataset and dataloader
valid_data_path = '/scratch/dx61/yd7308/real_feature_remove/real_feature_remove/'
valid_dataset = MyDataset(valid_data_path)
data_loader = DataLoader(valid_dataset, batch_size=40, shuffle=False)

# load the model
path = '/scratch/dx61/yd7308/saved_models/best_model_conv_625_mimicked_specified.pth'
device = torch.device('cuda')
model = conv.Simple_Conv()
# model = conv_attention.Simple_Attention1()
# model = resnet_nolstm_20_440.resnet18_builder()
model = model.to(device)
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load(path))
model.eval()

# get the accuracy
correct_1 = 0
correct_2 = 0
wrong_top_2 = []
total = 0
results_95 = []

with torch.no_grad():
    for data, labels in data_loader:
        data = data.to(device)
        labels = labels.to(device)
        pred = model(data)
        _, pred_label_1 = torch.max(pred, 1)
        _, pred_label_2 = torch.topk(pred, 2, 1)  # (batch_size, 2)
        for i in range(labels.size(0)):
            sorted_probs, sorted_indices = torch.sort(pred[i], descending=True)
            cumulative_prob = 0.0
            num_results = 0
            for prob in sorted_probs:
                cumulative_prob += prob.item()
                num_results += 1
                if cumulative_prob >= 0.95:
                    break
            results_95.append(num_results)
            if pred_label_2[i][0] != labels[i] and pred_label_2[i][1] == labels[i]:
                wrong_top_2.append((labels[i].item(), pred_label_2[i][0].item()))
        correct_1 += (pred_label_1 == labels).sum().item()
        correct_2 += (pred_label_2 == labels.unsqueeze(1)).any(1).sum().item()
        total += labels.size(0)

# df = pd.DataFrame(wrong_top_2, columns=['True Label', 'Wrong Predicted Label (Top-1)'])
# df.to_csv('/scratch/dx61/yd7308/wrong_top2_conv_b_mimicked_2345.csv', index=False)

# print the number of each element in results_95
print(pd.Series(results_95).value_counts())
print(correct_1 / total * 100)
print(correct_2 / total * 100)
