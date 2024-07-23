import torch
import torch.nn as nn


class Simple_Attention1(nn.Module):
    def __init__(self):
        super(Simple_Attention1, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=440, num_heads=8, batch_first=True)  # B * 512 * 440
        self.fc = nn.Linear(440, 7)

    def forward(self, x):
        x = self.attention(x, x, x)[0]
        x = torch.mean(x, dim=1)
        x = self.fc(x)
        return x


class Simple_Attention2(nn.Module):
    def __init__(self):
        super(Simple_Attention2, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)  # B * 440 * 512
        self.fc = nn.Linear(512, 7)

    def forward(self, x):
        x = self.attention(x, x, x)[0]
        x = torch.mean(x, dim=1)
        x = self.fc(x)
        return x
