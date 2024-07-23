import torch
import torch.nn as nn


def conv1d_relu(in_c: int, out_c: int, kernel: int, stride: int = 1, padding='valid'):
    return nn.Sequential(
        nn.Conv1d(in_channels=in_c, out_channels=out_c, kernel_size=kernel, stride=stride, padding=padding),
        nn.ReLU(inplace=True)
    )


class Attention(nn.Module):
    def __init__(self, step_dim=2498, features_dim=1200, bias=True):
        super(Attention, self).__init__()
        self.step_dim = step_dim
        self.features_dim = features_dim
        self.bias = bias

        self.weight = nn.Parameter(torch.empty(features_dim, ))  # (1200, )
        if self.bias:
            self.b = nn.Parameter(torch.empty(step_dim, ))  # (2498, )
        else:
            self.b = None

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias:
            nn.init.zeros_(self.b)


class RHAS_classifier(nn.Module):
    def __init__(self):
        super(RHAS_classifier, self).__init__()
        # input shape: (bs, n_channels, seq_len) = (batch, 4, 10000)    L_out = (L_in - kernel_size) / stride + 1
        self.conv1 = conv1d_relu(4, 256, 5)  # (b, 256, 9996)
        self.conv2 = conv1d_relu(256, 512, 3)  # (b, 512, 9994)
        self.conv3 = conv1d_relu(512, 768, 3)  # (b, 768, 9992)
        self.pool = nn.MaxPool1d(kernel_size=4)  # (b, 768, 2498)
        self.lstm = nn.LSTM(input_size=768, hidden_size=1200, num_layers=1, batch_first=True, bidirectional=True)  # (b, 768, 2498) => (b, 2498, 1200)
        self.attention = Attention()  # (b, 1200)
        self.fc = nn.Linear(1200, 5)

    def forward(self, x, mask=None):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)

        # reshape for LSTM: (batch, n_channels, seq_len) -> (batch, seq_len, n_channels)
        x = x.permute(0, 2, 1)

        # LSTM expects (batch, seq_len, features)
        x, _ = self.lstm(x)

        # x: (batch_size, step_dim, features_dim) = (b, 2498, 1200)     weight = (1200)
        eij = torch.matmul(x, self.attention.weight)  # (b, 2498)

        if self.attention.bias is not None:
            eij = eij + self.attention.b

        eij = torch.tanh(eij)

        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / (torch.sum(a, dim=1, keepdim=True) + 1e-7)

        a = a.unsqueeze(-1)  # (b, 2498, 1)
        weighted_input = x * a  # (b, 2498, 1200)

        x = torch.sum(weighted_input, dim=1)  # (b, 1200)
        x = self.fc(x)

        return x
