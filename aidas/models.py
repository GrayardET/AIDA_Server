import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

class biLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv1D = nn.Conv1d(input_size, 32, kernel_size = self.kernel_size, stride=1) #output (N, 32, 28)
        self.lstm1 = nn.LSTM(32, hidden_size, bidirectional=True)
        # self.lstm2 = nn.LSTM(32, 128, bidirectional=True)
        # self.lstm3 = nn.LSTM(32, 128, bidirectional=True)
        # self.lstm4 = nn.LSTM(32, 128, bidirectional=True)
        self.drop_out = nn.Dropout(0.15)
        self.dense = nn.Linear(256, 32)
        self.dense2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.conv1D(x)
        # print(f'shape after conv1D: {x.shape}')
        x = x.permute(0, 2, 1)
        # print(f'shape after permute: {x.shape}')
        out, (_, c_n) = self.lstm1(x)
        out = out[:, -1, :]
        # print(f'shape after LSTM: {out.shape}')
        x = self.drop_out(out)
        x = self.dense(x)
        x = self.dense2(x)
        # print(f'shape after all dense layers: {x.shape}')
        return x

class Model(torch.nn.Module):
    pass

