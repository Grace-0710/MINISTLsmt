import torch
import torch.nn as nn


class SequenceClassifier(nn.Module):

    def __init__(
        self,
        input_size, # fc-784차원, cnn-28*28, rnn-한 low씩 28차원의 low가 28개 들어옴
        hidden_size,
        output_size, #28
        n_layers=4, #lstm은 gradient vanising을 해결하지만 timp stemp에 관한것만 처리하므로 층갯수는 평균 4개 적당
        dropout_p=.2, #drop out은 층마다 들어가
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        super().__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=True,
        )
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size * 2),
            nn.Linear(hidden_size * 2, output_size),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        # |x| = (batch_size, h, w) -> gray scale이니까 차원은 1
        # 이미지이긴하지만 h=time step, w=입력의 크기(vector size)
        z, _ = self.rnn(x)
        # |z| = (batch_size, h, hidden_size * 2)
        z = z[:, -1]
        # |z| = (batch_size, hidden_size * 2)
        y = self.layers(z)
        # |y| = (batch_size, output_size)

        return y
