# encoding=utf-8
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.layer1 = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, bidirectional=True, dropout=0.5)
        self.layer2 = nn.Linear(self.hidden_dim*2, self.output_dim)

    def forward(self, x):
        # s, b, dim
        x, hn = self.layer1(x)
        s, b, h = x.size()
        x = x.view(s*b, h)
        x = self.layer2(x)
        # s, b, outputsize
        x = x.view(s, b, -1)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.layer1 = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, bidirectional=True, dropout=0.5)
        self.layer2 = nn.Sequential(
            nn.Linear(self.hidden_dim*2, self.output_dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        # s, b, dim
        x, hn = self.layer1(x)
        s, b, h = x.size()
        x = x.view(s*b, h)
        x = self.layer2(x)
        # s, b, outputsize
        x = x.view(s, b, -1)
        return x

class GRUDIS(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GRUDIS, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.layer1 = nn.GRU(self.input_dim, self.hidden_dim, self.num_layers, bidirectional=True, dropout=0.5)
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim*2, self.output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # s, b, dim
        x, hn = self.layer1(x)
        s, b, h = x.size()
        x = x.view(s * b, h)
        x = self.layer2(x)
        # s, b, outputsize
        x = x.view(s, b, -1)
        return x

class LinearClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LinearClassifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layer1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.layer2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.layer3 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.layer3(x)


class CNNDIS(nn.Module):
    def __init__(self):
        super(CNNDIS, self).__init__()
        # 第一对卷积-池化层,输入大小：100*1
        self.layer1 = nn.Sequential(
            # 输出:10*46*1
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(10, 1), stride=(2, 1)),
            nn.ReLU(inplace=True),
            # 输出：10*19*1
            nn.MaxPool2d(kernel_size=(10, 1), stride=(2, 1))
        )
        # 第二队卷积-池化层
        self.layer2 = nn.Sequential(
            # 输出：5*9*1
            nn.Conv2d(in_channels=10, out_channels=5, kernel_size=(3, 1), stride=(2, 1)),
            nn.ReLU(inplace=True),
            # 输出：5*4*1
            nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1))
        )
        # 全连接层
        self.layer3 = nn.Sequential(
            # 20*50
            nn.Linear(5*4*1, 50),
            # 50*50
            nn.Linear(50, 50),
            # 50*1
            nn.Linear(50, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        output = self.layer3(x)
        return output
