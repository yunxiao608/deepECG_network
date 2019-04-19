# encoding=utf-8
import torch.nn as nn
import torch
from Config import opt
from torch.autograd import Variable
import numpy as np

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
        # 第一对卷积-池化层,输入大小：3120*1
        self.layer1 = nn.Sequential(
            # 输出:10*601*1
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(120, 1), stride=(5, 1)),
            nn.ReLU(inplace=True),
            # 输出：10*186*1
            nn.MaxPool2d(kernel_size=(46, 1), stride=(3, 1))
        )
        # 第二队卷积-池化层
        self.layer2 = nn.Sequential(
            # 输出：5*51*1
            nn.Conv2d(in_channels=10, out_channels=5, kernel_size=(36, 1), stride=(3, 1)),
            nn.ReLU(inplace=True),
            # 输出：5*10*1
            nn.MaxPool2d(kernel_size=(24, 1), stride=(3, 1))
        )
        # 全连接层
        self.layer3 = nn.Sequential(
            # 50*75
            nn.Linear(5*10*1, 25),
            # 75*50
            nn.Linear(25, 25),
            # 50*1
            nn.Linear(25, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        output = self.layer3(x)
        return output


# rnn_encoder_decoder
class RNN_Encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(RNN_Encoder, self).__init__()
        self.E_in = input_dim
        self.E_h = hidden_dim
        self.E_out = output_dim
        self.num_layers = num_layers
        self.layer1 = nn.RNN(self.E_in, self.E_h, self.num_layers, bidirectional=True)
        self.layer2 = nn.Linear(self.E_h*2, self.E_out)

    def forward(self, x):
        # s, b, dim
        x, hn = self.layer1(x)
        s, b, h = x.size()
        x = x.view(s * b, h)
        x = self.layer2(x)
        # s, b, outputsize
        x = x.view(s, b, -1)
        return x


class RNN_Decoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(RNN_Decoder, self).__init__()
        self.D_in = input_dim
        self.D_h = hidden_dim
        self.D_out = output_dim
        self.num_layers = num_layers
        self.layer1 = nn.RNN(self.D_in, self.D_h, self.num_layers, bidirectional=True)
        self.layer2 = nn.Linear(self.D_h*2, self.D_out)

    def forward(self, y):
        # s, b, dim
        y, hn = self.layer1(y)
        s, b, h = y.size()
        y = y.view(s * b, h)
        y = self.layer2(y)
        # s, b, outputsize
        y = y.view(s, b, -1)
        return y


# lstm_encoder_decoder
class LSTM_Encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTM_Encoder, self).__init__()
        self.E_in = input_dim
        self.E_h = hidden_dim
        self.E_out = output_dim
        self.num_layers = num_layers
        self.layer1 = nn.LSTM(self.E_in, self.E_h, self.num_layers, bidirectional=True)
        self.layer2 = nn.Linear(self.E_h*2, self.E_out)

    def forward(self, x):
        # s, b, dim
        x, hn = self.layer1(x)
        s, b, h = x.size()
        x = x.view(s * b, h)
        x = self.layer2(x)
        # s, b, outputsize
        x = x.view(s, b, -1)
        return x


class LSTM_Decoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTM_Decoder, self).__init__()
        self.D_in = input_dim
        self.D_h = hidden_dim
        self.D_out = output_dim
        self.num_layers = num_layers
        self.layer1 = nn.LSTM(self.D_in, self.D_h, self.num_layers, bidirectional=True)
        self.layer2 = nn.Linear(self.D_h*2, self.D_out)

    def forward(self, y):
        # s, b, dim
        y, hn = self.layer1(y)
        s, b, h = y.size()
        y = y.view(s * b, h)
        y = self.layer2(y)
        # s, b, outputsize
        y = y.view(s, b, -1)
        return y

# autocoder
class autocoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(autocoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


LATENT_DIM = 1
# vae
class VAE(torch.nn.Module):

    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._enc_mu = torch.nn.Linear(1, LATENT_DIM)
        self._enc_log_sigma = torch.nn.Linear(1, LATENT_DIM)

    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(
            np.random.normal(
                0, 1, size=sigma.size())).float()

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * Variable(std_z, requires_grad=False)

    def forward(self, x):
        h_enc = self.encoder(x)
        z = self._sample_latent(h_enc)
        return self.decoder(z)







