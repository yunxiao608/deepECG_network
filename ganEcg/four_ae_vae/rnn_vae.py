# encoding=utf-8
from __future__ import division
import torch
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
from torch import nn
from helpfunction.helper import listsumavg
import pickle
import csv
from Config import opt

with open('pickle/pointx.pickle', 'rb') as f:
    pointlistx = pickle.load(f)

# convert to lead
for i in range(len(pointlistx)):
    pointlistx[i] = pointlistx[i]/200

listlength = len(pointlistx)

LATENT_DIM = 1


class Encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(Encoder, self).__init__()
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


class Decoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(Decoder, self).__init__()
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

        return mu + sigma * Variable(std_z, requires_grad=False).cuda()

    def forward(self, x):
        h_enc = self.encoder(x)
        z = self._sample_latent(h_enc)
        return self.decoder(z)




def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)



encoder = Encoder(input_dim=1, hidden_dim=50, output_dim=1, num_layers=2)
decoder = Decoder(input_dim=LATENT_DIM, hidden_dim=50, output_dim=1, num_layers=2)
vae = VAE(encoder, decoder)
if torch.cuda.is_available():
    vae = vae.cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(vae.parameters(), lr=0.000001)
l = None

NUM = int(listlength/opt.Seq_Len/opt.BATCH_SIZE)
Filename = 'resultdata/rnn_vae.csv'
outputFile = open(Filename, 'w')
outputWriter = csv.writer(outputFile)
outputWriter.writerow(['Epoch', 'Loss'])
for epoch in range(500):
    loss_list = []
    for batch in range(NUM):
        inputs = torch.randn(opt.BATCH_SIZE, opt.Seq_Len, 1)
        for i in range(opt.BATCH_SIZE):
            for j in range(opt.Seq_Len):
                    inputs[i][j][0] = pointlistx[batch *opt.Seq_Len*opt.BATCH_SIZE+i*opt.Seq_Len+j]
        inputs = Variable(inputs.view(opt.Seq_Len, opt.BATCH_SIZE, 1).cuda())
        dec = vae(inputs)
        ll = latent_loss(vae.z_mean, vae.z_sigma)
        criterion.cuda()
        loss = criterion(dec, inputs) + ll
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        l = loss.data[0].item()
        print 'epoch:{}, batch:{}, loss:{}'.format(epoch, batch, l)
        loss_list.append(l)
    loss_avg = listsumavg(loss_list)
    outputWriter.writerow([epoch, loss_avg])
outputFile.close()
torch.save(vae.state_dict(), 'modelsaved/rnn_vae.pkl')