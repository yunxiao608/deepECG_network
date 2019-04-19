# encoding=utf-8
from __future__ import division
from Model import Generator
import torch
from torch.autograd import Variable
from Model import RNN_Decoder, RNN_Encoder, LSTM_Encoder, LSTM_Decoder, autocoder, VAE

G = Generator(input_dim=5, hidden_dim=50, output_dim=1, num_layers=2)

def listmerge(examlist):
    resultlist = []
    for i in range(len(examlist)):
        for j in range(len(examlist[i])):
            resultlist.append(examlist[i][j])
    return resultlist


def generateEcg(modelname):
    G.load_state_dict(torch.load('modelsaved/'+modelname))
    noises = Variable(torch.randn(400, 1, 5))
    gen_points = G(noises).data[:, 0, 0].view(400, 1)
    gen_points = listmerge(gen_points.tolist())
    minlead = min(gen_points)
    maxlead = max(gen_points)
    for i in range(len(gen_points)):
        gen_points[i] = (gen_points[i]-minlead) / (maxlead-minlead)
    return gen_points

def rnn_ae_gen(modelname):
    mynet = autocoder(RNN_Encoder(1, 50, 1, 2), RNN_Decoder(1, 50, 1, 2))
    mynet.load_state_dict(torch.load('modelsaved/'+modelname))
    gen_points = mynet(torch.Tensor(400, 1, 1))
    gen_points = listmerge(gen_points.tolist())
    gen_points = listmerge(gen_points)
    minlead = min(gen_points)
    maxlead = max(gen_points)
    for i in range(len(gen_points)):
        gen_points[i] = (gen_points[i] - minlead) / (maxlead - minlead)
    return gen_points


def rnn_vae_gen(modelname):
    mynet = VAE(RNN_Encoder(1, 50, 1, 2), RNN_Decoder(1, 50, 1, 2))
    mynet.load_state_dict(torch.load('modelsaved/'+modelname))
    gen_points = mynet(torch.Tensor(400, 1, 1))
    gen_points = listmerge(gen_points.tolist())
    gen_points = listmerge(gen_points)
    minlead = min(gen_points)
    maxlead = max(gen_points)
    for i in range(len(gen_points)):
        gen_points[i] = (gen_points[i] - minlead) / (maxlead - minlead)
    return gen_points

def lstm_ae_gen(modelname):
    mynet = autocoder(LSTM_Encoder(1, 50, 1, 2), LSTM_Decoder(1, 50, 1, 2))
    mynet.load_state_dict(torch.load('modelsaved/'+modelname))
    gen_points = mynet(torch.Tensor(400, 1, 1))
    gen_points = listmerge(gen_points.tolist())
    gen_points = listmerge(gen_points)
    minlead = min(gen_points)
    maxlead = max(gen_points)
    for i in range(len(gen_points)):
        gen_points[i] = (gen_points[i] - minlead) / (maxlead - minlead)
    return gen_points

def lstm_vae_gen(modelname):
    mynet = VAE(LSTM_Encoder(1, 50, 1, 2), LSTM_Decoder(1, 50, 1, 2))
    mynet.load_state_dict(torch.load('modelsaved/'+modelname))
    gen_points = mynet(torch.Tensor(400, 1, 1))
    gen_points = listmerge(gen_points.tolist())
    gen_points = listmerge(gen_points)
    minlead = min(gen_points)
    maxlead = max(gen_points)
    for i in range(len(gen_points)):
        gen_points[i] = (gen_points[i] - minlead) / (maxlead - minlead)
    return gen_points


