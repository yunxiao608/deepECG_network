# encoding=utf-8
from __future__ import division
import torch
import torch.nn as nn
import pickle
from torch.autograd import Variable
from Config import opt
from Model import Generator, Discriminator
from helpfunction.helper import listsumavg
import csv

with open('pickle/pointx.pickle', 'rb') as f:
    pointlistx = pickle.load(f)

# convert to lead
for i in range(len(pointlistx)):
    pointlistx[i] = pointlistx[i]/200

D = Discriminator(input_dim=1, hidden_dim=50, output_dim=1, num_layers=2)
G = Generator(input_dim=5, hidden_dim=50, output_dim=1, num_layers=2)
if torch.cuda.is_available():
    D.cuda()
    G.cuda()

criterion = nn.BCELoss()
opt_D = torch.optim.Adam(D.parameters(), lr=opt.LR_D)
opt_G = torch.optim.Adam(G.parameters(), lr=opt.LR_G)

Filename = 'resultdata/dlstm.csv'
outputFile = open(Filename, 'w')
outputWriter = csv.writer(outputFile)
outputWriter.writerow(['Epoch', 'Dloss', 'Gloss'])

for epoch in range(opt.Epoches):
    dloss_list = []
    gloss_list = []
    for batch in range(int(len(pointlistx)/opt.Seq_Len/opt.BATCH_SIZE)):
        # 序列长度，batch_size,维度
        batch_noise = torch.randn(opt.Seq_Len, opt.BATCH_SIZE, 5).cuda()
        batch_noise = Variable(batch_noise)
        batch_gdata = G(batch_noise)

        # 真实数据
        batch_realdata = torch.randn(opt.Seq_Len, opt.BATCH_SIZE, 1).cuda()
        batch_realdata = batch_realdata.view(opt.BATCH_SIZE, opt.Seq_Len,1)
        for i in range(opt.BATCH_SIZE):
            for j in range(opt.Seq_Len):
                    batch_realdata[i][j][0] = pointlistx[batch *opt.Seq_Len*opt.BATCH_SIZE+i*opt.Seq_Len+j]

        batch_realdata = batch_realdata.view(opt.Seq_Len, opt.BATCH_SIZE, 1)
        batch_realdata = Variable(batch_realdata)

        # 判别器判别生成数据,真实数据
        prob_fake = D(batch_gdata)
        prob_real = D(batch_realdata)

        d_loss = -torch.mean(torch.log(prob_real) + torch.log(1 - prob_fake))
        g_loss = -torch.mean(torch.log(prob_fake))

        opt_D.zero_grad()
        d_loss.backward(retain_graph=True)
        opt_D.step()

        opt_G.zero_grad()
        g_loss.backward(retain_graph=True)
        opt_G.step()

        print 'Epoch:{} / BATCH: {} ,d_loss:{},g_loss:{}'.format(
            epoch, batch, d_loss.data.cpu().numpy(), g_loss.data.cpu().numpy())

        dloss_list.append(d_loss.data.cpu().numpy())
        gloss_list.append(g_loss.data.cpu().numpy())

    dloss_avg = listsumavg(dloss_list)
    gloss_avg = listsumavg(gloss_list)
    outputWriter.writerow([epoch, dloss_avg, gloss_avg])
outputFile.close()


torch.save(G.state_dict(), 'modelsaved/generatorlstm.pkl')

