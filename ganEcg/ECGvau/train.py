# encoding=utf-8
from __future__ import division
from Model import Generator, CNNDIS
import torch
import torch.nn as nn
from torch.autograd import Variable
from Config import opt
import csv

firFile = open('ecgdata.csv')
firReader = csv.reader(firFile)
firdata = list(firReader)

myfirlist = firdata[2:]
cleanedlist = [float(myfirlist[i][1]) for i in range(len(myfirlist))]
pointlistx = cleanedlist
print len(pointlistx)
D = CNNDIS()
G = Generator(input_dim=5, hidden_dim=150, output_dim=1, num_layers=20)
if torch.cuda.is_available():
    D.cuda()
    G.cuda()

criterion = nn.BCELoss()
opt_D = torch.optim.Adam(D.parameters(), lr=opt.LR_D)
opt_G = torch.optim.Adam(G.parameters(), lr=opt.LR_G)

Filename = 'newcnn.csv'
outputFile = open(Filename, 'w')
outputWriter = csv.writer(outputFile)
outputWriter.writerow(['Iteration', 'Dloss', 'Gloss'])


for epoch in range(opt.Epoches):
    dloss_list = []
    gloss_list = []
    for batch in range(int(len(pointlistx)/opt.Seq_Len/opt.BATCH_SIZE)):
        # 序列长度，batch_size,维度
        batch_noise = Variable(torch.randn(opt.BATCH_SIZE, opt.Seq_Len, 5).cuda())
        batch_gdata = G(batch_noise)
        batch_gdata = batch_gdata.view(-1, 1, opt.Seq_Len, 1)

        # 真实数据
        batch_realdata = torch.randn(opt.BATCH_SIZE, opt.Seq_Len, 1).cuda()
        for i in range(opt.BATCH_SIZE):
            for j in range(opt.Seq_Len):
                    batch_realdata[i][j][0] = pointlistx[batch *opt.Seq_Len*opt.BATCH_SIZE+i*opt.Seq_Len+j]

        batch_realdata = Variable(batch_realdata.view(opt.BATCH_SIZE, -1, opt.Seq_Len, 1))

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
        dlosscpu = d_loss.data.cpu().numpy()
        glosscpu = g_loss.data.cpu().numpy()
        dloss_list.append(d_loss.data.cpu().numpy())
        gloss_list.append(g_loss.data.cpu().numpy())

    # dloss_avg = listsumavg(dloss_list)
    # gloss_avg = listsumavg(gloss_list)
        outputWriter.writerow([batch+4*epoch, dlosscpu, glosscpu])
outputFile.close()

torch.save(G.state_dict(), 'newgeneratorcnn.pkl')


