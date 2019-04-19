import visdom
import csv
import numpy as np

# gengrudata = generateEcg('generatorgru.pkl')
# genlstmdata = generateEcg('generatorlstm.pkl')
#
# vis = visdom.Visdom(env='gendata')

x_list = []
for i in range(500):
    x_list.append(i)




# lstm_vae
lstmvaelossfile = open('resultdata/vae.csv')
lstmvaelossreader = csv.reader(lstmvaelossfile)
lstmvaeloss = list(lstmvaelossreader)

# lstm_auto
lstmautolossfile = open('resultdata/auto.csv')
lstmautolossreader = csv.reader(lstmautolossfile)
lstmautoloss = list(lstmautolossreader)

# rnn_vae
rnnvaelossfile = open('resultdata/rnn_vae.csv')
rnnvaelossreader = csv.reader(rnnvaelossfile)
rnnvaeloss = list(rnnvaelossreader)

# rnn_auto
rnnautolossfile = open('resultdata/rnn_auto.csv')
rnnautolossreader = csv.reader(rnnautolossfile)
rnnautoloss = list(rnnautolossreader)

# GAN
ganlossfile = open('resultdata/dcnn.csv')
ganlossreader = csv.reader(ganlossfile)
ganloss = list(ganlossreader)


# lstm_vae
lstmvaedata = lstmvaeloss[1:]
lstmvaelosslist = [float(lstmvaedata[i][1]) for i in range(len(lstmvaedata))]

# lstm_auto
lstmautodata = lstmautoloss[1:]
lstmautolosslist = [float(lstmautodata[i][1]) for i in range(len(lstmautodata))]

# rnnvae
rnnvaedata = rnnvaeloss[1:]
rnnvaelosslist = [float(rnnvaedata[i][1]) for i in range(len(rnnvaedata))]

# rnnauto
rnnautodata = rnnautoloss[1:]
rnnautolosslist = [float(rnnautodata[i][1]) for i in range(len(rnnautodata))]

# GAN
gandata = ganloss[1:]
ganlosslist = [float(gandata[i][1]) for i in range(len(gandata))]

vis = visdom.Visdom(env = 'genmodel')
x_list = []
for i in range(500):
    x_list.append(i)

setparameters = dict(legend = ['RNN-VAE', 'RNN-AE', 'LSTM-VAE', 'LSTM-AE', 'GAN'],
                     xlabel = 'epoch', ylabel = 'loss', title = 'losses of generative models')
vis.line(X=np.column_stack((np.array(x_list), np.array(x_list), np.array(x_list), np.array(x_list), np.array(x_list))),
         Y=np.column_stack((np.array(rnnvaelosslist), np.array(rnnautolosslist), np.array(lstmvaelosslist), np.array(lstmautolosslist),np.array(ganlosslist))),
         opts = setparameters)