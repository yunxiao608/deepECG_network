import visdom
import torch as t
import csv
import numpy as np


firFile = open('resultdata/dgru.csv')
secFile = open('resultdata/dlstm.csv')
tirFile = open('resultdata/dmlp.csv')
fourFile = open('resultdata/dcnn.csv')

firReader = csv.reader(firFile)
firdata = list(firReader)
secReader = csv.reader(secFile)
secdata = list(secReader)
tirReader = csv.reader(tirFile)
tirdata = list(tirReader)
fourReader = csv.reader(fourFile)
fourdata = list(fourReader)


myfirlist = firdata[1:]
dgrulosslist = [float(myfirlist[i][1]) for i in range(len(myfirlist))]
myseclist = secdata[1:]
dlstmlosslist = [float(myseclist[i][1]) for i in range(len(myseclist))]
mytirlist = tirdata[1:]
dlinearlosslist = [float(mytirlist[i][1]) for i in range(len(mytirlist))]
myfourlist = fourdata[1:]
dcnnlosslist = [float(myfourlist[i][1]) for i in range(len(myfourlist))]


vis = visdom.Visdom(env='yefei')


x_list = []
for i in range(500):
    x_list.append(i)

setparameters = dict(legend = ['D-GRU', 'D-LSTM', 'D-MLP', 'D-CNN'], title = 'loss of discriminator',
                                 xlabel = 'epoch', ylabel = 'loss')
# setparameters2 = dict(legend = ['GRU', 'LSTM', 'MLP', 'CNN'], title = 'loss of discriminator',
#                                  xlabel = 'epoch', ylabel = 'loss')
# setparameters3 = dict(legend = ['GRU', 'LSTM', 'MLP', 'CNN'], title = 'loss of discriminator',
#                                  xlabel = 'epoch', ylabel = 'loss')
# setparameters4 = dict(legend = ['GRU', 'LSTM', 'MLP', 'CNN'], title = 'loss of discriminator',
#                                  xlabel = 'epoch', ylabel = 'loss')

vis.line(X=np.column_stack((np.array(x_list[0:200]), np.array(x_list[0:200]), np.array(x_list[0:200]), np.array(x_list[0:200]))),
         Y=np.column_stack((np.array(dgrulosslist[0:200]), np.array(dlstmlosslist[0:200]), np.array(dlinearlosslist[0:200]), np.array(dcnnlosslist[0:200]))),
         win='after 200 epoch', opts = setparameters)
vis.line(X=np.column_stack((np.array(x_list[0:300]), np.array(x_list[0:300]), np.array(x_list[0:300]), np.array(x_list[0:300]))),
         Y=np.column_stack((np.array(dgrulosslist[0:300]), np.array(dlstmlosslist[0:300]), np.array(dlinearlosslist[0:300]), np.array(dcnnlosslist[0:300]))),
         win='after 300 epoch', opts = setparameters)
vis.line(X=np.column_stack((np.array(x_list[0:400]), np.array(x_list[0:400]), np.array(x_list[0:400]), np.array(x_list[0:400]))),
         Y=np.column_stack((np.array(dgrulosslist[0:400]), np.array(dlstmlosslist[0:400]), np.array(dlinearlosslist[0:400]), np.array(dcnnlosslist[0:400]))),
         win='after 400 epoch', opts = setparameters)
vis.line(X=np.column_stack((np.array(x_list[0:500]), np.array(x_list[0:500]), np.array(x_list[0:500]), np.array(x_list[0:500]))),
         Y=np.column_stack((np.array(dgrulosslist[0:500]), np.array(dlstmlosslist[0:500]), np.array(dlinearlosslist[0:500]), np.array(dcnnlosslist[0:500]))),
         win='after 500 epoch', opts = setparameters)
