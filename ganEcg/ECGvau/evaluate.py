import visdom
import torch as t
import csv
import numpy as np

File = open('newcnn.csv')

Reader = csv.reader(File)
data = list(Reader)
print data[0:10]

myfirlist = data[1:]
dlosslist = [float(myfirlist[i][1]) for i in range(len(myfirlist))]
glosslist = [float(myfirlist[j][2]) for j in range(len(myfirlist))]
print len(dlosslist)

vis = visdom.Visdom(env='oop')


x_list = []
for i in range(50000):
    x_list.append(i)

# setparameters1 = dict(title='loss of D', xlabel='iteration', ylabel='loss')
# setparameters2 = dict(title='loss of G', xlabel='iteration', ylabel='loss')
setparameters = dict(legend = ['dloss', 'gloss', 'MLP', 'CNN'], title = 'loss of discriminator', fillarea = True,
                                 xlabel = 'epoch', ylabel = 'loss')

vis.line(X=np.column_stack((np.array(x_list), np.array(x_list))),
         Y=np.column_stack((np.array(dlosslist[0:50000]), np.array(glosslist[0:50000]))))

# vis.line(X=np.array(x_list), Y=np.array(dlosslist[0:100000]), win = 'dloss', opts=setparameters1)
# vis.line(X=np.array(x_list), Y=np.array(glosslist[0:100000]), win = 'gloss', opts=setparameters2)