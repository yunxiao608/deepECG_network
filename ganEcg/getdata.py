# encoding=utf-8
import matplotlib.pyplot as plt
import os
import pickle
# 将字符转换成二进制


def encode(s):
    c1 = ' '.join([bin(ord(c)).replace('0b', '') for c in s])
    if len(c1) < 8:
        c2 = (8 - len(c1)) * '0'
        return c2 + c1

    else:
        return c1

path = os.getcwd()+'/data/'
filelist = os.listdir(path)
# print filelist
filecontent = ''
for i in range(len(filelist)):
    fo = open(path+str(filelist[i]), 'rb')
    filecontent += fo.read()
    fo.close()


arraybuff = []
for i in range(len(filecontent)):
    arraybuff.append(encode(filecontent[i]))
# print arraybuff

N = 3
M = len(arraybuff) / N
a = arraybuff
b = [([0] * N) for i in range(M)]
for i in range(M):
    for j in range(N):
        b[i][j] = a[i * 3 + j]

# print b


def x_func(i): return b[i][1][4:8] + b[i][0]


def y_func(j): return b[j][1][0:4] + b[j][2]


pointlistx = []
pointlisty = []
for i in range(M):
    pointlistx.append(int(x_func(i), base=2))
    pointlisty.append(int(y_func(i), base=2))

print len(pointlistx)
file = open('datasaved/pointx.pickle','wb')
pickle.dump(pointlistx, file)
file.close()
# plt.figure()
# plt.title('ECG')
# plt.plot(pointlistx[1:3120])
# plt.show()
