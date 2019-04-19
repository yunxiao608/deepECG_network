# encoding=utf-8
from matplotlib import pyplot
import matplotlib.pyplot as plt


x = [50, 100, 150, 200, 250, 300, 350, 400]
y_train = [0.299, 0.375, 0.348, 0.337, 0.267, 0.280, 0.248, 0.215]
y_test = [0.635, 0.765, 0.992, 0.846, 0.728, 0.820, 0.797, 0.803]
# plt.plot(x, y, 'ro-')
# plt.plot(x, y1, 'bo-')
# pl.xlim(-1, 11)  # 限定横轴的范围
# pl.ylim(-1, 110)  # 限定纵轴的范围


plt.plot(x, y_train, marker='o', mec='r', mfc='w', label='RMSE')
plt.plot(x, y_test, marker='*', ms=10, label='FD')
plt.legend()  # 让图例生效


plt.margins(0)
plt.subplots_adjust(bottom=0.10)
plt.xlabel('length')  # X轴标签
plt.ylabel("value")  # Y轴标签
pyplot.yticks([0.150, 0.300, 0.450, 0.600, 0.750, 0.900])
# plt.title("A simple plot") #标题
plt.show()