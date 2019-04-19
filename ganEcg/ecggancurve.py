from gen_Ecg import generateEcg
import visdom
import numpy as np
from helpfunction.helper import avg_list, variance

filename_cnn = 'generatorcnn.pkl'
gen_points_cnn = generateEcg(filename_cnn)
filename_gru = 'generatorgru.pkl'
gen_points_gru = generateEcg(filename_gru)
filename_lstm = 'generatorlstm.pkl'
gen_points_lstm = generateEcg(filename_lstm)
filename_mlp = 'generatormlp.pkl'
gen_points_mlp = generateEcg(filename_mlp)




vis = visdom.Visdom(env='ecggancurve')

x_list = []
for i in range(400):
    x_list.append(i)

setparameters1 = dict(xlabel = 'timestep[s]', ylabel = 'lead', title = 'BiLSTM-CNN')
setparameters2 = dict(xlabel = 'timestep[s]', ylabel = 'lead', title = 'BiLSTM-GRU')
setparameters3 = dict(xlabel = 'timestep[s]', ylabel = 'lead', title = 'BiLSTM-LSTM')
setparameters4 = dict(xlabel = 'timestep[s]', ylabel = 'lead', title = 'BiLSTM-MLP')
vis.line(X = np.array(x_list), Y = np.array(gen_points_cnn), win='dcnn', opts = setparameters1)
vis.line(X = np.array(x_list), Y = np.array(gen_points_gru), win='dgru', opts = setparameters2)
vis.line(X = np.array(x_list), Y = np.array(gen_points_lstm), win='dlstm', opts = setparameters3)
vis.line(X = np.array(x_list), Y = np.array(gen_points_mlp), win='dmlp', opts = setparameters4)






print 'cnn (mean, deviation): '+ str(avg_list(gen_points_cnn)) + ', ' + str(variance(gen_points_cnn, avg_list(gen_points_cnn)))
print 'gru (mean, deviation): '+ str(avg_list(gen_points_gru)) + ', ' + str(variance(gen_points_gru, avg_list(gen_points_gru)))
print 'lstm (mean, deviation): '+ str(avg_list(gen_points_lstm)) + ', ' + str(variance(gen_points_lstm, avg_list(gen_points_lstm)))
print 'mlp (mean, deviation): '+str(avg_list(gen_points_mlp)) + ', ' + str(variance(gen_points_mlp, avg_list(gen_points_mlp)))