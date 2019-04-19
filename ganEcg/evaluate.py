from __future__ import division
from gen_Ecg import generateEcg
from helpfunction.helper import RMSE,PRD,frdist,array_convert
import pickle
from gen_Ecg import  rnn_ae_gen, rnn_vae_gen, lstm_ae_gen, lstm_vae_gen

with open('pickle/pointx.pickle', 'rb') as f:
    pointlistx = pickle.load(f)

mylistx = pointlistx[0:400]
Minvalue = min(mylistx)
Maxvalue = max(mylistx)
mylistx = [(mylistx[i]- Minvalue)/(Maxvalue-Minvalue) for i in range(len(mylistx))]


filename_cnn = 'generatorcnn.pkl'
gen_points_cnn = generateEcg(filename_cnn)
# filename_gru = 'generatorgru.pkl'
# gen_points_gru = generateEcg(filename_gru)
# filename_lstm = 'generatorlstm.pkl'
# gen_points_lstm = generateEcg(filename_lstm)
# filename_mlp = 'generatormlp.pkl'
# gen_points_mlp = generateEcg(filename_mlp)
#
#
print '*****cnn*****'
print 'RMSE: '+ str(RMSE(mylistx, gen_points_cnn))
print 'PRD: '+ str(PRD(mylistx, gen_points_cnn))
print 'frdistance: '+ str(frdist(array_convert(mylistx), array_convert(gen_points_cnn)))
#
# print '*****gru*****'
# print 'RMSE: '+ str(RMSE(mylistx, gen_points_gru))
# print 'PRD: '+ str(PRD(mylistx, gen_points_gru))
# print 'frdistance: '+ str(frdist(array_convert(mylistx), array_convert(gen_points_gru)))
#
# print '*****lstm*****'
# print 'RMSE: '+ str(RMSE(mylistx, gen_points_lstm))
# print 'PRD: '+ str(PRD(mylistx, gen_points_lstm))
# print 'frdistance: '+ str(frdist(array_convert(mylistx), array_convert(gen_points_lstm)))
#
# print '*****mlp*****'
# print 'RMSE: '+ str(RMSE(mylistx, gen_points_mlp))
# print 'PRD: '+ str(PRD(mylistx, gen_points_mlp))
# print 'frdistance: '+ str(frdist(array_convert(mylistx), array_convert(gen_points_mlp)))


filename_rnnae = 'rnn_auto.pkl'
gen_points_rnnae = rnn_ae_gen(filename_rnnae)

filename_rnnva = 'rnn_vae.pkl'
gen_points_rnnva = rnn_vae_gen(filename_rnnva)

filename_lstmae= 'lstm_auto.pkl'
gen_points_lstmae = lstm_ae_gen(filename_lstmae)

filename_lstmva = 'lstm_vae.pkl'
gen_points_lstmva = lstm_vae_gen(filename_lstmva)





print '*****rnn-ae*****'
print 'RMSE: '+ str(RMSE(mylistx, gen_points_rnnae))
print 'PRD: '+ str(PRD(mylistx, gen_points_rnnae))
print 'frdistance: '+ str(frdist(array_convert(mylistx), array_convert(gen_points_rnnae)))

print '*****rnn-vae*****'
print 'RMSE: '+ str(RMSE(mylistx, gen_points_rnnva))
print 'PRD: '+ str(PRD(mylistx, gen_points_rnnva))
print 'frdistance: '+ str(frdist(array_convert(mylistx), array_convert(gen_points_rnnva)))

print '*****lstm-ae*****'
print 'RMSE: '+ str(RMSE(mylistx, gen_points_lstmae))
print 'PRD: '+ str(PRD(mylistx, gen_points_lstmae))
print 'frdistance: '+ str(frdist(array_convert(mylistx), array_convert(gen_points_lstmae)))

print '*****lstm-vae*****'
print 'RMSE: '+ str(RMSE(mylistx, gen_points_lstmva))
print 'PRD: '+ str(PRD(mylistx, gen_points_lstmva))
print 'frdistance: '+ str(frdist(array_convert(mylistx), array_convert(gen_points_lstmva)))