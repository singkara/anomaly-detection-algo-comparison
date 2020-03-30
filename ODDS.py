import numpy as np
import scipy.io as sio
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
from sklearn.ensemble import BaggingRegressor
import matplotlib.pyplot as plt
import random
from sklearn.ensemble import RandomForestClassifier
import scipy as sp
import csv
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
import scipy.io as sio
from DTM import *
#from iNNE import *

def rank_func(array):
    ranked = array.copy()
    for i in range(ranked.shape[0]):
        row = ranked[i,:]
        temp = row.argsort()
        sorted_row = (array.shape[1]-1) - np.arange(len(row))[temp.argsort()]
        ranked[i,:] = sorted_row
    return ranked

# Preprocessing
dataname_list = ['annthyroid','mammography']

data_number = len(dataname_list)
auc_IF = np.zeros(data_number)
auc_LOF = np.zeros(data_number)
auc_DTM = np.zeros(data_number)
auc_BDTM = np.zeros(data_number)
auc_iNNE = np.zeros(data_number)

ap_IF = np.zeros(data_number)
ap_LOF = np.zeros(data_number)
ap_DTM = np.zeros(data_number)
ap_BDTM = np.zeros(data_number)
ap_iNNE = np.zeros(data_number)

for i in range(len(dataname_list)):

    dataname = dataname_list[i]
    print(dataname)

    data_file = dataname+'.mat'
    data = sio.loadmat(data_file)

    X_train = data['X'].astype('double')
    y_label = np.squeeze(data['y']).astype('int')
    contamination = sum(y_label)/len(y_label)

    rng = np.random.RandomState(42)
    neigh = max(10,int(np.floor(0.03*X_train.shape[0])))

    # Iforest
    print('IF')
    max_samples = min(256,X_train.shape[0])
    clf_IF = IsolationForest(random_state=rng, max_samples = max_samples, contamination = contamination,n_jobs=4)
    clf_IF.fit(X_train)
    y_score_IF = clf_IF.decision_function(X_train)  # higher score = inlier, lower score = outlier
    #fpr_IF, tpr_IF, thresholds_IF = roc_curve(y_label, y_score_IF)
    auc_IF[i]= roc_auc_score(y_label, -y_score_IF)
    ap_IF[i] = average_precision_score(y_label, -y_score_IF)

    #LOF
    print('LOF')
    clf_LOF = LocalOutlierFactor(n_neighbors=neigh, contamination=contamination,n_jobs=4)
    clf_LOF.fit_predict(X_train)
    y_score_LOF = clf_LOF.negative_outlier_factor_  # higher score = inlier, lower score = outlier
    #fpr_LOF, tpr_LOF, thresholds_LOF = roc_curve(y_label, y_score_LOF)
    auc_LOF[i] = roc_auc_score(y_label, -y_score_LOF)
    ap_LOF[i] = average_precision_score(y_label, -y_score_LOF)

    #DTM
    print('DTM')
    clf_DTM = DTM(n_neighbors=neigh, contamination=contamination, n_jobs=4)
    y_score_DTM = clf_DTM.fit_predict(X_train)
    #fpr_DTM, tpr_DTM, thresholds_DTM = roc_curve(y_label, -DTM_score)
    auc_DTM[i] = roc_auc_score(y_label, -y_score_DTM)
    ap_DTM[i] = average_precision_score(y_label, -y_score_DTM)


auc = np.vstack([auc_IF,auc_LOF,auc_DTM]).T
ap = np.vstack([ap_IF,ap_LOF,ap_DTM]).T

rank_auc = rank_func(auc)
rank_ap = rank_func(ap)

auc_rankavg = np.mean(rank_auc,axis=0)
ap_rankavg = np.mean(rank_ap,axis=0)

auc = np.vstack([auc,auc_rankavg])
ap = np.vstack([ap,ap_rankavg])
print(auc)
print(ap)

np.save('data/real_auc',auc)
np.save('data/real_ap',ap)

def df_convert(filename):
    dat = np.load(filename)
    dat = pd.DataFrame(dat)
    dat.rename(index={0:'annthyroid',1:'arrhythmia',2:'breastw',3:'cardio',4:'glass',
             5:'ionosphere',6:'letter',7:'lympho',8:'mammography',9:'mnist',10:'musk',
            11:'optdigits',12:'pendigits',13:'pima',14:'satellite',15:'satimage-2',16:'shuttle',
            17:'speech',18:'thyroid',19:'vertebral',20:'vowels',21:'wbc',22:'wine',23:'(rank)'}, inplace=True)
    dat.columns = ['IF','LOF','DTM2']
    return dat

df_auc = df_convert('data/real_auc.npy')
df_ap = df_convert('data/real_ap.npy')