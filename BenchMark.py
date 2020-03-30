import numpy as np
import scipy.io as sio
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import random
from sklearn.ensemble import RandomForestClassifier
import scipy as sp
import csv
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
import scipy.io as sio
from DTM import *
from joblib import Parallel, delayed
import multiprocessing
from sklearn.ensemble import BaggingRegressor

# Preprocessing
dataname_list = ['synthetic','abalone','comm.and.crime','concrete','fault','gas','imgseg','landsat','letter.rec',
                 'magic.gamma','opt.digits','pageb','particle','shuttle','skin','spambase','wave','wine',
                 'yearp','yeast']
#tt = 0
for tt in range(3,len(dataname_list)):
    dataname = dataname_list[tt]
    print(dataname)
    header_file = '/Users/apple/Documents/AD_Datasets/'+dataname+'/meta_data/'+'meta_'+dataname+'.csv'
    header = pd.read_csv(header_file)

    header_size = header.shape[0]
    print(header_size)


    def process(i):
        current_dat = header.iloc[i]
        current_dat_name = current_dat['bench.id']
        # Define Datasets
        # print(str(i)+': '+current_dat_name)
        filename = '/Users/apple/Documents/AD_Datasets/' + dataname + '/benchmarks/' + current_dat_name + '.csv'
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile)
            data = list(reader)
        data = np.array(data)

        X_train = data[1:, 6:].astype('double')
        anomaly_type = data[1:, 5]
        y_label = np.zeros(len(anomaly_type))
        # normal_ind = np.where(anomaly_type == 'nominal')[0]
        anomaly_ind = np.where(anomaly_type == 'anomaly')[0]
        y_label[anomaly_ind] = 1
        # X_normal = X_train[normal_ind,:]
        # X_outlier = X_train[anomaly_ind,:]
        # contamination = len(anomaly_ind)/len(y_label)

        rng = np.random.RandomState(42)

        # BaggedDTM
        #     #################################################################################################
        #     # max_samples = min(2048,X_train.shape[0])
        #     # y = np.random.uniform(size=X_train.shape[0])
        #     # bag_neigh = max(10, int(np.floor(0.03 * max_samples)))
        #     # clf_bagDTM = BaggingRegressor(base_estimator=DTM(n_neighbors=bag_neigh,contamination=0.1),
        #     #                               n_estimators=100, max_samples=max_samples, bootstrap=False, random_state=rng)
        #     # y_score_BDTM = clf_bagDTM.fit(X_train, y).predict(X_train)
        #     # # fpr_DTM, tpr_DTM, thresholds_DTM = roc_curve(y_label, -DTM_score)
        #     # auc_BDTM_score = roc_auc_score(y_label, -y_score_BDTM)
        #     # ap_BDTM_score = average_precision_score(y_label, -y_score_BDTM)

        # sp
        #################################################################################################
        max_samples = min(20,X_train.shape[0])
        y = np.random.uniform(size=X_train.shape[0])
        bag_neigh = 1
        clf_spDTM = BaggingRegressor(base_estimator=DTM(n_neighbors=bag_neigh,contamination=0.1),
                                      n_estimators=1, max_samples=max_samples, bootstrap=False, random_state=rng)
        y_score_spDTM = clf_spDTM.fit(X_train, y).predict(X_train)
        auc_spDTM_score = roc_auc_score(y_label, -y_score_spDTM)
        ap_spDTM_score = average_precision_score(y_label, -y_score_spDTM)

        # aNNE
        #################################################################################################
        clf_aNNE = BaggingRegressor(base_estimator=DTM(n_neighbors=bag_neigh,contamination=0.1),
                                      n_estimators=100, max_samples=max_samples, bootstrap=False, random_state=rng)
        y_score_aNNE = clf_aNNE.fit(X_train, y).predict(X_train)
        auc_aNNE_score = roc_auc_score(y_label, -y_score_aNNE)
        ap_aNNE_score = average_precision_score(y_label, -y_score_aNNE)

        return [auc_spDTM_score,auc_aNNE_score], [ap_spDTM_score, ap_aNNE_score]



    num_cores = multiprocessing.cpu_count()
    inputs = range(header_size)
    results = Parallel(n_jobs=num_cores,verbose=True,backend="threading")(delayed(process)(i) for i in inputs)
    #np.save('data/'+dataname+'_auc_BDTM',results)

    myarray = np.asarray(results)
    auc_summary = myarray[:,0,:]
    ap_summary = myarray[:,1,:]
    np.save('data/'+dataname+'_auc',auc_summary)
    np.save('data/'+dataname+'_ap',ap_summary)



