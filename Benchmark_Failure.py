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
from joblib import Parallel, delayed
import multiprocessing

dataname_list = ['synthetic', 'abalone', 'comm.and.crime', 'concrete', 'fault', 'gas', 'imgseg', 'landsat',
                 'letter.rec',
                 'magic.gamma', 'opt.digits', 'pageb', 'particle', 'shuttle', 'skin', 'spambase', 'wave', 'wine',
                 'yearp', 'yeast']

sp_auc_fail = 0
aNNE_auc_fail = 0
sp_ap_fail = 0
aNNE_ap_fail = 0
sp_either_fail = 0
aNNE_either_fail = 0

total = 0

for tt in range(len(dataname_list)):
    dataname = dataname_list[tt]
    header_file = '/Users/apple/Documents/AD_Datasets/' + dataname + '/meta_data/' + 'meta_' + dataname + '.csv'
    header = pd.read_csv(header_file)

    ci_auc = header['auc.ci.0.999']
    ci_ap = header['ap.ci.0.999']

    auc_summary = np.load('data/' + dataname + '_auc.npy')
    ap_summary = np.load('data/' + dataname + '_ap.npy')

    sp_auc_fail += sum(auc_summary[:, 0] < ci_auc)
    aNNE_auc_fail += sum(auc_summary[:, 1] < ci_auc)

    sp_ap_fail += sum(ap_summary[:, 0] < ci_ap)
    aNNE_ap_fail += sum(ap_summary[:, 1] < ci_ap)

    sp_either_fail += sum((auc_summary[:, 0] < ci_auc) | (ap_summary[:, 0] < ci_ap))
    aNNE_either_fail += sum((auc_summary[:, 1] < ci_auc) | (ap_summary[:, 1] < ci_ap))

    total += header.shape[0]

sp_auc_failrate = sp_auc_fail / total
aNNE_auc_failrate = aNNE_auc_fail / total

sp_ap_failrate = sp_ap_fail / total
aNNE_ap_failrate = aNNE_ap_fail / total

sp_either_failrate = sp_either_fail / total
aNNE_either_failrate = aNNE_either_fail / total

print(sp_auc_failrate)
print(aNNE_auc_failrate)

print(sp_ap_failrate)
print(aNNE_ap_failrate)

print(sp_either_failrate)
print(aNNE_either_failrate)