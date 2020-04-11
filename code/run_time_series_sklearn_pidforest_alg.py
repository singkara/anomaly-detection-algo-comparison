import numpy as np
import matplotlib.pyplot as plt
from scripts.forest import Forest
from scripts.DTM import DTM

from sklearn.ensemble import IsolationForest, BaggingRegressor
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

import time
import scipy.io as sio
from sklearn import metrics

datasets = ['nyc_taxi','ambient_temperature_system_failure','cpu_utilization_asg_misconfiguration','machine_temperature_system_failure','ec2_request_latency_system_failure']

L = len(datasets)
trials = 5
run_lof_svm = 1

for i in range(0, L):
    mat_data = sio.loadmat('../data/' + datasets[i] + '.mat')
    X = mat_data['X']
    t = X[0, 0]
    if str(type(t)) == "<class 'numpy.uint8'>":
        X = np.int_(X)
    y = mat_data['y']
    file_name = 'experiment_results/' + datasets[i] + '.txt'
    File_object = open(file_name, "w")
    time_all = np.zeros((trials, 5))
    precision_all = np.zeros((trials, 5))
    auc_all = np.zeros((trials, 5))

    for j in range(0, trials):

        print('\n\n******' + datasets[i] + ' trial ' + str(j + 1) + '*******\n\n')

        print('\n******Iso-Forest*******\n')
        start = time.time()
        clf = IsolationForest(contamination=0.1, behaviour='new')
        clf.fit(X)
        end = time.time()
        time_all[j, 0] = end - start
        iso_scores = clf.score_samples(X)

        if run_lof_svm == 0:
            lof_scores = iso_scores
            osvm_scores = iso_scores
        elif j == 0:

            print('\n******LOF*******\n')
            start = time.time()
            lof = LocalOutlierFactor()
            lof.fit(X)
            end = time.time()
            time_all[j, 1] = end - start
            lof_scores = lof.negative_outlier_factor_

            print('\n******1-class SVM*******\n')
            start = time.time()
            osvm = OneClassSVM(kernel='rbf')
            osvm.fit(X)
            end = time.time()
            time_all[j, 2] = end - start
            osvm_scores = osvm.score_samples(X)

        print('\n******Our Algo*******\n')
        start = time.time()
        t1, _ = np.shape(X)
        # n_samples = int(max(t1/250,100))
        # n_samples = int(t1/50)
        n_samples = 100
        kwargs = {'max_depth': 10, 'n_trees': 50, 'max_samples': n_samples, 'max_buckets': 3, 'epsilon': 0.1,
                  'sample_axis': 1,
                  'threshold': 0}
        forest = Forest(**kwargs)
        forest.fit(np.transpose(X))
        indices, outliers, scores, pst, our_scores = forest.predict(np.transpose(X), err=0.1, pct=50)
        end = time.time()
        time_all[j, 3] = end - start

        print('\n****** DTM *******\n')
        rng = np.random.RandomState(42)
        max_samples = min(20, X.shape[0])
        bag_neigh = 1
        clf_spDTM = BaggingRegressor(base_estimator=DTM(n_neighbors=bag_neigh, contamination=0.1),
                                     n_estimators=1, max_samples=max_samples, bootstrap=False, random_state=rng)
        start = time.time()
        y_score_spDTM = clf_spDTM.fit(X, y).predict(X)

        end = time.time()
        time_all[j, 4] = end - start

        precision_iso, recall_iso, thresholds_iso = metrics.precision_recall_curve(y, -iso_scores, pos_label=1)
        precision_lof, recall_lof, thresholds_lof = metrics.precision_recall_curve(y, -lof_scores, pos_label=1)
        precision_osvm, recall_osvm, thresholds_osvm = metrics.precision_recall_curve(y, -osvm_scores, pos_label=1)
        precision_our, recall_our, thresholds_our = metrics.precision_recall_curve(y, -our_scores, pos_label=1)
        precision_dtm, recall_dtm, thresholds_dtm = metrics.precision_recall_curve(y, -y_score_spDTM, pos_label=1)

        precision_all[j, 0] = max(2 * precision_iso * recall_iso / (precision_iso + recall_iso))
        precision_all[j, 1] = max(2 * precision_lof * recall_lof / (precision_lof + recall_lof))
        precision_all[j, 2] = max(2 * precision_osvm * recall_osvm / (precision_osvm + recall_osvm))
        precision_all[j, 3] = max(2 * precision_our * recall_our / (precision_our + recall_our))
        precision_all[j, 4] = max(2 * precision_dtm * recall_dtm / (precision_dtm + recall_dtm))

        auc_all[j, 0] = metrics.roc_auc_score(y, -iso_scores)
        auc_all[j, 1] = metrics.roc_auc_score(y, -lof_scores)
        auc_all[j, 2] = metrics.roc_auc_score(y, -osvm_scores)
        auc_all[j, 3] = metrics.roc_auc_score(y, -our_scores)
        auc_all[j, 4] = metrics.roc_auc_score(y, -y_score_spDTM)

        for k in range(0, 4):
            print('{:.4f}\t'.format(precision_all[j, k]))
        print('\n')

        for k in range(0, 4):
            print('{:.4f}\t'.format(auc_all[j, k]))
        print('\n')

    File_object.write(str(kwargs))

    File_object.write('\n\nIF\tLOF\tSVM\tOur-Algo\n\n')

    for j in range(0, trials):
        for k in range(0, 4):
            File_object.write('{:.4f}\t'.format(precision_all[j, k]))
        File_object.write('\n')

    File_object.write('\n')

    for k in range(0, 4):
        File_object.write('{:.4f}\t'.format(np.mean(precision_all[:, k])))
    File_object.write('\n')

    for k in range(0, 4):
        File_object.write('{:.4f}\t'.format(np.std(precision_all[:, k])))
    File_object.write('\n')

    File_object.write('\nIF\tLOF\tSVM\tOur-Algo\n\n')

    for j in range(0, trials):
        for k in range(0, 4):
            File_object.write('{:.4f}\t'.format(auc_all[j, k]))
        File_object.write('\n')

    File_object.write('\n')

    for k in range(0, 4):
        File_object.write('{:.4f}\t'.format(np.mean(auc_all[:, k])))
    File_object.write('\n')

    for k in range(0, 4):
        File_object.write('{:.4f}\t'.format(np.std(auc_all[:, k])))
    File_object.write('\n')

    File_object.close()

    file_name = 'experiment_results/' + datasets[i] + '_results.mat'
    sio.savemat(file_name, {'time_all': time_all, 'precision_all': precision_all, 'auc_all': auc_all})

