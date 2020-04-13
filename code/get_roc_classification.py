import numpy as np
import matplotlib.pyplot as plt

from scripts.DTM import DTM
from scripts.forest import Forest
from sklearn.ensemble import IsolationForest, BaggingRegressor
import pandas as pd
import scipy.io as sio
from sklearn import metrics
from pyod.models.knn import KNN
from pyod.models.pca import PCA
import rrcf

datasets = ['thyroid','mammography','satimage-2','vowels','siesmic','musk','smtp','http']

L = len(datasets)
trials = 1

for i in range(0,L):
    mat_data = sio.loadmat('../data/'+datasets[i]+'.mat')
    X = mat_data['X']
    #if str(type(t)) == "<class 'numpy.uint8'>":
     #   X = np.int_(X)
    X = np.float_(X)
    [n,d] = np.shape(X)
    y = mat_data['y']

    auc_all = np.zeros((trials,8))
    
    for j in range(0,trials):
    
        print('\n\n******'+datasets[i]+' trial '+str(j+1)+'*******\n\n')
        f = plt.figure()
        plt.rcParams.update({'font.size': 14})
        
        print('\n******PIDForest*******\n')
        n_samples = 100
        kwargs = {'max_depth': 10, 'n_trees':50,  'max_samples': n_samples, 'max_buckets': 3, 'epsilon': 0.1, 'sample_axis': 1, 
          'threshold': 0}
        forest = Forest(**kwargs)
        forest.fit(np.transpose(X))
        indices, outliers, scores , pst, alg_scores = forest.predict(np.transpose(X), err = 0.1, pct=50)
        alg_scores = - alg_scores
        fpr_alg, tpr_alg, thresholds_alg = metrics.roc_curve(y, alg_scores, pos_label=1)
        #plt.plot(fpr_alg,tpr_alg, 'b')
        thresh_len = len(fpr_alg)
        sample_thresh = np.int_( [k * thresh_len/10 for k in range(10)] )
        sample_thresh = np.concatenate( [sample_thresh, np.asarray([thresh_len-1]) ])
        plt.plot(fpr_alg[sample_thresh],tpr_alg[sample_thresh], c = 'k', marker = ">", markersize=15, label='PIDForest')
        auc_all[j,0] = metrics.roc_auc_score(y, alg_scores)
        
        print('\n******iForest*******\n')
        clf = IsolationForest(contamination = 0.1, behaviour = 'new')
        clf.fit(X)
        alg_scores = clf.score_samples(X)
        alg_scores = - alg_scores
        fpr_alg, tpr_alg, thresholds_alg = metrics.roc_curve(y, alg_scores, pos_label=1)
        #plt.plot(fpr_alg,tpr_alg, 'g')
        thresh_len = len(fpr_alg)
        sample_thresh = np.int_( [k * thresh_len/10 for k in range(10)] )
        sample_thresh = np.concatenate( [sample_thresh, np.asarray([thresh_len-1]) ])
        plt.plot(fpr_alg[sample_thresh],tpr_alg[sample_thresh], c = 'g', marker = "v", markersize=5, label='iForest')
        auc_all[j,1] = metrics.roc_auc_score(y, alg_scores)
        
        print('\n******RRCF*******\n')
        num_trees = 2500
        tree_size = 256
        forest = []
        while len(forest) < num_trees:
            # Select random subsets of points uniformly from point set
            ixs = np.random.choice(n, size=(n // tree_size, tree_size),
                   replace=False)
            # Add sampled trees to forest
            trees = [rrcf.RCTree(X[ix], index_labels=ix) for ix in ixs]
            forest.extend(trees)
                
        # Compute average CoDisp
        avg_codisp = pd.Series(0.0, index=np.arange(n))
        index = np.zeros(n)
        for tree in forest:
            codisp = pd.Series({leaf : tree.codisp(leaf) for leaf in tree.leaves})
            avg_codisp[codisp.index] += codisp
            np.add.at(index, codisp.index.values, 1)
        avg_codisp /= index
        alg_scores = avg_codisp
        fpr_alg, tpr_alg, thresholds_alg = metrics.roc_curve(y, alg_scores, pos_label=1)
        #plt.plot(fpr_alg,tpr_alg, 'r')
        thresh_len = len(fpr_alg)
        sample_thresh = np.int_( [k * thresh_len/10 for k in range(10)] )
        sample_thresh = np.concatenate( [sample_thresh, np.asarray([thresh_len-1]) ])
        #plt.scatter(fpr_alg[sample_thresh],tpr_alg[sample_thresh], c = 'r', marker = "*", s=100, label='RRCF')
        plt.plot(fpr_alg[sample_thresh],tpr_alg[sample_thresh], c = 'r', marker = "*", markersize=5, label='RRCF')
        auc_all[j,2] = metrics.roc_auc_score(y, alg_scores)

        print('\n******kNN*******\n')
        clf = KNN()
        clf.fit(X)
        alg_scores = clf.decision_scores_ 
        fpr_alg, tpr_alg, thresholds_alg = metrics.roc_curve(y, alg_scores, pos_label=1)
        thresh_len = len(fpr_alg)
        sample_thresh = np.int_( [k * thresh_len/10 for k in range(10)] )
        sample_thresh = np.concatenate( [sample_thresh, np.asarray([thresh_len-1]) ])
        plt.plot(fpr_alg[sample_thresh],tpr_alg[sample_thresh], c = 'y', marker = "P", markersize=5, label='kNN')
        auc_all[j,5] = metrics.roc_auc_score(y, alg_scores)
        
        print('\n******PCA*******\n')
        clf = PCA()
        clf.fit(X)
        alg_scores = clf.decision_scores_ 
        fpr_alg, tpr_alg, thresholds_alg = metrics.roc_curve(y, alg_scores, pos_label=1)
        thresh_len = len(fpr_alg)
        sample_thresh = np.int_( [k * thresh_len/10 for k in range(10)] )
        sample_thresh = np.concatenate( [sample_thresh, np.asarray([thresh_len-1]) ])
        plt.plot(fpr_alg[sample_thresh],tpr_alg[sample_thresh], c = 'b', marker = ".", markersize=5, label='PCA')
        auc_all[j,6] = metrics.roc_auc_score(y, alg_scores)

        print('\n******DTM*******\n')
        rng = np.random.RandomState(42)
        max_samples = min(20, X.shape[0])
        bag_neigh = 1
        clf_spDTM = BaggingRegressor(base_estimator=DTM(n_neighbors=bag_neigh, contamination=0.1), n_estimators=1,max_samples=max_samples, bootstrap=False, random_state=rng)
        y_score_spDTM = clf_spDTM.fit(X, y).predict(X)
        fpr_alg, tpr_alg, thresholds_alg = metrics.roc_curve(y, -y_score_spDTM, pos_label=1)
        thresh_len = len(fpr_alg)
        sample_thresh = np.int_( [k * thresh_len/10 for k in range(10)] )
        sample_thresh = np.concatenate( [sample_thresh, np.asarray([thresh_len-1]) ])
        plt.plot(fpr_alg[sample_thresh],tpr_alg[sample_thresh], c = 'm', marker = "X", markersize=15, label='DTM')
        auc_all[j,7] = metrics.roc_auc_score(y, y_score_spDTM)
        
    file_name = 'experiment_results/' + datasets[i] + '.pdf'
    
    for k in range(0,8):
        print('{:.4f}\t'.format( auc_all[j,k] ))
    print('\n')

    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc='best')
    plt.title('ROC curve')
    # plt.show()
    f.savefig(file_name, bbox_inches='tight')
                
    file_name = 'experiment_results/' + datasets[i] + '_results_plot.mat'
    sio.savemat(file_name, {'auc_all':auc_all})