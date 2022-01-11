import numpy as np,pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples


"""
This module is devoled to produce a clustering algorithm based on K-Means and modified to select the effective 
number of strategies given the correlation matrix of strategies' returns as input.  

"""

def cluster_KMeans_base(corr, max_num_clusters=10, n_init=10):
    dist,silh=((1-corr.fillna(0))/2.)**.5,pd.Series() # distance matrix
    for init in range(n_init):
        for i in xrange(2,max_num_clusters+1): # find optimal num clusters
            kmeans_ = KMeans(n_clusters=i,n_jobs=1,n_init=1)
            kmeans_ = kmeans_.fit(dist)
            silh_ = silhouette_samples(dist,kmeans_.labels_)
            stat  = (silh_.mean()/silh_.std(),silh.mean()/silh.std())
            if np.isnan(stat[1]) or stat[0]>stat[1]:
                silh,kmeans = silh_, kmeans_
    
    n_clusters = len( np.unique( kmeans.labels_ ) )
    new_idx    = np.argsort(kmeans.labels_)
    corr1      = corr.iloc[new_idx] # reorder rows
    corr1      = corr1.iloc[:,new_idx] # reorder columns
    clstrs     = {i:corr.columns[np.where(kmeans.labels_==i)[0] ].tolist() for \
    i in np.unique(kmeans.labels_) } # cluster members
    silh = pd.Series(silh, index=dist.index)
    return corr1, clstrs, silh


def make_new_outputs(corr, clstrs, clstrs2):
    clstrs_new, new_indx = {},[]
    for i in clstrs.keys():
        clstrs_new[len(clstrs_new.keys())] = list(clstrs[i])
    for i in clstrs2.keys():
        clstrs_new[len(clstrs_new.keys())] = list(clstrs2[i])
    map(new_indx.extend, clstrs_new.values())
    corr_new = corr.loc[new_indx,new_indx]
    dist    = ((1-corr.fillna(0))/2.)**.5
    kmeans_labels = np.zeros(len(dist.columns))
    for i in clstrs_new.keys():
        idxs = [dist.index.get_loc(k) for k in clstrs_new[i]]
        kmeans_labels[idxs] = i
    silh_new = pd.Series(silhouette_samples(dist, kmeans_labels), index=dist.index)
    return corr_new, clstrs_new, silh_new


def cluster_KMeans_top(corr, max_num_clusters=10, n_init=10):
    corr1, clstrs, silh = cluster_KMeans_base(corr, max_num_clusters=corr.shape[1]-1, n_init=n_init)
    cluster_tstats = {i : np.mean(silh[clstrs[i]]) / np.std(silh[clstrs[i]]) for i in clstrs.keys()}
    tstat_mean = np.mean(cluster_tstats.values())
    redo_clusters = [i for i in cluster_tstats.keys() if cluster_tstats[i]<tstat_mean]
    if len(redo_clusters)<= 2:
        return corr1, clstrs, silh
    else:
        keys_redo=[]; map(keys_redo.extend,[clstrs[i] for i in redo_clusters])
        corrTmp = corr.loc[keys_redo, keys_redo]
        meanRedoTstat = np.mean([cluster_tstats[i] for i in redo_clusters])
        corr2, clstrs2, silh2 = cluster_KMeans_top(corrTmp, \
        max_num_clusters=corrTmp.shape[1]-1, n_init=n_init)
        # Make new outputs, if necessary
        corr_new, clstrs_new, silh_new = make_new_outputs(corr, \
        {i:clstrs[i] for i in clstrs.keys() if i not in redo_clusters},clstrs2)
        new_tstat_mean = np.mean([np.mean(silh_new[clstrs_new[i]]) / np.std(silh_new[clstrs_new[i]]) \
        for i in clstrs_new.keys()])
        if new_tstat_mean <= meanRedoTstat:
            return corr1, clstrs, silh
        else:
            return corr_new, clstrs_new, silh_new