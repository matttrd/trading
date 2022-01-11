import numpy as np
import pandas as pd
from util.clustering import cluster_KMeans_top


def cov2corr(cov):
    # Derive the correlation matrix from a covariance matrix
    std=np.sqrt(np.diag(cov))
    corr=cov/np.outer(std, std)
    corr[corr<-1],corr[corr>1]=-1,1 # numerical error
    return corr


def getIVP(cov, use_extended_terms=False):
    # Compute the minimum-variance portfolio
    ivp = 1./np.diag(cov)
    if use_extended_terms:
        n = float(cov.shape[0])
        corr = cov2corr(cov)
        # Obtain average off-diagonal correlation
        rho = (np.sum(np.sum(corr))-n)/(n**2-n)
        invSigma = np.sqrt(ivp)
        ivp -= rho*invSigma*np.sum(invSigma)/(1.+(n-1)*rho)
    ivp /= ivp.sum()
    return ivp


def cluster_strategies(returns: list, max_num_clusters = 10, n_init = 10):
    '''
    Cluster the N strategies to find the "effective" number of strategies K < N
   
   :param returns: List of pd.Series or numpy array
    '''
    corr, clstrs, silh = cluster_KMeans_top(corr, max_num_clusters=max_num_clusters, n_init=n_init)


def mixGaussians(mu1, mu2, sigma1, sigma2, prob1, n_obs):
    # Random draws from a mixture of gaussians
    returns1 = np.random.normal(mu1, sigma1, size=int(n_obs*prob1))
    returns2 = np.random.normal(mu2, sigma2, size=int(n_obs)-returns1.shape[0])
    returns  = np.append(returns1,returns2,axis=0)
    np.random.shuffle(returns)
    return returns
