import matplotlib.pyplot as mpl
import numpy as np
import pandas as pd
from IPython import embed

def plotMinFFD(df0, save_path):
    from statsmodels.tsa.stattools import adfuller
    out=pd.DataFrame(columns=['adfStat','pVal','lags','nObs','95% conf','corr'])
    for d in np.linspace(0,1,11):
        df1=np.log(df0[['close']]).resample('1D').last() # downcast to daily obs
        df2=fracDiff_FFD(df1,d,thres=1e-3)
        corr=np.corrcoef(df1.loc[df2.index,'close'],df2['close'])[0,1]
        try:
            df2=adfuller(df2['close'],maxlag=1,regression='c',autolag=None)
        except:
            embed()
        out.loc[d]=list(df2[:4])+[df2[4]['5%']]+[corr] # with critical value
    out.to_csv(save_path+'/testMinFFD.csv')
    out[['adfStat','corr']].plot(secondary_y='adfStat')
    mpl.axhline(out['95% conf'].mean(),linewidth=1,color='r',linestyle='dotted')
    mpl.savefig(save_path+'/testMinFFD.png')
    return

def fracDiff_FFD(series,d,thres=1e-5):
    '''
    Constant width window (new solution)
    Note 1: thres determines the cut-off weight for the window
    Note 2: d can be any positive fractional, not necessarily bounded [0,1].
    '''
    #1) Compute weights for the longest series
    w=getWeights(d, series.shape[0], thres)
    width=len(w)-1
    # if d >= 0.1:
    #     embed()
    #2) Apply weights to values
    df={}
    for name in series.columns:
        # embed()
        seriesF,df_=series[[name]].fillna(method='ffill').dropna(),pd.Series()
        for iloc1 in range(width,seriesF.shape[0]):
            loc0,loc1=seriesF.index[iloc1-width],seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1,name]):continue # exclude NAs
            df_[loc1]=np.dot(w.T,seriesF.loc[loc0:loc1])[0,0]
        df[name]=df_.copy(deep=True)
    df=pd.concat(df,axis=1)
    return df

def getWeights(d,size, thresh):
    # thresh >0 drops insignificant weights
    w=[1.]
    for k in range(1,size):
        w_=-w[-1]/k*(d-k+1)
        w.append(w_)
    w=np.array(w[::-1])#.reshape(-1,1)
    return w[np.abs(w)>=thresh].reshape(-1,1)