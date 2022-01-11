"""
This module contains functionality for determining bet sizes for investments based on machine learning predictions.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, moment
from trading.util.multiprocessing import mpPandasObj


def avg_active_signals(signals, num_threads):
    # compute the average signal among those active
    #1) time points where signals change (either one starts or one ends)
    t_pnts = set(signals['t1'].dropna().values)
    t_pnts = t_pnts.union(signals.index.values)
    t_pnts = list(t_pnts); t_pnts.sort()
    out    = mpPandasObj(mp_avg_active_signals,('molecule',t_pnts),num_threads, signals=signals)
    return out


def mp_avg_active_signals(signals, molecule):
    """
    At time loc, average signal among those still active.
    Signal is active if:
    a) issued before or at loc AND
    b) loc before signal's endtime, or endtime is still unknown (NaT).
    """
    out=pd.Series()
    for loc in molecule:
        df  = (signals.index.values<=loc)&((loc<signals['t1'])|pd.isnull(signals['t1']))
        act = signals[df].index
        if len(act)>0:out[loc]=signals.loc[act,'signal'].mean()
        else:out[loc]=0 # no signals active at this time
    return out


def discrete_signal(signal,step_size):
    # discretize signal
    signal = (signal/step_size).round()*step_size # discretize
    signal[signal >  1] =1 # cap
    signal[signal < -1] =-1 # floor
    return signal


def bet_size_probability(events, prob, num_classes, pred=None, step_size=0.0, average_active=False, num_threads=1):
    """
    Calculates the bet size using the predicted probability. Note that if 'average_active' is True, the returned
    pandas.Series will be twice the length of the original since the average is calculated at each bet's open and close.

    :param events: (pandas.DataFrame) Contains at least the column 't1', the expiry datetime of the product, with
     a datetime index, the datetime the position was taken.
    :param prob: (pandas.Series) The predicted probability.
    :param num_classes: (int) The number of predicted bet sides.
    :param pred: (pd.Series) The predicted bet side. Default value is None which will return a relative bet size
     (i.e. without multiplying by the side).
    :param step_size: (float) The step size at which the bet size is discretized, default is 0.0 which imposes no
     discretization.
    :param average_active: (bool) Option to average the size of active bets, default value is False.
    :param num_threads: (int) The number of processing threads to utilize for multiprocessing, default value is 1.
    :return: (pandas.Series) The bet size, with the time index.
    """
    # get signals from predictions
    if prob.shape[0]==0:
        return pd.Series()
    #1) generate signals from multinomial classification (one-vs-rest, OvR)
    signal = (prob-1./num_classes)/(prob*(1.-prob))**.5 # t-value of OvR
    size   = (2*norm.cdf(signal)-1)
    signal = size * pred if pred else size
    if 'side' in events:
        signal*=events.loc[signal.index,'side'] # meta-labeling
    
    #2) compute average signal among those concurrently open
    df = signal.to_frame('signal').join(events[['t1']],how='left')
    if average_active:
        df = avg_active_signals(df, num_threads)
    signal = discrete_signal(signal=df, step_size=step_size)
    return signal

