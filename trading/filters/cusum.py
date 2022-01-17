import numba
from numba import njit, types
import pandas as pd
import numpy as np


'''
It it is worth to say that all the implementations HERE are not for real-time use. 
For real-time trading we should implement a different version.
'''


def cusum(time_series, th):
    '''
    Extract events where cusum filter cuts the threshold
    time_series: (pd.Series) Raw time series
    th: (float) Threshold
    
    Returns the series of time events
    '''

    t_events,s_pos,s_neg = [],0,0
    diff = time_series.diff()
    for i in diff.index[1:]:
        s_pos,s_neg = max(0,s_pos+diff.loc[i]),min(0,s_neg+diff.loc[i])
    if s_neg < -th:
        s_neg = 0; t_events.append(i)
    elif s_pos > th:
        s_pos = 0; t_events.append(i)
    return pd.DatetimeIndex(t_events)