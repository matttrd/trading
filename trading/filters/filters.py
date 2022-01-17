import numba
from numba import njit, types
import pandas as pd
import numpy as np


'''
It it is worth to say that all the implementations HERE are not for real-time use. 
For real-time trading we should implement a different version.
'''

def combine_filters(time_events):
    '''
    performace logical AND of different columns (filters)
    '''
    conds = time_events.sum(1)
    return (conds == time_events.shape[1]).astype(np.int)

