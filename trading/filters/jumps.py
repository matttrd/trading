import numba
from numba import njit, types
import pandas as pd
import numpy as np


def soft_pos_jumps(time_series, win_len, threshold, perc):
    out = np.zeros_like(time_series).astype(np.float32)
    _soft_pos_jumps(time_series, win_len, threshold, perc, out)
    return out

def soft_neg_jumps(time_series, win_len, threshold, perc):
    out = np.zeros_like(time_series).astype(np.float32)
    _soft_pos_jumps(-time_series.astype(np.float32), win_len, threshold, perc, out)
    return out

# @njit((types.float32[:],types.int64,types.int64,types.float64,types.boolean, types.float32[:]))
# def _soft_pos_jumps(time_series, horizon, step, threshold, perc, out):
#     '''
#     '''

#     molecule = np.arange(len(time_series))
#     hrzns=range(1, horizon, step)

#     for dt0 in molecule:
#         if dt0+max(hrzns) > time_series.shape[0]: continue
#         deltas = np.zeros(len(hrzns))
#         for j, hrzn in enumerate(hrzns):
#             dt1 = time_series[dt0+hrzn-1]
#             window = time_series[dt0:dt1]
#             delta = window[-1] - window[0]
#             if perc:
#                 delta = delta / (window[0] + 1e-6)
#             deltas[j] = delta
#         dt1 = deltas.argmax()
#         if deltas.max() >= threshold:
#             out[dt0:dt1] = 1

@njit((types.float32[:],types.int64,types.float64,types.boolean, types.float32[:]))
def _soft_pos_jumps(time_series, horizon, threshold, perc, out):
    '''
    Look-back
    '''

    ts_len = len(time_series)
    min_idx = 0
    i = 1
    while(i < ts_len):
        #     # I need to recompute the min
        #     min_idx = np.argmin(time_series[i-horizon:i])
        delta = time_series[i] - time_series[min_idx]
        if perc:
            delta = delta / time_series[min_idx]
        if delta >= threshold and i-min_idx <= horizon:
            out[min_idx:i] = 1
            min_idx = i
        elif delta <= 0:
            min_idx = i
        i+= 1


@njit((types.float32[:],types.int32, types.float64, types.boolean, types.boolean, types.float32[:]))
def _pos_jumps(time_series, win_len, threshold, perc, derivative, out):
    ts_len = len(time_series)
    max_change = 0.
    counter = 0
    for i in range(1,ts_len):
        diff = time_series[i] - time_series[i-1]
        if perc:
            diff = diff / time_series[i-1]
        if diff > 0.:
            counter += 1
            if perc:
                max_change *= (1 + diff)
            else:
                max_change += diff
        else:
            if perc:
                max_change = max_change - 1
            if derivative:
                max_change = max_change / counter
            if max_change >= threshold and counter <= win_len:
                for j in range(counter):
                    out[i-j] = 1
            counter = 0       

@njit
def _neg_jumps(time_series, win_len,threshold, perc, derivative, out):
    return _pos_jumps(-time_series, win_len,threshold, perc, derivative, out)

def pos_jumps(time_series, win_len, threshold, perc, derivative, soft=False):
    time_series = time_series.astype(np.float32)
    out = np.zeros_like(time_series, dtype=np.float32)
    if soft:
        _soft_pos_jumps(time_series, win_len, threshold, perc, out)
    else:
        _pos_jumps(time_series, win_len,threshold, perc, derivative, out)
    return out

def neg_jumps(time_series, win_len, threshold, perc, derivative, soft=False):
    time_series = time_series.astype(np.float32)
    out = np.zeros_like(time_series, dtype=np.float32)
    if soft:
        soft_neg_jumps(time_series, win_len, threshold, perc, out)
    else:
        _neg_jumps(time_series, win_len, threshold, perc, derivative, out)
    return out

def detect_jumps(time_series_df, win_len, threshold, perc, derivative=False, pos=True, soft=False):
    if pos:
        out = pos_jumps(time_series, win_len, threshold, perc, derivative=derivative, soft=soft)
    else:   
        out = neg_jumps(time_series, win_len, threshold, perc, derivative=derivative, soft=soft)
    delta = out[1:] - out[:-1]
    t_events = np.where(delta == -1)
    return time_series_df.index[t_events]


