from numba import njit

def cusum(time_series, th):
    '''
    Extract events where cusum filter cuts the threshold
    time_series: (np.array or pd.Series) Raw time series
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


def _soft_pos_jumps(time_series, ts_len, win_len, threshold, perc=False):
    '''
    Finds the maximum variation 
    '''
    ts = np.zeros_like(time_series)
    max_val = time_series[1]
    min_val = time_series[0]
    counter = 0
    i = 0
    while (True):
        for j in range(i, win_len):
            delta = max_val - min_val
            if perc:
                delta = delta / min_val
            if derivative:
                delta = delta / counter
            if delta >= threshold:
                ts[i] = 1

            if time_series[i] > max_val:



@njit
def _pos_jumps(time_series, ts_len, threshold, perc=False, derivative=False):
    ts = np.zeros_like(time_series)
    diff = time_series[1:] - time_series[:-1]
    if perc:
        diff = diff / time_series[:-1]
    max_change = 0.
    counter = 0
    for i in range(ts_len-1):
        if diff[i] >= 0:
            counter += 1
            if perc:
                max_change += diff[i]
            else:
                max_change *= (1 + diff[i])
        else:
            if derivative:
                max_change = max_change / counter
            if max_change >= threshold:
                ts[i] = 1
            counter = 0       
    return ts

@njit
def _neg_jumps(time_series, ts_len, threshold):
    return _pos_jumps(-time_series, ts_len, threshold)

def combine_filters(time_events):
    '''
    performace logical AND of different columns (filters)
    '''
    conds = time_events.sum(1)
    return (conds == time_events.shape[1]).astype(np.int)
     
def positive_jumps(time_series, win_len, th):
    diff = time_series.diff()
    for i in range(len(diff)):
        if     
