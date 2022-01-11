import pandas as pd
import numpy as np
from trading.util.multiprocessing import mpPandasObj

def get_bins(events, close):
    """
    Compute event's outcome (including side information, if provided).
    events is a DataFrame where:
    —events.index is event's starttime
    —events[’t1’] is event's endtime
    —events[’trgt’] is event's target
    —events[’side’] (optional) implies the algo's position side
    Case 1: (’side’ not in events): bin in (-1,1) <— label by price action
    Case 2: (’side’ in events):     bin in (0,1)  <— label by pnl (meta-labeling)
    """
    #1) prices aligned with events
    events_= events.dropna(subset=['t1'])
    px = events_.index.union(events_['t1'].values).drop_duplicates()
    px = close.reindex(px,method='bfill')
    #2) create out object
    out = pd.DataFrame(index=events_.index)
    out['ret'] = px.loc[events_['t1'].values].values/px.loc[events_.index]-1
    if 'side' in events_:
        out['ret'] *= events_['side'] # meta-labeling
    out['bin'] = np.sign(out['ret'])
    if 'side' in events_:
        out.loc[out['ret'] <= 0,'bin'] = 0 # meta-labeling
    return out


def apply_pt_sl_on_t1(close, events, pt_sl, molecule): 
    # apply stop loss/profit taking, if it takes place before t1 (end of event)
    events_ = events.loc[molecule]
    out     = events_[['t1']].copy(deep=True)
    if pt_sl[0] > 0:
        pt = pt_sl[0]*events_['trgt']
    else:
        pt = pd.Series(index=events.index) # NaNs
    if pt_sl[1] > 0:
        sl =- pt_sl[1]*events_['trgt']
    else:
        sl = pd.Series(index=events.index) # NaNs
    for loc,t1 in events_['t1'].fillna(close.index[-1]).iteritems():
        df0 = close[loc:t1] # path prices
        df0 = (df0/close[loc]-1)*events_.at[loc,'side'] # path returns
        out.loc[loc,'sl'] = df0[df0 < sl[loc]].index.min() # earliest stop loss.
        out.loc[loc,'pt'] = df0[df0 > pt[loc]].index.min() # earliest profit taking.
    return out


def add_vertical_barrier(t_events, close, num_days=0, num_hours=0, num_minutes=0, num_seconds=0):
    """
    Adding a Vertical Barrier
    For each index in t_events, it finds the timestamp of the next price bar at or immediately after
    a number of days num_days. This vertical barrier can be passed as an optional argument t1 in get_events.
    This function creates a series that has all the timestamps of when the vertical barrier would be reached.
    :param t_events: (pd.Series) Series of events (e.g. symmetric CUSUM filter)
    :param close: (pd.Series) Close prices
    :param num_days:    (int) Number of days to add for vertical barrier
    :param num_hours:   (int) Number of hours to add for vertical barrier
    :param num_minutes: (int) Number of minutes to add for vertical barrier
    :param num_seconds: (int) Number of seconds to add for vertical barrier
    :return: (pd.Series) Timestamps of vertical barriers
    """
    t1 = close.index.searchsorted(t_events+pd.Timedelta(days=num_days, 
                                                       hours=num_hours,
                                                     minutes=num_minutes,
                                                     seconds=num_seconds))
    t1 = t1[t1 < close.shape[0]]
    t1 = pd.Series(close.index[t1], index=t_events[:t1.shape[0]]) # NaNs at end
    return t1


def get_events(close, t_events, pt_sl, trgt, min_ret, num_threads, t1=False, side=None):
    #1) get target
    trgt = trgt.loc[trgt.index.intersection(t_events)]
    trgt = trgt[trgt > min_ret] # min return
    #2) get t1 (max holding period)
    if t1 is False: 
        t1 = pd.Series(pd.NaT, index=t_events)
    #3) form events object, apply stop loss on t1
    if side is None: 
        side_, pt_sl_ = pd.Series(1., index=trgt.index),[pt_sl[0],pt_sl[0]]
    else:
        side_, pt_sl_ = side.loc[trgt.index],pt_sl[:2]
    events=pd.concat({'t1':t1, 'trgt':trgt, 'side':side_}, axis=1).dropna(subset=['trgt'])
    df0=mpPandasObj(apply_pt_sl_on_t1, ('molecule',events.index), num_threads, 
                    close=close, events=events, pt_sl=pt_sl_)
    events['t1'] = df0.dropna(how='all').min(axis=1) # pd.min ignores nan
    if side is None:
        events = events.drop('side',axis=1)
    return events