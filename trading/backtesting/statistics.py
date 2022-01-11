import warnings
import pandas as pd
import scipy.stats as ss
import numpy as np



def avg_aum(positions: pd.Series) -> float:
    """
    Returns the average 

     :param positions: (pd.Series) position series with timestamps as indices
    """


def ratio_of_longs(positions: pd.Series) -> tuple:
    """
    :param positions: (pd.Series) position series with timestamps as indices
    :return: tuple where the first value is the fraction of long positions and the 
    second value is the fraction of number of longs vs shorts

    The first value tells what is the imbalanced in dollars of long positions
    The second value tells what is the imbalanced in numbers of long positions
    """

    buys  = positions[positions >= 0]
    sells = positions[positions < 0]

    return buys.sum() / sells.sum(), len(buys) / len(sells)


def timing_of_flattening_and_flips(target_pos: pd.Series) -> pd.DatetimeIndex:
    """
    Derives the timestamps of flattening or flipping trades from a pandas series
    of target positions. Can be used for position changes analysis, such as
    frequency and balance of position changes.

    Flattenings - times when open position is bing closed (final target position is 0).
    Flips - times when positive position is reversed to negative and vice versa.

    :param target_pos: (pd.Series) Target position series with timestamps as indices
    :return: (pd.DatetimeIndex) Timestamps of trades flattening, flipping and last bet
    """

    df   = target_pos[target_pos==0].index
    df1  = target_pos.shift(1); df1 = df1[df1!=0].index
    bets = df.intersection(df1) # flattening
    df   = target_pos.iloc[1:]*target_pos.iloc[:-1].values
    bets = bets.union(df[df < 0].index).sort_values() # target_pos flips
    if target_pos.index[-1] not in bets:
        bets = bets.append(target_pos.index[-1:]) # last bet
    return bets


def average_holding_period(target_pos: pd.Series) -> float:
    """
    Estimates the average holding period (in days) of a strategy, given a pandas series
    of target positions using average entry time pairing algorithm.

    Idea of an algorithm:

    * entry_time = (previous_time * weight_of_previous_position + time_since_beginning_of_trade * increase_in_position )
      / weight_of_current_position
    * holding_period ['holding_time' = time a position was held, 'weight' = weight of position closed]
    * res = weighted average time a trade was held

    :param target_pos: (pd.Series) Target position series with timestamps as indices
    :return: (float) Estimated average holding period, NaN if zero or unpredicted
    """

    hp, t_entry = pd.DataFrame(columns=['dT','w']), 0.
    p_diff, t_diff = target_pos.diff(), (target_pos.index-target_pos.index[0])/np.timedelta64(1,'D')
    for i in range(1,target_pos.shape[0]):
        if p_diff.iloc[i]*target_pos.iloc[i-1]>=0: # increased or unchanged
            if target_pos.iloc[i]!=0:
                t_entry=(t_entry*target_pos.iloc[i-1]+t_diff[i]*p_diff.iloc[i])/target_pos.iloc[i]
        else: # decreased
            if target_pos.iloc[i]*target_pos.iloc[i-1]<0: # flip
                hp.loc[target_pos.index[i],['dT','w']]=(t_diff[i]-t_entry,abs(target_pos.iloc[i-1]))
                t_entry=t_diff[i] # reset entry time
            else:
                hp.loc[target_pos.index[i],['dT','w']]=(t_diff[i]-t_entry,abs(p_diff.iloc[i]))
    if hp['w'].sum()>0:
        hp=(hp['dT']*hp['w']).sum()/hp['w'].sum()
    else:
        hp=np.nan
    return hp


def bets_concentration(returns: pd.Series) -> float:
    """
    Derives the concentration of returns from given pd.Series of returns.

    Algorithm is based on Herfindahl-Hirschman Index where return weights
    are taken as an input.

    :param returns: (pd.Series) Returns from bets
    :return: (float) Concentration of returns (nan if less than 3 returns)
    """
    if returns.shape[0]<=2:
        return np.nan
    wght = returns/returns.sum()
    hhi  = (wght**2).sum()
    hhi  = (hhi-returns.shape[0]**(-1))/(1.-returns.shape[0]**(-1))
    return hhi


def all_bets_concentration(returns: pd.Series, frequency: str = 'M') -> tuple:
    """
    Given a pd.Series of returns, derives concentration of positive returns, negative returns
    and concentration of bets grouped by time intervals (daily, monthly etc.).
    If after time grouping less than 3 observations, returns nan.

    Properties or results:

    * low positive_concentration ⇒ no right fat-tail of returns (desirable)
    * low negative_concentration ⇒ no left fat-tail of returns (desirable)
    * low time_concentration ⇒ bets are not concentrated in time, or are evenly concentrated (desirable)
    * positive_concentration == 0 ⇔ returns are uniform
    * positive_concentration == 1 ⇔ only one non-zero return exists

    :param returns: (pd.Series) Returns from bets
    :param frequency: (str) Desired time grouping frequency from pd.Grouper
    :return: (tuple of floats) Concentration of positive, negative and time grouped concentrations
    """
    rHHI_pos = bets_concentration(returns[returns>=0]) # concentration of positive returns per bet
    rHHI_neg = bets_concentration(returns[returns<0]) # concentration of negative returns per bet
    tHHI     = bets_concentration(returns.groupby(pd.Grouper(freq=frequency)).count()) # concentr. bets/month
    return rHHI_pos, rHHI_neg, tHHI



