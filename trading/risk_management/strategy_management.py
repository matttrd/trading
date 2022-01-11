import numpy as np, 
import scipy.stats as ss


def bin_HR(sl: float, pt: float, freq: int, tSR: float):
    """
    Given a trading rule characterized by the parameters {sl,pt,freq},
    what's the min precision p required to achieve a Sharpe ratio tSR?
    1) Inputs
        sl: stop loss threshold
        pt: profit taking threshold
        freq: number of bets per year
        tSR: target annual Sharpe ratio
    2) Output
        p: the min precision rate p required to achieve tSR
    """

    a = (freq + tSR**2)*(pt - sl)**2
    b = (2*freq*sl - tSR**2*(pt - sl))*(pt - sl)
    c = freq*sl**2
    p = (-b + (b**2 - 4*a*c)**.5)/(2.*a)
    return p


def prob_failure(returns: pd.Series, freq: int, tSR: float) -> float:
    '''
    Derive probability that strategy may fail

    :param : returns (pd.Series) 
    :param : freq (int) : number of bets per year
    :param : tSR (float): target Sharpe Ratio
    '''

    r_pos, r_neg = returns[returns > 0].mean(), returns[returns <= 0].mean()
    p = returns[returns > 0].shape[0] / float(returns.shape[0])
    thres_p = bin_HR(r_neg, r_pos, freq, tSR)
    risk    = ss.norm.cdf(thres_p, p, p*(1-p)) # approximation to bootstrap
    return risk


