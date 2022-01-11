"""
Various volatility estimators
"""
import pandas as pd
import numpy as np
from .fast_ewma import ewma

# pylint: disable=redefined-builtin

def get_daily_vol(close, lookback=100):
    """
    Computes the daily volatility at intraday estimation points.

    In practice we want to set profit taking and stop-loss limits that are a function of the risks involved
    in a bet. Otherwise, sometimes we will be aiming too high (tao ≫ sigma_t_i,0), and sometimes too low
    (tao ≪ sigma_t_i,0 ), considering the prevailing volatility. 
    Computes the daily volatility at intraday estimation points, applying a span of lookback days to an exponentially weighted moving
    standard deviation.

    Note: This function is used to compute dynamic thresholds for profit taking and stop loss limits.

    :param close: (pd.Series) Closing prices
    :param lookback: (int) Lookback period to compute volatility
    :return: (pd.Series) Daily volatility value
    """

    # daily vol, reindexed to close
    df=close.index.searchsorted(close.index-pd.Timedelta(days=1))
    df=df[df>0]
    df=pd.Series(close.index[df-1], index=close.index[close.shape[0]-df.shape[0]:])
    df=close.loc[df.index]/close.loc[df.values].values-1 # daily returns
    df=df.ewm(span=lookback).std()
    return df.dropna()


def get_parkinson_vol(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """
    Parkinson volatility estimator

    :param high: (pd.Series): High prices
    :param low: (pd.Series): Low prices
    :param window: (int): Window used for estimation
    :return: (pd.Series): Parkinson volatility
    """
    df = np.sqrt(1 / (4 * np.log(2)) * ((high / low).log()**2).rolling(window).mean())
    return df


def get_garman_klass_vol(open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
                         window: int = 20) -> pd.Series:
    """
    Garman-Klass volatility estimator

    :param open: (pd.Series): Open prices
    :param high: (pd.Series): High prices
    :param low: (pd.Series): Low prices
    :param close: (pd.Series): Close prices
    :param window: (int): Window used for estimation
    :return: (pd.Series): Garman-Class volatility
    """
    df = np.sqrt(((0.5 * (high / low).log() ** 2) - (2 * np.log(2) - 1) * (close / open).log() ** 2).rolling(window).mean())
    return df


def get_yang_zhang_vol(open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
                       window: int = 20) -> pd.Series:
    """

    Yang-Zhang volatility estimator

    :param open: (pd.Series): Open prices
    :param high: (pd.Series): High prices
    :param low: (pd.Series): Low prices
    :param close: (pd.Series): Close prices
    :param window: (int): Window used for estimation
    :return: (pd.Series): Yang-Zhang volatility
    """

    pass