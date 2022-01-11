import warnings
import pandas as pd
import scipy.stats as ss
import numpy as np



def drawdown_and_time_under_water(returns: pd.Series, dollars: bool = False) -> tuple:
    """
    Calculates drawdowns and time under water for pd.Series of either relative price of a
    portfolio or dollar price of a portfolio.

    Intuitively, a drawdown is the maximum loss suffered by an investment between two consecutive high-watermarks.
    The time under water is the time elapsed between an high watermark and the moment the PnL (profit and loss)
    exceeds the previous maximum PnL. We also append the Time under water series with period from the last
    high-watermark to the last return observed.

    Return details:

    * Drawdown series index is the time of a high watermark and the value of a
      drawdown after it.
    * Time under water index is the time of a high watermark and how much time
      passed till the next high watermark in years. Also includes time between
      the last high watermark and last observation in returns as the last element.

    :param returns: (pd.Series) Returns from bets
    :param dollars: (bool) Flag if given dollar performance and not returns.
                    If dollars, then drawdowns are in dollars, else as a %.
    :return: (tuple of pd.Series) Series of drawdowns and time under water
    """

    # compute series of drawdowns and the time under water associated with them
    df0 = returns.to_frame('pnl')
    df0['hwm'] = series.expanding().max()
    df1 = df0.groupby('hwm').min().reset_index()
    df1.columns = ['hwm','min']
    df1.index = df0['hwm'].drop_duplicates(keep='first').index # time of hwm
    df1 = df1[df1['hwm'] > df1['min']] # hwm followed by a drawdown
    if dollars:
        dd=df1['hwm']-df1['min']
    else:
        dd=1-df1['min']/df1['hwm']
    tuw=((df1.index[1:]-df1.index[:-1])/np.timedelta64(1,'Y')).values# in years
    tuw=pd.Series(tuw, index=df1.index[:-1])
    return dd,tuw


def sharpe_ratio(returns: pd.Series, entries_per_year: int = 252, risk_free_rate: float = 0) -> float:
    """
    Calculates annualized Sharpe ratio for pd.Series of normal or log returns.

    Risk_free_rate should be given for the same period the returns are given.
    For example, if the input returns are observed in 3 months, the risk-free
    rate given should be the 3-month risk-free rate.

    :param returns: (pd.Series) Returns - normal or log
    :param entries_per_year: (int) Times returns are recorded per year (252 by default)
    :param risk_free_rate: (float) Risk-free rate (0 by default)
    :return: (float) Annualized Sharpe ratio
    """
    returns = returns - risk_free_rate
    sharpe_ratio = returns.nanmean()/returns.nanstd()
    asr = sharpe_ratio*entries_per_year**.5
    return asr


def information_ratio(returns: pd.Series, benchmark: float = 0, entries_per_year: int = 252) -> float:
    """
    Calculates annualized information ratio for pd.Series of normal or log returns.

    Benchmark should be provided as a return for the same time period as that between
    input returns. For example, for the daily observations it should be the
    benchmark of daily returns.

    It is the annualized ratio between the average excess return and the tracking error.
    The excess return is measured as the portfolio’s return in excess of the benchmark’s
    return. The tracking error is estimated as the standard deviation of the excess returns.

    :param returns: (pd.Series) Returns - normal or log
    :param benchmark: (float) Benchmark for performance comparison (0 by default)
    :param entries_per_year: (int) Times returns are recorded per year (252 by default)
    :return: (float) Annualized information ratio
    """
    return sharpe_ratio(returns, entries_per_year, benchmark)


def probabilistic_sharpe_ratio(observed_sr: float, benchmark_sr: float, number_of_returns: int,
                               skewness_of_returns: float = 0, kurtosis_of_returns: float = 3) -> float:
    """
    Calculates the probabilistic Sharpe ratio (PSR) that provides an adjusted estimate of SR,
    by removing the inflationary effect caused by short series with skewed and/or
    fat-tailed returns.

    Given a user-defined benchmark Sharpe ratio and an observed Sharpe ratio,
    PSR estimates the probability that SR ̂is greater than a hypothetical SR.
    - It should exceed 0.95, for the standard significance level of 5%.
    - It can be computed on absolute or relative returns.

    :param observed_sr: (float) Sharpe ratio that is observed
    :param benchmark_sr: (float) Sharpe ratio to which observed_SR is tested against
    :param number_of_returns: (int) Times returns are recorded for observed_SR
    :param skewness_of_returns: (float) Skewness of returns (0 by default)
    :param kurtosis_of_returns: (float) Kurtosis of returns (3 by default)
    :return: (float) Probabilistic Sharpe ratio
    """
    delta = ((observed_sr - benchmark_sr) * np.sqrt(number_of_returns-1))
    adj = 1 / np.sqrt(1 - skewness_of_returns * observed_sr + (kurtosis_of_returns-1)/4 * observed_sr**2)
    return ss.norm.cdf(adj * delta)


def deflated_sharpe_ratio(observed_sr: float, sr_estimates: list, number_of_returns: int,
                          skewness_of_returns: float = 0, kurtosis_of_returns: float = 3,
                          estimates_param: bool = False, benchmark_out: bool = False) -> float:
    """
    Calculates the deflated Sharpe ratio (DSR) - a PSR where the rejection threshold is
    adjusted to reflect the multiplicity of trials. DSR is estimated as PSR[SR∗], where
    the benchmark Sharpe ratio, SR∗, is no longer user-defined, but calculated from
    SR estimate trails.

    DSR corrects SR for inflationary effects caused by non-Normal returns, track record
    length, and multiple testing/selection bias.
    - It should exceed 0.95, for the standard significance level of 5%.
    - It can be computed on absolute or relative returns.

    Function allows the calculated SR benchmark output and usage of only
    standard deviation and number of SR trails instead of full list of trails.

    :param observed_sr: (float) Sharpe ratio that is being tested
    :param sr_estimates: (list) Sharpe ratios estimates trials list or
        properties list: [Standard deviation of estimates, Number of estimates]
        if estimates_param flag is set to True.
    :param number_of_returns: (int) Times returns are recorded for observed_SR
    :param skewness_of_returns: (float) Skewness of returns (0 by default)
    :param kurtosis_of_returns: (float) Kurtosis of returns (3 by default)
    :param estimates_param: (bool) Flag to use properties of estimates instead of full list
    :param benchmark_out: (bool) Flag to output the calculated benchmark instead of DSR
    :return: (float) Deflated Sharpe ratio or Benchmark SR (if benchmark_out)
    """
    gamma = 0.5772156649
    if estimates_param:
        std, N = sr_estimates
    else:
        std, N = np.std(sr_estimates), len(sr_estimates)

    sr0 = std * ((1-gamma) * ss.norm.ppf(1-1/N) + gamma * ss.norm.ppf(1 - 1/(N*np.exp(1))))
    
    if benchmark_out:
        return sr0
    return probabilistic_sharpe_ratio(observed_sr, sr0, number_of_returns, 
                                      skewness_of_returns, kurtosis_of_returns)
    

def minimum_track_record_length(observed_sr: float, benchmark_sr: float,
                                skewness_of_returns: float = 0,
                                kurtosis_of_returns: float = 3,
                                alpha: float = 0.05) -> float:
    """
    Calculates the minimum track record length (MinTRL) - "How long should a track
    record be in order to have statistical confidence that its Sharpe ratio is above
    a given threshold?”

    If a track record is shorter than MinTRL, we do not have enough confidence
    that the observed Sharpe ratio is above the designated Sharpe ratio threshold.

    MinTRLis expressed in terms of number of observations, not annual or calendar terms.

    :param observed_sr: (float) Sharpe ratio that is being tested
    :param benchmark_sr: (float) Sharpe ratio to which observed_SR is tested against
    :param number_of_returns: (int) Times returns are recorded for observed_SR
    :param skewness_of_returns: (float) Skewness of returns (0 by default)
    :param kurtosis_of_returns: (float) Kurtosis of returns (3 by default)
    :param alpha: (float) Desired significance level (0.05 by default)
    :return: (float) Minimum number of track records
    """

    inv_psr = ss.norm.ppf(1 - alpha)
    delta_sr = (observed_sr - benchmark_sr)
    adj = np.sqrt(1 - skewness_of_returns * observed_sr + (kurtosis_of_returns-1)/4 * observed_sr**2)
    return (inv_psr / delta_sr * adj)**2 + 1