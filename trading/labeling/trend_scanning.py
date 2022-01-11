import statsmodels.api as sm1
import pandas as pd
import numpy as np


def t_val_lin_r(close):
    # tValue from a linear trend
    x=np.ones((close.shape[0],2))
    x[:,1]=np.arange(close.shape[0])
    ols=sm1.OLS(close,x).fit()
    return ols.tvalues[1]


def trend_scanning_labels(price_series: pd.Series, t_events: list = None, observation_window: int = 20,
                          min_sample_length: int = 5, step: int = 1) -> pd.DataFrame:
    """
    Trend scanning is both a classification and regression labeling technique.
    That can be used in the following ways:
    1. Classification: By taking the sign of t-value for a given observation we can set {-1, 1} labels to define the
       trends as either downward or upward.
    2. Classification: By adding a minimum t-value threshold you can generate {-1, 0, 1} labels for downward, no-trend,
       upward.
    3. The t-values can be used as sample weights in classification problems.
    4. Regression: The t-values can be used in a regression setting to determine the magnitude of the trend.
    The output of this algorithm is a DataFrame with t1 (time stamp for the farthest observation), t-value, returns for
    the trend, and bin.
    This function allows using both forward-looking and backward-looking window (use the look_forward parameter).
    :param price_series: (pd.Series) Close prices used to label the data set
    :param t_events: (list) Filtered events, array of pd.Timestamps
    :param observation_window: (int) Maximum look forward window used to get the trend value
    :param min_sample_length:  (int) Minimum sample length used to fit regression
    :param step: (int) Optimal t-value index is searched every 'step' indices
    :return: (pd.DataFrame) Consists of t1, t-value, ret, bin (label information). t1 - label endtime, tvalue,
        ret - price change %, bin - label value based on price change sign
    """
    if t_events:
        price_series = price_series.loc[t_events]
    
    datetime_index = price_series.index
    price_series   = price_series.reset_index(drop=True)
    molecule       = price_series.index
    out            = pd.DataFrame(index=molecule, columns=['t1','tVal','bin'])
    
    hrzns=range(min_sample_length, observation_window, step)
    for dt0 in molecule:
        df0   = pd.Series()
        iloc0 = price_series.index.get_loc(dt0)
        if iloc0+max(hrzns) > price_series.shape[0]: continue
        for hrzn in hrzns:
            dt1 = price_series.index[iloc0+hrzn-1]
            df1 = price_series.loc[dt0:dt1]
            df0.loc[dt1] = t_val_lin_r(df1.values)

        dt1 = df0.replace([-np.inf,np.inf,np.nan],0).abs().idxmax()
        out.loc[dt0,['t1','tVal','bin']] = datetime_index[df0.index[-1]], df0[dt1], np.sign(df0[dt1]) # prevent leakage
    # out['t1']  = pd.to_datetime(out['t1'])
    out['bin'] = pd.to_numeric(out['bin'], downcast="signed")
    out.index = datetime_index
    return out.dropna(subset=['bin'])

