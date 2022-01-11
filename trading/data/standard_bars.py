"""
This module contains the functions to help users create structured financial data from raw unstructured data,
in the form of time, tick, volume, and dollar bars.
"""

# Imports
from typing import Union, Iterable, Optional

import numpy as np
import pandas as pd

from .base_bars import BaseBars, create_final_bars
from .utils import get_millisec_from_str, save_to_csv

from tqdm import tqdm


class StandardBars(BaseBars):
    """
    Contains all of the logic to construct the standard bars from chapter 2. This class shouldn't be used directly.
    We have added functions to the package such as get_dollar_bars which will create an instance of this
    class and then construct the standard bars, to return to the user.

    This is because we wanted to simplify the logic as much as possible, for the end user.
    """

    def __init__(self, metric: str, threshold: int = 50000, batch_size: int = 20000000):
        """
        Constructor

        :param metric: (str) Type of run bar to create. Example: "dollar"
        :param threshold: (int) Threshold at which to sample
        :param batch_size: (int) Number of rows to read in from the csv, per batch
        """
        super(StandardBars, self).__init__(metric, batch_size)
        self.threshold = threshold
        self._reset_cache()

    def _reset_cache(self):
        """
        Implementation of abstract method _reset_cache for standard bars
        """
        self.cache = None

    def append_cache(self, frame):
        if self.cache is not None:
            frame = pd.concat((self.cache, frame))
            self._reset_cache() 
        return frame

    def _set_cache(self, new_cache):
        self.cache = new_cache

    def _extract_bars(self, data: Union[list, pd.DataFrame]) -> list:
        """
        For loop which compiles the various bars: dollar, volume, or tick.
        We did investigate the use of trying to solve this in a vectorised manner but found that a For loop worked well.

        :param data: (tuple) Contains 4 columns - date_time, price, volume and maker_buy.
        :return: (list) Extracted bars
        """
        def sample_fun(xs, y): return np.int64(xs / y)# * y

        frames = []

        for frame in tqdm(data):
            if "volume" not in frame.columns:
                    frame.rename(columns={'qty': 'volume'}, inplace=True)

            frame = self.append_cache(frame)
            
            if self.metric == "tick":
                reference = np.ones(len(frame))
                ref_str = "cum_ticks"
            elif self.metric == "volume":
                reference = frame["volume"]
                ref_str = "volume"
            elif self.metric == "dollar":
                reference = frame["volume"] * frame["price"]
                ref_str = "cum_dollar"
            else:
                raise ValueError("[tick, volume or dollar]")
            
            cum_ref = np.cumsum(reference)

            if cum_ref.iloc[-1] < self.threshold:
                # batch is not big enough to create a sample
                self._set_cache(frame)
                continue
            else:
                # frame = frame.reset_index(drop=True)
                frame["cum_ticks"] = np.ones(len(frame))
                frame['cum_dollar'] = frame["volume"] * frame["price"]
                frame["datetime"] = frame.index
                grouping = sample_fun(cum_ref, self.threshold)
                gframe = frame.groupby(grouping).agg({'price': 'ohlc', 'volume': 'sum', 'isBuyerMaker': 'sum',
                                                      'datetime': 'first', 'cum_ticks': 'sum', 'cum_dollar': 'sum'})
                gframe.columns = list(map(lambda x: x[1], gframe.columns))
                gframe = gframe.set_index("datetime", drop=True)

                # gframe["datetime"] = gframe["datetime"].apply(lambda x: pd.to_datetime(x,unit='ms'))
                # TODO: for crypto it's easy but for markets with closign hours we should also check where the time is finished
                EPS = 0.01
                if gframe[ref_str].iloc[-1] < self.threshold / (1 + EPS):
                    self._set_cache(frame[grouping==grouping[-1]])
                    gframe = gframe.iloc[:-1]
                    # if gframe[ref_str].min() < 95:
                    #     embed()
                frames.append(gframe)
        return frames


def get_dollar_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], threshold: Union[float, pd.Series] = 70000000,
                    start_date: str = None, end_date: str = None, batch_size: int = 20000000, verbose: bool = True, to_csv: bool = False, output_path: Optional[str] = None):
    """
    Creates the dollar bars: date_time, open, high, low, close, volume, cum_buy_volume, cum_ticks, cum_dollar_value, cum_buy_maker

    Following the paper "The Volume Clock: Insights into the high frequency paradigm" by Lopez de Prado, et al,
    it is suggested that using 1/50 of the average daily dollar value, would result in more desirable statistical
    properties.

    :param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame containing raw tick data
                            in the format[date_time, price, volume]
    :param threshold: (float, or pd.Series) A cumulative value above this threshold triggers a sample to be taken.
                      If a series is given, then at each sampling time the closest previous threshold is used.
                      (Values in the series can only be at times when the threshold is changed, not for every observation)
    :param batch_size: (int) The number of rows per batch. Less RAM = smaller batch size.
    :param verbose: (bool) Print out batch numbers (True or False)
    :param to_csv: (bool) Save bars to csv after every batch run (True or False)
    :param output_path: (str) Path to csv file, if to_csv is True
    :return: (pd.DataFrame) Dataframe of dollar bars
    """
    obars = StandardBars(metric="dollar", threshold=threshold, batch_size=batch_size)
    bars = create_final_bars(obars, file_path_or_df, start_date=start_date, end_date=end_date)
    if to_csv:
        save_to_csv(bars, output_path)
    return bars


def get_volume_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], threshold: Union[float, pd.Series] = 70000000,
                    start_date: str = None, end_date: str = None, batch_size: int = 20000000, verbose: bool = True, to_csv: bool = False, output_path: Optional[str] = None):
    """
    Creates the volume bars: date_time, open, high, low, close, volume, cum_buy_volume, cum_ticks, cum_dollar_value.

    Following the paper "The Volume Clock: Insights into the high frequency paradigm" by Lopez de Prado, et al,
    it is suggested that using 1/50 of the average daily volume, would result in more desirable statistical properties.

    :param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame containing raw tick data
                            in the format[date_time, price, volume]
    :param threshold: (float, or pd.Series) A cumulative value above this threshold triggers a sample to be taken.
                      If a series is given, then at each sampling time the closest previous threshold is used.
                      (Values in the series can only be at times when the threshold is changed, not for every observation)
    :param batch_size: (int) The number of rows per batch. Less RAM = smaller batch size.
    :param verbose: (bool) Print out batch numbers (True or False)
    :param to_csv: (bool) Save bars to csv after every batch run (True or False)
    :param output_path: (str) Path to csv file, if to_csv is True
    :return: (pd.DataFrame) Dataframe of volume bars
    """
    obars = StandardBars(metric="volume", threshold=threshold, batch_size=batch_size)
    bars = create_final_bars(obars, file_path_or_df, start_date=start_date, end_date=end_date)
    if to_csv:
        save_to_csv(bars, output_path)
    return bars


def get_tick_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], threshold: Union[float, pd.Series] = 70000000,
                  start_date: str = None, end_date: str = None, batch_size: int = 20000000, verbose: bool = True, to_csv: bool = False, output_path: Optional[str] = None):
    """
    Creates the tick bars: date_time, open, high, low, close, volume, cum_buy_volume, cum_ticks, cum_dollar_value.

    :param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame containing raw tick data
                             in the format[date_time, price, volume]
    :param threshold: (float, or pd.Series) A cumulative value above this threshold triggers a sample to be taken.
                      If a series is given, then at each sampling time the closest previous threshold is used.
                      (Values in the series can only be at times when the threshold is changed, not for every observation)
    :param batch_size: (int) The number of rows per batch. Less RAM = smaller batch size.
    :param verbose: (bool) Print out batch numbers (True or False)
    :param to_csv: (bool) Save bars to csv after every batch run (True or False)
    :param output_path: (str) Path to csv file, if to_csv is True
    :return: (pd.DataFrame) Dataframe of volume bars
    """
    obars = StandardBars(metric="tick", threshold=threshold, batch_size=batch_size)
    bars = create_final_bars(obars, file_path_or_df, start_date=start_date, end_date=end_date)
    if to_csv:
        save_to_csv(bars, output_path)
    return bars