from typing import Union, Iterable, Optional

import numpy as np
import pandas as pd
import datetime

from .base_bars import BaseBars, create_final_bars
from .utils import get_millisec_from_str, save_to_csv
from tqdm import tqdm
from IPython import embed

# pylint: disable=too-many-instance-attributes

class TimeBars(BaseBars):
    """
    Contains all of the logic to construct the time bars. This class shouldn't be used directly.
    Use get_time_bars instead
    """

    def __init__(self, resolution: str, batch_size: int = 100000000, start_date: str = None,
                       end_date: str = None):
        """
        Constructor
        :param resolution: (str) Type of bar resolution: ['D', 'H', 'MIN', 'S']
        :param batch_size: (int) Number of rows to read in from the csv, per batch
        """

        super(TimeBars, self).__init__("time", batch_size)
        self.resolution = resolution
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

    def _extract_bars(self, data: Union[list, tuple, np.ndarray]) -> list:
        """
        For loop which compiles time bars.
        :param data: (tuple) Contains 3 columns - date_time, price, and volume.
        :return: (list) Extracted bars
        """
        frames = []
        def sample_fun(xs, y): return np.int64(xs / y)
        resolution = get_millisec_from_str(self.resolution)

        for frame in tqdm(data):
            if "volume" not in frame.columns:
                    frame.rename(columns={'qty': 'volume'}, inplace=True)
                    
            frame = self.append_cache(frame)
            # frame = frame.sort_values("datetime")
            # first = frame['datetime'].iloc[0]
            first = frame.index[0]

            # cum_ref = (frame['datetime'] - first)
            cum_ref = frame.index - first
            # if frame['datetime'].iloc[-1] - frame['datetime'].iloc[0] < resolution:
            if (frame.index[-1] - frame.index[0])/datetime.timedelta(milliseconds=1) < resolution:
                # batch is not big enough to create a sample
                self._set_cache(frame)
            else:
                # frame = frame.reset_index(drop=True)
                frame["cum_ticks"]  = np.ones(len(frame))
                frame['cum_dollar'] = frame["volume"] * frame["price"]
                frame["datetime"] = frame.index
                grouping = sample_fun(cum_ref, datetime.timedelta(milliseconds=resolution))
                # gframe = frame.groupby(grouping).agg({'price': 'ohlc', 'volume': 'sum', 
                #             'isBuyerMaker': 'sum', 'cum_ticks': 'sum', 'cum_dollar': 'sum', 'datetime': 'first'})
                gframe = frame.groupby(grouping).agg({'datetime': 'first','price': 'ohlc', 'volume': 'sum', 
                            'isBuyerMaker': 'sum', 'cum_ticks': 'sum', 'cum_dollar': 'sum'})
                gframe.columns = list(map(lambda x: x[1], gframe.columns))
                gframe = gframe.set_index("datetime", drop=True)
                # gframe["datetime"] = gframe["datetime"].apply(lambda x: pd.to_datetime(x,unit='ms'))
                ## remove last sample to be sure that it will be complete the next time
                # reminder = frame[frame["datetime"] > int(gframe["datetime"].iloc[-1].timestamp() * 1000)]
                reminder = frame[frame.index > gframe.index[-1]]
                if len(reminder) > 0:
                    self._set_cache(reminder)
                    gframe = gframe.iloc[:-1]
                frames.append(gframe)
        return frames


def get_time_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], resolution: str = '1D', batch_size: int = 100000000,
                  start_date: str = None, end_date: str = None, verbose: bool = True, to_csv: bool = False, output_path: Optional[str] = None):
    """
    Creates Time Bars: date_time, open, high, low, close, volume, cum_buy_volume, cum_ticks, cum_dollar_value.
    :param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame containing raw tick data
                            in the format[date_time, price, volume]
    :param resolution: (str) Resolution type ('D', 'H', 'MIN', 'S')
    :param num_units: (int) Number of resolution units (3 days for example, 2 hours)
    :param batch_size: (int) The number of rows per batch. Less RAM = smaller batch size.
    :param verbose: (int) Print out batch numbers (True or False)
    :param to_csv: (bool) Save bars to csv after every batch run (True or False)
    :param output_path: (str) Path to csv file, if to_csv is True
    :return: (pd.DataFrame) Dataframe of time bars, if to_csv=True return None
    """
    obars = TimeBars(resolution=resolution, 
                     batch_size=batch_size)

    # bars = obars.batch_run(file_path_or_df, 
    #                  start_date = start_date,
    #                  end_date   = end_date)
    # bars = pd.concat(bars).rename(columns={"isBuyerMaker": "cum_buy_maker"})
    bars = create_final_bars(obars, file_path_or_df, start_date=start_date, end_date=end_date)
    if to_csv:
        save_to_csv(bars, output_path)

    return bars