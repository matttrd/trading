from abc import ABC, abstractmethod
from typing import Tuple, Union, Generator, Iterable, Optional

from more_itertools import sliced
from .data_reading import load_file
from itertools import chain
import pandas as pd


def create_final_bars(obars, file_path_or_df, start_date=None, end_date=None):
    bars = obars.batch_run(file_path_or_df, start_date=start_date, end_date=end_date)
    bars = pd.concat(bars).rename(columns={"isBuyerMaker": "cum_buy_maker"})
    # bars.set_index("datetime", inplace=True)
    bars['vwap'] = bars['cum_dollar'] / bars['volume']
    return bars 


def _split_data_frame_in_batches(df: pd.DataFrame, chunksize: int) -> list:
    # pylint: disable=invalid-name
    """
    Splits df into chunks of chunksize
    :param df: (pd.DataFrame) Dataframe to split
    :param chunksize: (int) Number of rows in chunk
    :return: (list) Chunks (pd.DataFrames)
    """

    index_slices = sliced(range(len(df)), chunksize)
    chunks = []
    for index_slice in index_slices:
        chunks.append(df.iloc[index_slice])
    
    return chuncks


class BaseBars(ABC):
    """
    Abstract base class which contains the structure which is shared between the various standard and information
    driven bars. There are some methods contained in here that would only be applicable to information bars but
    they are included here so as to avoid a complicated nested class structure.
    """

    def __init__(self, metric: str, batch_size: int = 2e7, source="binance"):
        """
        Constructor

        :param metric: (str) Type of imbalance bar to create. Example: dollar_imbalance.
        :param batch_size: (int) Number of rows to read in from the csv, per batch.
        """
        self.metric = metric
        self.batch_size = batch_size
        self.cache = None
        self.source = source


    def batch_run(self, file_path_or_df: Union[str, Iterable[str], pd.DataFrame], verbose: bool = True, to_csv: bool = False,
                  start_date: str = None, end_date: str = None, output_path: Optional[str] = None) -> Union[pd.DataFrame, None]:
        """
        Reads csv file(s) or pd.DataFrame in batches and then constructs the financial data structure in the form of a DataFrame.
        The csv file or DataFrame must have only 3 columns: date_time, price, & volume.
        :param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame containing
                                raw tick data  in the format[date_time, price, volume]
        :param verbose: (bool) Flag whether to print message on each processed batch or not
        :param to_csv: (bool) Flag for writing the results of bars generation to local csv file, or to in-memory DataFrame
        :param output_path: (bool) Path to results file, if to_csv = True
        :return: (pd.DataFrame or None) Financial data structure
        """
        df_generator = self._batch_iterator(file_path_or_df, start_date=start_date, end_date=end_date)
        # from IPython import embed
        # embed()
        # df_generator = list(df_generator)
        list_data = self._extract_bars(df_generator)
        return list_data

    def _batch_iterator(self, file_path_or_df: Union[str, Iterable[str], pd.DataFrame],
                        start_date: str = None, end_date: str = None ) -> Generator[pd.DataFrame, None, None]:
        """
        :param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame
                                containing raw tick data in the format[date_time, price, volume]
        """
        # if isinstance(file_path_or_df, pd.DataFrame):
        #     return _split_data_frame_in_batches(file_path_or_df)
        def inner_iterator(file_path_or_df,  start_date=start_date, end_date=end_date):
            if isinstance(file_path_or_df, list):
                for file in file_path_or_df:
                    if file.endswith(".h5"):
                        df = load_file(file, self.source, self.batch_size, start_date=start_date, end_date=end_date)
                        yield df

        return chain.from_iterable(inner_iterator(file_path_or_df, start_date=start_date, end_date=end_date))

    def _read_first_row(self, file_path: str):
        """
        :param file_path: (str) Path to the csv file containing raw tick data in the format[date_time, price, volume]
        """

        pass

    def run(self, data: Union[list, tuple, pd.DataFrame]) -> list:
        """
        Reads a List, Tuple, or Dataframe and then constructs the financial data structure in the form of a list.
        The List, Tuple, or DataFrame must have only 3 attrs: date_time, price, & volume.

        :param data: (list, tuple, or pd.DataFrame) Dict or ndarray containing raw tick data in the format[date_time, price, volume]

        :return: (list) Financial data structure
        """ 


        pass

    @abstractmethod
    def _extract_bars(self, data: pd.DataFrame) -> list:
        """
        This method is required by all the bar types and is used to create the desired bars.

        :param data: (pd.DataFrame) Contains 3 columns - date_time, price, and volume.
        :return: (list) Bars built using the current batch.
        """

    @abstractmethod
    def _reset_cache(self):
        """
        This method is required by all the bar types. It describes how cache should be reset
        when new bar is sampled.
        """

    @staticmethod
    def _assert_csv(test_batch: pd.DataFrame):
        """
        Tests that the csv file read has the format: date_time, price, volume, isBuyerMaker.
        If not then the user needs to create such a file. This format is in place to remove any unwanted overhead.

        :param test_batch: (pd.DataFrame) The first row of the dataset.
        """
        assert test_batch.shape[1] == 4, 'Must have only 4 columns in csv: date_time, price, & volume.'
        assert isinstance(test_batch.iloc[0, 1], float), 'price column in csv not float.'
        assert not isinstance(test_batch.iloc[0, 2], str), 'volume column in csv not int or float.'

        try:
            pd.to_datetime(test_batch.iloc[0, 0])
        except ValueError:
            raise ValueError('csv file, column 0, not a date time format:',
                             test_batch.iloc[0, 0])


    def _apply_tick_rule(self, price: float) -> int:
        """
        Applies the tick rule as defined on page 29 of Advances in Financial Machine Learning.

        :param price: (float) Price at time t
        :return: (int) The signed tick
        """

        pass

    def _get_imbalance(self, price: float, signed_tick: int, volume: float) -> float:
        """
        Advances in Financial Machine Learning, page 29.

        Get the imbalance at a point in time, denoted as Theta_t

        :param price: (float) Price at t
        :param signed_tick: (int) signed tick, using the tick rule
        :param volume: (float) Volume traded at t
        :return: (float) Imbalance at time t
        """

        pass