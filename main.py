import os
import numpy as np
import pandas as pd
from glob2 import glob
from trading.data.standard_bars import get_dollar_bars, get_volume_bars
from trading.data.time_bars import get_time_bars
from IPython import embed
from trading.features.frac_diff import plotMinFFD
from trading.labeling.triple_barrier import get_events, get_bins

# def get_dataframe(path):
#     csv_files = glob(path)
#     # print(csv_files)
#     dataframes = []
#     for file in csv_files:
#         tmp = pd.read_csv(file, names=["otime",
#                                        "open",
#                                        "high",
#                                        "low",
#                                        "close",
#                                        "volume",
#                                        "ctime",
#                                        "qav",
#                                        "not",
#                                        "tbbv",
#                                        "tbqv",
#                                        "ignore"])
#         tmp = tmp.set_index("otime", drop=True)
#         tmp.index = pd.to_datetime(tmp.index, unit='ms')
#         dataframes.append(tmp)

#     df = pd.concat(dataframes)
#     df = df.sort_index()
#     return df

# def main():
#     path = "/home/matteo/crypto/binance-public-data/data/futures/BTCUSDT/um/monthly/klines/1d/*.csv"
#     save_path = "/home/matteo/trading_results/"
#     df = get_dataframe(path)
#     plotMinFFD(df,save_path)

def main():
    files = sorted(glob('/home/matteo/crypto/binance-public-data/data/spot/monthly/trades/ETHUSDT/*.h5'))
    # files = files[-2:-1]
    # print(files)
    # dfs_dollar = get_dollar_bars(files, threshold=3.654307e+07)
    # dfs_dollar = get_dollar_bars(files, threshold=1e+06, start_date='2018-07-01', end_date='2018-08-01')
    dfs_time = get_time_bars(files, resolution='1T', start_date='2018-07-01', end_date='2018-08-01')#, batch_size=10000)
    # dfs_time.index = dfs_time.index.round('1H')

    # from trading.labeling.triple_barrier import get_events, get_bins
    # from trading.util.volatility import get_daily_vol

    # close = dfs_time["close"]
    # trgt  = get_daily_vol(close)
    # out   = get_events(close,close.index,[0.05, 0.05],trgt,0.01,1)
    # save_path = "/home/matteo/trading_results/"
    # plotMinFFD(dfs_dollar,save_path)

    # t1    = add_vertical_barrier(close.index, close, num_days=3, num_hours=0, num_minutes=0, num_seconds=0)
    out   = get_events(close,close.index,[0.05, 0.05],trgt,0.01,1, t1=None)

    embed()


if __name__ == "__main__":
    main()



