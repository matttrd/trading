import pandas as pd

def load_file(file_path, source, batch_size, start_date=None, end_date=None):
    read_func = loaders["binance"]
    return read_func(file_path, batch_size, start_date=start_date, end_date=end_date)


def read_binance_data(file_path, batch_size, start_date=None, end_date=None):
    # chunks = pd.read_csv(file_path, header=None, names=[ "id","price", "volume","quoteQty", "datetime", 
    #                             "isBuyerMaker", "isBestMatch"], usecols=[1,2,4,5], 
    #                             chunksize=batch_size, iterator=False)
    where=""
    if start_date:
        where+="index >= '{}'".format(start_date)
    if end_date:
        if where == "":
            where+= "index <= '{}'".format(start_date)
        else:
            where+= " and index <= '{}'".format(end_date)
        # where+="index >= '1980-04-04' and index<= '1980-05-01'"
    chunks = pd.read_hdf(file_path, 
                          # header=None, 
                          where=where if len(where) > 0 else None,
                          # names=["tradeId", "price", "qty", "quoteQty",
                          #         "time", "isBuyerMaker", "isBestMatch"],
                          # usecols=[1,2,4,5], 
                          chunksize=batch_size, 
                          iterator=False)
    # reader
    for chunk in chunks:
        yield chunk

loaders = {"binance": read_binance_data}
