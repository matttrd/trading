import pandas as pd

def get_millisec_from_str(resolution):
    dict_r = {'L':1, 'S': 1000, 'T': 60*1000, 'H': 3600*1000, 'D': 24*3600*1000}
    res_millis = dict_r[resolution[-1]]
    return int(resolution[:-1]) * res_millis

def save_to_csv(data: pd.DataFrame, output_path: str):
    data.to_csv(output_path)
    return