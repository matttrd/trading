import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def get_price_changes_distribution(prices, log=False, save_path=None):
    if log:
        changes = np.log(prices.iloc[1:].values /  prices.iloc[:-1].values)
    else:
        changes = prices.diff()

    ### standardize
    zscore = (changes - changes.mean()) / changes.std()

    z_distr = np.random.randn(100000)
    plt.clf()
    sns.kdeplot(z_distr)
    sns.histplot(zscore, stat="density")
    # plt.legend(["Standard Distr", "Z-score"])
    if save_path:
        plt.savefig(f"{save_path}/price_changes_distribution_imbalance", format=".png")
        plt.close()
    else:
        plt.show()


def look_data_imbalance(df_var, var_name="volume", keep_var="minute", save_path=None):

    '''
    Look for the volume imbalance for hours, minutes, or seconds
    data: pd.Dataframe 
    
    keep_var: Variable not marginalized

    '''

    datetime = df_var.index
    hour    = datetime.hour
    second  = datetime.second
    minute  = datetime.minute

    df_plot = pd.DataFrame()
    df_plot[var_name] = df_var.values
    df_plot["second"] = second.values
    df_plot["minute"] = minute.values
    df_plot["hour"] = hour.values
    
    plt.clf()
    plt.figure(figsize=(15,10))
    sns.barplot(data=df_plot, x=keep_var,y=var_name)#, stat="density" 
    if save_path:
        plt.savefig(f"{save_path}/{var_name}_imbalance", format=".png")
        plt.close()
    else:
        plt.show()

