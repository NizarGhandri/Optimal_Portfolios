#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install dask')

from dask import dataframe
import pandas as pd
import yfinance as yf
import os 
import logging 
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
import seaborn as sns
import matplotlib.pyplot as plt
import bahc

class MeanVariancePortfolio():
    def __init__(self, cfg):
        self.cfg = cfg
        self.data = self.load_data()
    
    def load_data(self):
        return self.preprocess(pd.concat([pd.read_parquet(os.path.join(self.cfg.data_dir, f))['Close'] for f in  os.listdir(self.cfg.data_dir)]))
    
    def preprocess(self, x, percent0 = 0.5, percent1 = 0.2):
        tmp = x.dropna(axis=0, thresh=int(percent0*x.shape[1])).dropna(axis=1, thresh=int(percent1*x.shape[0])).fillna(method="ffill")
        dropped = set(x.columns) - set(tmp.columns) 
        logging.info("Preprocessing dropped the following stocks" + "-".join(list(dropped)))
        return tmp
    
    def clean_portfolio(self):
        self.data.fillna(method = 'ffill', inplace = True)
        columns_missing = self.data.columns[self.data.isna().sum() > 10].values
        self.data.drop(columns_missing, inplace= True, axis=1)
        self.data.fillna(method = 'bfill', inplace = True)
        return self
    
    def min_var_portfolio(mu, cov, target_return):
        inv_cov = np.linalg.inv(cov)
        ones = np.ones(len(mu))[:, np.newaxis]

        a = ones.T @ inv_cov @ ones
        b = mu.T @ inv_cov @ ones
        c = mu.T.to_numpy() @ inv_cov @ mu

        a = a[0][0]
        b = b.loc['mu', 0]
        c = c.loc[0, 'mu']

        num1 = (a * inv_cov @ mu - b * inv_cov @ ones) * target_return
        num2 = (c * inv_cov @ ones- b * inv_cov @ mu)
        den = a*c - b**2

        w = (num1 + num2) / den

        var = w.T.to_numpy() @ cov.to_numpy() @ w.to_numpy()
        return w, var**0.5
    
    def __call__(self, training_period = 10, num_assets = 50, rf = 0.05, bahc_bool = False, plot_bool = True): 
        def get_log_returns_matrix(portfolio_data):
            log_returns_matrix = np.log(portfolio_data/portfolio_data.shift(1))
            log_returns_matrix.fillna(0, inplace=True)
            log_returns_matrix = log_returns_matrix[(log_returns_matrix.T != 0).any()]
            return log_returns_matrix
        
        def get_stocks_reordered(log_returns_matrix):
            cov_daily = log_returns_matrix.cov()
            stocks = list(cov_daily.columns)
            link = linkage(cov_daily, 'average')
            reordered_cov_daily = cov_daily.copy()
            stocks_reordered = [stocks[i] for i in leaves_list(link)]
            reordered_cov_daily = reordered_cov_daily[stocks_reordered]
            reordered_cov_daily = reordered_cov_daily.reindex(stocks_reordered)
            return stocks_reordered, reordered_cov_daily
        
        def get_bahc_cov_matrix(log_returns_matrix, stocks_reordered):
            cov_bahc = pd.DataFrame(bahc.filterCovariance(np.array(log_returns_matrix).T))
            cov_bahc.columns, cov_bahc.index = log_returns_matrix.columns, log_returns_matrix.columns
            cov_bahc = cov_bahc[stocks_reordered]
            cov_bahc = cov_bahc.reindex(stocks_reordered)
            return cov_bahc
        
        def get_weights(mu_vector, cov_matrix, rf):
            ones = np.ones(mu_vector.shape[0])[:, np.newaxis]
            num = np.linalg.inv(cov_matrix) @ (mu_vector - rf * ones)
            den = ones.T @ np.linalg.inv(cov_matrix) @ (mu_vector - rf * ones)
            w = (np.asarray(num) / np.asarray(den))
            weights = pd.DataFrame(index = mu_vector.index, columns = ['Weights'])
            weights['Weights'] = w
            return w, weights
        
        def get_cumulative_returns(w, log_returns_matrix):
            weighted_returns = (w.T * log_returns_matrix)
            portfolio_returns = weighted_returns.sum(axis=1)
            cumulative_returns = (portfolio_returns + 1).cumprod()
            return cumulative_returns
        
        def plot_cumulative_returns(cumulative_returns):
            fig = plt.figure()
            ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
            ax1.plot(cumulative_returns)
            ax1.set_xlabel('Date')
            ax1.set_ylabel("Cumulative Returns")
            ax1.set_title("Portfolio Cumulative Returns")
            plt.show()
        
        portfolio = self.clean_portfolio()
        portfolio_data = self.data.iloc[:, :num_assets]
        stocks_universe = portfolio_data.columns

        log_returns_matrix = get_log_returns_matrix(portfolio_data)
        mu_vector = pd.DataFrame(index = stocks_universe, columns = ['mu'])

        if bahc_bool == False:
            cov_matrix = log_returns_matrix.cov() * 252
        else:
            stocks_reordered, _ = get_stocks_reordered(log_returns_matrix)
            cov_matrix = get_bahc_cov_matrix(log_returns_matrix, stocks_reordered) * 252

        for stock in stocks_universe:
            series = portfolio_data[stock]
            log_returns = np.log(series/series.shift(1)).dropna()
            ann_log_return = np.sum(log_returns) / training_period
            mu_vector.loc[stock] = ann_log_return

        w_tangency, _ = get_weights(mu_vector, cov_matrix, rf)
        cumulative_returns_tangent = get_cumulative_returns(w_tangency, log_returns_matrix)
        
        if(plot_bool): plot_cumulative_returns(cumulative_returns_tangent)

        return cumulative_returns_tangent

