#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install dask')


# In[1]:


from dask import dataframe
import pandas as pd
import yfinance as yf
import os 
import logging 
import numpy as np


# In[1]:


class MarkowitzPortfolio():
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
    
    def __call__(self, num_portfolios = 20000, seed = 101):
        returns = self.data.pct_change() # mean daily returns
        mean_daily_returns = returns.mean()
        returns_annual = mean_daily_returns * 250 # returns annual

        cov_daily = returns.cov() # daily covariance of returns
        cov_annual = cov_daily * 250  # annual covariance of returns

        results = np.zeros((3, num_portfolios))
        ret_data = returns[1:]
        num_assets = cov_annual.shape[0]

        stock_weights = []

        # set random seed for reproduction's sake
        np.random.seed(seed)

        for i in range(num_portfolios):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)

            portfolio_return = np.dot(weights, returns_annual)
            portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))

            results[0, i] = portfolio_return # portfolio returns
            results[1, i] = portfolio_std_dev # portfolio volatility
            results[2, i] = results[0, i] / results[1, i] # portfolio's sharpe ratio

            stock_weights.append(weights)

        # a dictionary for Returns and Risk values for each portfolio
        portfolio = {'Returns': results[0],
                     'Volatility': results[1],
                     'Sharpe Ratio': results[2]}

        # extend original dictionary to accomodate each ticker and weight in the portfolio
        for counter, symbol in enumerate(cov_annual.columns):
            portfolio[symbol] = [w[counter] for w in stock_weights]

        # make a nice dataframe of the extended dictionary
        portfolio_df = pd.DataFrame(portfolio)
        # get better labels for desired arrangement of columns
        column_order = ['Returns', 'Volatility', 'Sharpe Ratio'] + [stock for stock in cov_annual.columns]
        # reorder dataframe columns
        portfolio_df = portfolio_df[column_order]

        # find minimum volatility and maximum sharpe values in the portfolio
        min_volatility = portfolio_df['Volatility'].min()
        max_sharpe = portfolio_df['Sharpe Ratio'].max()

        # min-variance portfolio
        min_variance_portfolio = portfolio_df.loc[portfolio_df['Volatility'] == min_volatility]
        wts_min_variance = min_variance_portfolio.drop(['Returns', 'Volatility', 'Sharpe Ratio'], axis = 1)
        wts_min_variance = wts_min_variance.values.tolist()
        wts_min_variance = wts_min_variance[0]
        weighted_returns_min_variance = (wts_min_variance * ret_data)

        # sharpe portfolio
        sharpe_portfolio = portfolio_df.loc[portfolio_df['Sharpe Ratio'] == max_sharpe]
        wts_sharpe = sharpe_portfolio.drop(['Returns', 'Volatility', 'Sharpe Ratio'], axis = 1)
        wts_sharpe = wts_sharpe.values.tolist()
        wts_sharpe = wts_sharpe[0]
        weighted_returns_sharpe = (wts_sharpe * ret_data)

        portfolio_returns_min_variance = weighted_returns_min_variance.sum(axis=1)
        cumulative_returns_min_variance = (portfolio_returns_min_variance + 1).cumprod()

        portfolio_returns_sharpe = weighted_returns_sharpe.sum(axis=1)
        cumulative_returns_sharpe = (portfolio_returns_sharpe + 1).cumprod()

        return cumulative_returns_min_variance, cumulative_returns_sharpe

