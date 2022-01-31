import os
import pandas as pd
from data_querier import SharesOutStandingQuerier
import logging 
from functools import reduce














class MarketWeighted():


    def __init__(self, cfg):  
        self.cfg = cfg  
        self.data = self.load_data()
        self.sharesout = SharesOutStandingQuerier(self.data.columns, ("2012-01-01", "2021-12-31"), username="ghandri")


    def load_data(self):
        return self.preprocess(pd.concat([pd.read_parquet(os.path.join(self.cfg.data_dir, f))["Close"] for f in os.listdir(self.cfg.data_dir)]))

    def preprocess(self, x, percent0=0.7, percent1=0.2):
        tmp = x.dropna(thresh=int(percent0*x.shape[1])).dropna(axis=1, thresh=int(percent1*x.shape[0])).fillna(method="ffill")
        dropped = set(x.columns) - set(tmp.columns) 
        logging.info("Preprocessing dropped the following stocks %s ".format(["-".join(list(dropped))]))
        return tmp

    def __call__(self):
        weights = self.compute_weights()
        returns = ((self.data.diff()/self.data) + 1)[weights.columns].fillna(1) * weights
        return returns.sum(axis=1).cumprod()

    def compute_weights(self):
        mapper = map(lambda x: x[1].drop(columns=["ticker"]).set_index("date").rename(columns={"shrout":x[0]}),
               filter(lambda x: len(x[1]) == self.data.shape[0], # some are missing certain dates 
               self.sharesout.sharesout.merge(self.sharesout.permcos, on="permco").drop(columns=["permco"]).groupby(["ticker"]).__iter__()))

        reducer = reduce(lambda x, y: x.merge(y, left_index=True, right_index=True), mapper) * self.data
        return reducer.div(reducer.sum(axis=1), axis=0) 
        




     

