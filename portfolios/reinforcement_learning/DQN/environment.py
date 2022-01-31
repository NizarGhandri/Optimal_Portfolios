from abc import ABC, abstractmethod
from utils import *
import logging
import os 
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import from_numpy
from ...reinforcement_learning.env import Env




class States(Dataset):



    def __init__(self, cfg, mode, **kwargs) -> None:
        self.cfg = cfg
        self.mode = mode
        self.data, self.data_test = self.load_data()
        self.estimators_max_window = kwargs.get("max_window", 15)

    def __len__(self):
        if(self.mode == DataModes.TRAINING):
            return len(self.data)
        else:
            return len(self.data_test)

    def __getitem__(self, idx):
        if(self.mode == DataModes.TRAINING):
            return self.data[idx]
        else:
            return self.data_test[idx]



    def load_data(self):
        preprocessed = self.preprocess(pd.concat([pd.read_parquet(os.path.join(self.cfg.data_dir, f))["Close"] for f in os.listdir(self.cfg.data_dir)]))
        preprocessed = from_numpy(preprocessed.values).float()
        return preprocessed[:-self.cfg.test_size], preprocessed[-self.cfg.test_size:]

    def preprocess(self, x, percent0=0.5, percent1=0.2):
        tmp = x.dropna(axis=0, thresh=int(percent0*x.shape[1])).dropna(axis=1, thresh=int(percent1*x.shape[0])).fillna(method="ffill")
        dropped = set(x.columns) - set(tmp.columns) 
        logging.info("Preprocessing dropped the following stocks" + "-".join(list(dropped)))
        tmp = np.log(((tmp.diff()/tmp) + 1).fillna(1))
        return tmp


    def generate_indicators(self, idx, x):
        return x




class MarketEnviormentTorch(Env):


    def __init__(self, mode, cfg) -> None:
        super().__init__()
        self.dataset = States(mode, cfg)
        self.lose_penalty = -99
        self.state_dim = len(self.dataset[0])
        self.action_dim = self.state_dim
        self.min_episodes = 100



    def __iter__(self):
        return enumerate(DataLoader(self.dataset, batch_size=1, shuffle=False, )) 


    def step(self, idx, action):
        reward = action@self.dataset[idx]
        done = (reward < np.log(2)) and (idx >= self.min_episodes)
        next_state = self.dataset[idx+1] if idx+1 < len(self.dataset) else None
        return next_state, reward, done 



    
    


