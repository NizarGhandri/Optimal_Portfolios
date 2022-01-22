from abc import ABC, abstractmethod
from utils import *
import logging
import os 
import pandas as pd
from torch.utils.data import Dataset, Dataloader
import numpy as np



class Env(ABC):


    @abstractmethod
    def step():
        pass 


    @abstractmethod
    def __iter__():
        pass 


class States(Dataset):



    def __init__(self, cfg, **kwargs) -> None:
        self.cfg = cfg
        self.data = self.load_data()
        self.estimators_max_window = kwargs.get("max_window", 15)

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass



    def load_data(self):
        return self.preprocess(pd.concat([pd.read_parquet(os.path.join(self.cfg.data_dir, f))["Close"] for f in os.listdir(self.cfg.data_dir)]))

    def preprocess(self, x, percent0=0.5, percent1=0.2):
        tmp = x.dropna(axis=0, thresh=int(percent0*x.shape[1])).dropna(axis=1, thresh=int(percent1*x.shape[0])).fillna(method="ffill")
        dropped = set(x.columns) - set(tmp.columns) 
        logging.info("Preprocessing dropped the following stocks" + "-".join(list(dropped)))
        tmp = np.log(((tmp.diff()/tmp) + 1).fillna(1))
        return tmp


    def generate_indicators(self):
        pass




class MarketEnviorment(Env):


    def __init__(self) -> None:
        super().__init__()
        self.dataset = States()



    def __iter__(self):
        return Dataloader(self.dataset, batch_size=1, shuffle=False, ) 


    def step(self, idx):
        return self.dataset[idx]



    
    


