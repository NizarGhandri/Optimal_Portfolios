
import numpy as np
import wrds
import dask
import pandas as pd
import yfinance as yf
import os
import logging



TABLE_NUM = 0
STOCK_COLUMN = "Symbol"



class DataQuerierWrds: 

    def __init__ (self, company_permcos_name, dates, **kwargs): 
        self.company_permcos_name = pd.Dataframe({"ticker": company_permcos_name})
        self.dates = dates
        self.db=wrds.Connection(wrds_username=kwargs["username"])
        if (kwargs["first_connection"]):
            self.db.create_pgpass_file()
        self.__get_permcos()
        self.futures = [self.__query_company(*params) for params in self.permcos]
        os.environ["OMP_NUM_THREADS"] = "1"
        

        
    
    def __get_permcos(self):
        try:
            query_res = self.db.raw_sql("select  permco , ticker, namedt, nameenddt,comnam "
                                "from crsp.stocknames "
                                "where namedt <'2009-01-01' and nameenddt >'2009-01-01'")
            self.permcos = query_res[["permco", "ticker"]].merge(self.company_permcos_name, on="ticker").values
        finally:
            self.db.close()




    def __call__ (self):
        return dask.compute(self.futures)
        


    @dask.delayed
    def __query_company(self, permco, name):
        params= {'permco': permco, 'low': self.dates[0], 'high': self.dates[1]}
        stock=self.db.raw_sql("select * "
           "from crsp.dsf "
           "where permco in {permco} "
           "and date >= {low}"
            "and date <= {high}", params=params)
        stock["ret"] = np.log(stock["ret"] + 1) #get gross return
        stock=stock.rename(index=stock["date"], columns={"ret": name})
        return stock





class DataQuerierYF: 

    def __init__ (self, cfg, load_on_init=True, save=True, **kwargs):
        self.cfg = cfg
        self.company_list = pd.read_html(self.cfg.link)[TABLE_NUM][STOCK_COLUMN].tolist()
        logging.info("Attempting load for %d stocks from %s".format([len(self.company_list), self.cfg.link]))
        self.from_params = len(kwargs)
        self.save_data = save
        self.kwargs = kwargs
        if (load_on_init):
            self()
            


    
    def __call__(self):
        self.__get_tickers(self.kwargs)
        if(self.save_data):
            self.__save()
        logging.info("Loaded %d stocks from %s".format([len(self.stock_hist.columns), self.cfg.link]))
        return self.stock_hist


    def __save(self): 
        self.stock_hist.to_parquet(self.cfg.data_path)
        
        


    def __get_tickers(self, kwargs):
        if (self.from_params):
            self.stock_hist = yf.download(  
                                        tickers = self.company_list,
                                        **kwargs
                                    )
        else: 
            self.stock_hist = yf.download(  
                                        tickers = self.company_list,
                                        **self.cfg.params
                                    )
        return self.stock_hist


        
