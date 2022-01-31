from abc import ABC, abstractmethod


class Env(ABC):


    @abstractmethod
    def step():
        pass 


    @abstractmethod
    def __iter__():
        pass 