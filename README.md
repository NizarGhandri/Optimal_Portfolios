# A Comparative Analysis of Different Asset Allocation Techniques
This repository is for the final project for the EPFL's Financial Big Data class.
In this project we investigate multiple ways to build optimal portfolios relying mainly on big data analysis.

## Group Members:
- Ana Lucia Carrizo Delgado, Msc. Data Science
- Nizar Ghandri, Msc. Data Science

## Due date: 30th of January of 2022

##  execution.ipynb
In this notebook we find the performance of the equally weighted portfolio and the value weighted portfolio. 

## MeanVariancePortolio.ipynb 
In this notebook we detail the process to computing the tangency portfolio using a rolling window and filtered and unfiltered covariance matrices.

## portfolios folder 
Here we find the executable scripts to run the different asset allocation models. 
- equally_weighted.py
- market_weights.py
- mean_variance.py

#### reinforcement_learning folder:
- agent.py
- environment.py

## config.py
This file contains the configuration attributes used during preprocessing and training.

## data_querier.py
This file contains the methods to query our data from Yahoo Finance.
