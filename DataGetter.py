# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 20:19:25 2018

@author: Pranav Devarinti
"""

import csv
import pandas_datareader.data as web
from pytrends.request import TrendReq
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
Minmax = MinMaxScaler(feature_range=(-1,1))
def Get_Data(tickers,start_date,end_date):
    pytrends = TrendReq(hl='en-US', tz=360)
    total_dates = start_date + ' ' + end_date
    # In[]
    panel_data = []
    trend_data = []
    for i in tickers:
        panel_data.append(Minmax.fit_transform(web.DataReader(i, 'iex', start_date, end_date)).T)
    for i in tickers:
        pytrends.build_payload([i], cat=0, timeframe=total_dates, geo='', gprop='')
        trend_data.append(Minmax.fit_transform(pytrends.interest_over_time()).T)
        print('Collecting Google Trends Data')
    class Data():
        tick_list = tickers
        start = start_date
        end = end_date
        stock_data = panel_data
        google_data = trend_data
    return(Data.stock_data,Data.google_data)
# Fix in Future
# In[]
