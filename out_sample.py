# -*- coding: utf-8 -*-
"""
VIX-RSI Strategy: Out-of-sample Back Test  (2014-2018)

@author: Godfrey Ojo
"""
# Import relevant libraries and Modules
import numpy as np
import pandas as pd
from pandas_datareader import data as web
import datetime
start = datetime.datetime(2014,1,1)
end = datetime.date.today()
# set the number of trading days
no_days = 252
# Dowmload dataframe from yahoo finance by prompting user to enter name of valid ticker 
stock = web.DataReader(input("Please Input the name of the Ticker:\n"), "yahoo", start, end)
# Upload file containing values of VIX-RSI from directory
VIX= pd.read_csv('C:\\Users\\User\\Documents\\vix-rsi_2.csv', parse_dates=True, index_col='Date')
# Define a function to return the RSI using the VIX daily adjusted close price   
def RSI_VIX(vix, n):
    """ This function calculates and return the RSI of the VIX Index"""
    close = stock['Adj Close'] # Get just the close
    delta = close.diff() # Get the difference in price from previous step
    # Get rid of the first row, which is NaN since it did not have a previous 
    # row to calculate the differences
    delta = delta[1:] 
    # Make the positive gains (up) and negative gains (down) Series
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    # Calculate the EWMA
    roll_up1 = up.ewm(com=n,min_periods=0,adjust=True,ignore_na=False).mean()
    roll_down1 = abs(down.ewm(com=n,min_periods=0,adjust=True,ignore_na=False).mean())
    # Calculate the RSI based on EWMA
    RS = roll_up1 / roll_down1
    RSI  = 100.0 - (100.0 / (1.0 + RS))
    return RSI
#rsi_vix = (RSI_VIX(stock,20))
#rsi_vix.to_csv('RSI_20.csv')
def RSI_PRICE(ticker,n):
    """ This function calculates and returns the RSI of the chosen ticker"""
    close = ticker['Adj Close']
    # Get just the close
    # Get the difference in price from previous step
    delta = close.diff()
    # Get rid of the first row, which is NaN since it did not have a previous 
    # row to calculate the differences
    delta = delta[1:] 
    # Make the positive gains (up) and negative gains (down) Series
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    # Calculate the EWMA
    roll_up1 = up.ewm(com=n,min_periods=0,adjust=True,ignore_na=False).mean()
    roll_down1 = abs(down.ewm(com=n,min_periods=0,adjust=True,ignore_na=False).mean())
    # Calculate the RSI based on EWMA
    RS = roll_up1 / roll_down1
    RSI  = 100.0 - (100.0 / (1.0 + RS))
    return RSI
#rsi_price = (RSI_PRICE(stock,5))
# Implement the VIX-RSI strategy by prompting the user to input the upper and the lower RSI of the VIX
Y = int(input('Please input upper RSI:\n')) 
X = int(input('Please input lower RSI:\n')) 
# Initialize the VIX-RSI Trading Signals
VIX['Stance'] = np.where(VIX['RSI_2'] > Y, -1, 0)
VIX['Stance'] = np.where(VIX['RSI_2'] < X, 1, VIX['Stance'])
# Bucket generated signals into a dataframe
stock['Signal'] = pd.DataFrame(VIX['Stance'])
# Calculate market return (Buy-and-Hold Strategy)
VIX['Market Returns'] = np.log(stock['Close'] / stock['Close'].shift(1))
# Calculate Strategy's return
VIX['Strategy'] = VIX['Market Returns'] * VIX['Stance'].shift(1)
# Plot Market Vs Strategy's Return
VIX[['Market Returns','Strategy']].cumsum().plot(title='Out-of-sample VIX-RSI_2 Performance on JNK',grid=True,figsize=(15,4))
# Print out the number of trade signals
print(VIX['Stance'].value_counts())
# Calculate Performance Metrics
# Initialize number of trading days per year
market_ret= pd.DataFrame(VIX['Market Returns'])
strategy_ret = pd.DataFrame(VIX['Strategy'])
annual_mkt = (np.mean(market_ret) *(no_days))
mkt_std = (np.std(market_ret) *np.sqrt(no_days))
mkt_sharpe = annual_mkt / mkt_std
annual_str = (np.mean(strategy_ret) *(no_days))
str_std = (np.std(strategy_ret) *np.sqrt(no_days))
str_sharpe = annual_str / str_std
# Print the Results
print('Annual Market Return', annual_mkt)
print('Annual Market Volatility', mkt_std )
print('Market Sharpe Ratio', mkt_sharpe)
print('Annual Strategy Return', annual_str)
print('Annual Market Volatility', str_std )
print('Market Sharpe Ratio', str_sharpe)
