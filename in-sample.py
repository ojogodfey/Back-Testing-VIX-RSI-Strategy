# -*- coding: utf-8 -*-
"""
VIX-RSI Strategy: In-sample Back Test  (1993-2013)

@author: Godfrey Ojo
"""
# Import relevant libraries and modules
import numpy as np
import pandas as pd
from pandas_datareader import data as web
import datetime
start = datetime.datetime(1993,1,1)
end = datetime.datetime(2013,12,31)
# Initialize number of trading days per year
no_days = int(252)
# Dowmload dataframe from yahoo finance by prompting user to enter name of valid ticker 
stock = web.DataReader(input("Please Input the name of the Ticker:\n"), "yahoo", start, end)
# Load Calculated VIX-RSI Values from directory
VIX= pd.read_csv('C:\\Users\\User\\Desktop\\vix-rsi.csv', parse_dates=True, index_col='Date')
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
VIX['Stance'] = np.where(VIX['RSI_2'] > Y, 1, 0)
VIX['Stance'] = np.where(VIX['RSI_2'] < X, -1, VIX['Stance'])
# Bucket generated signals into a dataframe
stock['Signal'] = pd.DataFrame(VIX['Stance']) 
# Calculate market return (Buy-and-Hold Strategy)
VIX['Market Returns'] = np.log(stock['Close'] / stock['Close'].shift(1))
# Calculate Strategy's return
VIX['Strategy'] = VIX['Market Returns'] * VIX['Stance'].shift(1)
# Plot Market Vs Strategy's Return
VIX[['Market Returns','Strategy']].cumsum().plot(grid=True,figsize=(8,5))
# Print out the number of trade signals
print(VIX['Stance'].value_counts())
# Calculate Performance Metrics
market_ret= pd.DataFrame(VIX['Market Returns'])
strategy_ret = pd.DataFrame(VIX['Strategy'])
annual_mkt = (np.mean(market_ret) *(no_days))
mkt_std = (np.std(market_ret) *np.sqrt(no_days))
mkt_sharpe = (annual_mkt / mkt_std)
annual_str = (np.mean(strategy_ret) *(no_days))
str_std = (np.std(strategy_ret) *np.sqrt(no_days))
str_sharpe = (annual_str / str_std)
# Print the Results
print('Annual Market Return is', annual_mkt)
print('Annual Market Volatility is', mkt_std )
print('Market Sharpe Ratio is', mkt_sharpe)
print('Annual Strategy Return is', annual_str)
print('Annual Strategy Volatility is', str_std )
print('Strategy Sharpe Ratio is', str_sharpe)

