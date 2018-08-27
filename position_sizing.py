# -*- coding: utf-8 -*-
"""
VIX-RSI Strategy Performance Tuning using Position Sizing Models (Out-of Sample (2014-2018))

@author: Godfrey Ojo
"""
# Import relevant libraries and modules
import numpy as np
import pandas as pd
from pandas_datareader import data as web
import datetime
start = datetime.datetime(2014,1,1)
end = datetime.date.today()
# Dowmload dataframe from yahoo finance by prompting user to enter name of valid ticker 
stock = web.DataReader(input("Please Input the name of the Ticker:\n"), "yahoo", start, end)
# Upload out-of-sample VIX-RSI data frame from directory
VIX= pd.read_csv('C:\\Users\\User\\Documents\\vix-rsi_2.csv', parse_dates=True, index_col='Date')
# Initialize Initial and Risk Capital
initial_capital = float(100000.0) 
risk_capital = 0.02
# Implement the VIX-RSI strategy by prompting the user to input the upper and the lower RSI of the VIX
Y = int(input('Please input upper RSI:\n')) 
X = int(input('Please input lower RSI:\n'))
# Initial the trading signals 
VIX['Stance'] = np.where(VIX['RSI_2'] > Y, -1, 0)
VIX['Stance'] = np.where(VIX['RSI_2'] < X, 1, VIX['Stance'])
stock['Signal'] = pd.DataFrame(VIX['Stance'])
# Calculate market return
VIX['Market Returns'] = np.log(stock['Close'] / stock['Close'].shift(1))
# calculate strategy's return
VIX['Strategy'] = VIX['Market Returns'] * VIX['Stance'].shift(1)
# plot combine performnace plot of market vs strategy
VIX[['Market Returns','Strategy']].cumsum().plot(grid=True,figsize=(8,5))
# print the number of signals
print(VIX['Stance'].value_counts())
# Define a function that calculates and return postion size to trade using ATR of 5 days
def ATR(df,n):
    "This function calculates and returns the average true range"
    # Current High less the current Low
    df["TR1"] = df["High"] - df["Low"]
    # Current High less the previous Close (absolute value)
    df["TR2"] = abs(df["High"] - df["Close"].shift(1))
    # Current Low less the previous Close (absolute value)
    df["TR3"] = abs(df["Low"] - df["Close"].shift(1))
    df["TR"] = df[["TR1", "TR2", "TR3"]].max(axis=1)
    df["ATR"] = df.TR.ewm(span=n,min_periods=n,adjust=False).mean()
    df = df.dropna()
    df['ATR_5'] = df['ATR'] *5
    df['Size'] = (initial_capital * risk_capital) / df['ATR_5']
    return df
atr = ATR(stock,2)
# Define a function that implements the strategy and return updated portfolio using volatility sizing model
def volatility_sizing():
    """ This function implements the strategy and returns the updated portfolio"""
    # Create a DataFrame `positions`
    positions = pd.DataFrame(index=stock.index).fillna(0.0)
    # Buy APPLE shares worth 10% of Initial Capital of  per trade
    positions['stock'] = atr.Size * stock['Signal']
    # Initialize the portfolio with value owned   
    portfolio = positions.multiply(stock['Adj Close'], axis=0)
    # Store the difference in shares owned 
    pos_diff = positions.diff()
    # Add `holdings` to portfolio
    portfolio['holdings'] = (positions.multiply(stock['Close'], axis=0)).sum(axis=1)
    # Add `cash` to portfolio
    portfolio['cash'] = initial_capital - (pos_diff.multiply(stock['Close'], axis=0)).sum(axis=1).cumsum()   
    # Add `total` to portfolio
    portfolio['total'] = portfolio['cash'] + portfolio['holdings']
    # Add `returns` to portfolio
    portfolio['returns'] = portfolio['total'].pct_change()
    return portfolio
portfolio=volatility_sizing()
# define a function to implements the strategy and returns the updated portfolio using fixed percentage model
def fixed_size():
    """ This function implements the strategy and returns the updated portfolio"""
    # Create a DataFrame `positions`
    fixed_size = 0.1 * initial_capital
    positions1 = pd.DataFrame(index=stock.index).fillna(0.0)
    # Buy APPLE shares worth 10% of Initial Capital of  per trade
    positions1['stock'] = fixed_size * stock['Signal']
    # Initialize the portfolio with value owned   
    portfolio1 = positions1.multiply(stock['Adj Close'], axis=0)
    # Store the difference in shares owned 
    pos_diff1 = positions1.diff()
    # Add `holdings` to portfolio
    portfolio1['holdings'] = (positions1.multiply(stock['Close'], axis=0)).sum(axis=1)
    # Add `cash` to portfolio
    portfolio1['cash'] = initial_capital - (pos_diff1.multiply(stock['Close'], axis=0)).sum(axis=1).cumsum()   
    # Add `total` to portfolio
    portfolio1['total'] = portfolio1['cash'] + portfolio1['holdings']
    # Add `returns` to portfolio
    portfolio1['returns'] = portfolio1['total'].pct_change()
    return portfolio1
portfolio1=fixed_size()
# define a functions calculates and returns the value of the optimal F
def optimal():
    """ This functions calculates and returns the value of the optimal F"""
    wins = (portfolio1['returns'] > 0).sum() #Calculate the sum of absolute positive daily returns
    loss = abs(portfolio1['returns'] < 0).sum()#Calculate the sum of absolute negative daily returns
    prob_win= wins/(wins + loss) # calculate probaility of wins
    win_loss_ratio = wins/loss # calculate win loss ratio
    optimal_f = prob_win*(win_loss_ratio +1.0) -1 /win_loss_ratio # calculate the optimal F
    return optimal_f
optimal_p= optimal()
# Define a functions to backtests and returns the updated portfolio using the optimal F
def optimized_f():
    """ This functions backtests and returns the updated portfolio using the optimal F"""
    initial_capital= float(100000.0)
    trade_amount = optimal_p *initial_capital
    # Create a DataFrame `positions`
    positions2 = pd.DataFrame(index=stock.index).fillna(0.0)
    # Buy trade amount of Apple stocks per trade
    positions2['stock'] = trade_amount* stock['Signal']   
    # Initialize the portfolio with value owned   
    portfolio2 = positions2.multiply(stock['Adj Close'], axis=0)
    # Store the difference in shares owned 
    pos_diff = positions2.diff()
    # Add `holdings` to portfolio
    portfolio2['holdings'] = (positions2.multiply(stock['Adj Close'], axis=0)).sum(axis=1)
    # Add `cash` to portfolio
    portfolio2['cash'] = initial_capital - (pos_diff.multiply(stock['Adj Close'], axis=0)).sum(axis=1).cumsum()   
    # Add `total` to portfolio
    portfolio2['total'] = portfolio2['cash'] + portfolio2['holdings']
    # Add `returns` to portfolio
    portfolio2['returns'] = portfolio2['total'].pct_change()
    return portfolio2
portfolio2=optimized_f()
# Calculate risk performance metrics for the volatility size model 
#Initial key performance inputs
df = pd.Series(portfolio['total'])
start_value = df[0]
end_value = df[-1]
n_period = len(df)
# Generate positive and negative returns
portfolio['Wins'] = np.where(portfolio['returns'] > 0, 1, 0) # Positive Trades
portfolio['Losses'] = np.where(portfolio['returns'] < 0, 1,0) # Negative trades
positive_trades= portfolio['Wins'].sum()
negative_trades= portfolio['Losses'].sum()    
# Calculate the win % and win to loss ratio
total_trades = positive_trades + negative_trades # Calculate the total number of trades for the period
win_ratio = positive_trades/total_trades # Calculate the win rate per trade
hit_rate = positive_trades / negative_trades # calculate the win to loss ratio
# calculate mean return per trade
mean_return_per_trade = np.sum(portfolio['returns']) / total_trades
# Define a function calculate the gain to pain
def gain_pain():
    """ This function calculates and returns the Gain to Pain Ratio for the Portfolio"""
    sum_returns = portfolio['returns'].sum()# Calculate the sum of returns
    sum_neg_returns = abs(portfolio['returns'] < 0).sum()#Calculate the sum of absolute negative monthly returns
    gain_to_pain = sum_returns / sum_neg_returns #Calculate the Gain to Pain ratio
    return gain_to_pain
g_p = gain_pain()
def max_drawdown(X):
    """ This function calculates and returns the daily drawndown for the portfolio"""
    mdd = 0
    peak = X[0]
    for x in X:
        if x > peak: 
            peak = x
        dd = (peak - x) / peak
        if dd > mdd:
            mdd = dd
    return mdd    
drawSeries = max_drawdown(df)
MaxDD = abs(drawSeries.min()*100)
def CAGR(start, end, period):
    """ This Function Calculates and Returns the Cumulative Annual Growth Rate for the Portfolio"""
    return (end/start) **(252/period) -1
cagr = (CAGR(start_value, end_value, n_period))* 100
sterling_ratio = cagr /MaxDD
#Tabulate and print Portfolio KPIs using pandas dataframe and python dictionary
portfolio_KPIs =pd.DataFrame( {'Initial Capital':start_value, 'Portfolio End Value':end_value, 'No of Positive Trades':positive_trades,
                        'No of Negative Trades':negative_trades, 'Total No of Trades':total_trades,
                        'Win Rate':win_ratio, 'Hit Rate':hit_rate, 'Mean Return Per Trade':mean_return_per_trade,
                        'Gain to Pain':g_p, 'Maximum Drawdown': MaxDD, 'CAGR':cagr, 'Sterling Ratio': sterling_ratio }, index=[0])
#print('portfolio',portfolio_KPIs.T)
#portfolio_KPIs.T.to_csv('C:\\Users\\User\\Documents\\SLV.csv')
# Calculate risk performance metrics for the fixed percentage model 
#Initial key performance inputs
df1 = pd.Series(portfolio1['total'])
start_value1 = df1[0]
end_value1 = df1[-1]
n_period1 = len(df1)
# Generate positive and negative returns
portfolio1['Wins'] = np.where(portfolio1['returns'] > 0, 1, 0) # Positive Trades
portfolio1['Losses'] = np.where(portfolio1['returns'] < 0, 1,0) # Negative trades
positive_trades1= portfolio1['Wins'].sum()
negative_trades1= portfolio1['Losses'].sum()    
# Calculate the win % and win to loss ratio
total_trades1 = positive_trades1 + negative_trades1 # Calculate the total number of trades for the period
win_ratio1 = positive_trades1/total_trades1 # Calculate the win rate per trade
hit_rate1 = positive_trades1 / negative_trades1 # calculate the win to loss ratio
# calculate mean return per trade
mean_return_per_trade1 = np.sum(portfolio1['returns']) / total_trades1
# Define a function calculate the gain to pain
def gain_pain1():
    """ This function calculates and returns the Gain to Pain Ratio for the Portfolio"""
    sum_returns1 = portfolio1['returns'].sum()# Calculate the sum of returns
    sum_neg_returns1 = abs(portfolio1['returns'] < 0).sum()#Calculate the sum of absolute negative monthly returns
    gain_to_pain1 = sum_returns1 / sum_neg_returns1 #Calculate the Gain to Pain ratio
    return gain_to_pain1
g_p1 = gain_pain1()
def max_drawdown1(X):
    """ This function calculates and returns the daily drawndown for the portfolio"""
    mdd1 = 0
    peak1 = X[0]
    for x in X:
        if x > peak1: 
            peak1 = x
        dd1 = (peak1 - x) / peak1
        if dd1 > mdd1:
            mdd1 = dd1
    return mdd1    
drawSeries1 = max_drawdown1(df1)
MaxDD1 = abs(drawSeries1.min()*100)
def CAGR1(start1, end1, period1):
    """ This Function Calculates and Returns the Cumulative Annual Growth Rate for the Portfolio"""
    return (end1/start1) **(252/period1) -1
cagr1 = (CAGR1(start_value1, end_value1, n_period1))* 100
sterling_ratio1 = cagr1 /MaxDD1
#Tabulate and print Portfolio KPIs using pandas dataframe and python dictionary
portfolio1_KPIs =pd.DataFrame( {'Initial Capital':start_value1, 'Portfolio End Value':end_value1, 'No of Positive Trades':positive_trades1,
                        'No of Negative Trades':negative_trades1, 'Total No of Trades':total_trades1,
                        'Win Rate':win_ratio1, 'Hit Rate':hit_rate1, 'Mean Return Per Trade':mean_return_per_trade1,
                        'Gain to Pain':g_p1, 'Maximum Drawdown': MaxDD1, 'CAGR':cagr1, 'Sterling Ratio': sterling_ratio1 }, index=[0])
#print('portfolio1',portfolio1_KPIs.T)
#portfolio1_KPIs.T.to_csv('C:\\Users\\User\\Documents\\SLV1.csv')
# Calculate risk performance metrics for the Optimal F model 
#Initial key performance inputs
df2 = pd.Series(portfolio2['total'])
start_value2 = df2[0]
end_value2 = df2[-1]
n_period2 = len(df2)
# Generate positive and negative returns
portfolio2['Wins'] = np.where(portfolio2['returns'] > 0, 1, 0) # Positive Trades
portfolio2['Losses'] = np.where(portfolio2['returns'] < 0, 1,0) # Negative trades
positive_trades2= portfolio2['Wins'].sum()
negative_trades2= portfolio2['Losses'].sum()    
# Calculate the win % and win to loss ratio
total_trades2 = positive_trades2 + negative_trades2 # Calculate the total number of trades for the period
win_ratio2 = positive_trades2/total_trades2 # Calculate the win rate per trade
hit_rate2 = positive_trades2 / negative_trades2 # calculate the win to loss ratio
# calculate mean return per trade
mean_return_per_trade2 = np.sum(portfolio2['returns']) / total_trades2
# Define a function calculate the gain to pain
def gain_pain2():
    """ This function calculates and returns the Gain to Pain Ratio for the Portfolio"""
    sum_returns2 = portfolio2['returns'].sum()# Calculate the sum of returns
    sum_neg_returns2 = abs(portfolio2['returns'] < 0).sum()#Calculate the sum of absolute negative monthly returns
    gain_to_pain2 = sum_returns2 / sum_neg_returns2 #Calculate the Gain to Pain ratio
    return gain_to_pain2
g_p2 = gain_pain2()
def max_drawdown2(X):
    """ This function calculates and returns the daily drawndown for the portfolio"""
    mdd2 = 0
    peak2 = X[0]
    for x in X:
        if x > peak2: 
            peak2 = x
        dd2 = (peak2 - x) / peak2
        if dd2 > mdd2:
            mdd2 = dd2
    return mdd2    
drawSeries2 = max_drawdown2(df2)
MaxDD2 = abs(drawSeries2.min()*100)
def CAGR2(start2, end2, period2):
    """ This Function Calculates and Returns the Cumulative Annual Growth Rate for the Portfolio"""
    return (end2/start2) **(252/period2) -1
cagr2 = (CAGR2(start_value2, end_value2, n_period2))* 100
sterling_ratio2 = cagr2 /MaxDD2
#Tabulate and print Portfolio KPIs using pandas dataframe and python dictionary
portfolio2_KPIs =pd.DataFrame( {'Initial Capital':start_value2, 'Portfolio End Value':end_value2, 'No of Positive Trades':positive_trades2,
                        'No of Negative Trades':negative_trades2, 'Total No of Trades':total_trades2,
                        'Win Rate':win_ratio2, 'Hit Rate':hit_rate2, 'Mean Return Per Trade':mean_return_per_trade2,
                        'Gain to Pain':g_p2, 'Maximum Drawdown': MaxDD2, 'CAGR':cagr2, 'Sterling Ratio': sterling_ratio2 }, index=[0])
#print('portfolio2',portfolio2_KPIs.T)
#portfolio2_KPIs.T.to_csv('C:\\Users\\User\\Documents\\SLV2.csv')
# concatenate the 3 position sizing models into 1 dataframe
equity_curves = pd.DataFrame({'Volatility Size':portfolio.total,'Fixed Size':portfolio1.total,'Optimal F':portfolio2.total},index=stock.index)
#Plot Portfoilo performance for the selected ticker
equity_curves.plot(title='Strategy Performance on JNK under different Position Sizing Models',grid=True,figsize=(15,4))




