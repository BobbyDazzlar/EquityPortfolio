import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('n50.csv', parse_dates=['Date'], index_col='Date')
start_date="2016-01-29"
end_date="2021-01-28"
df = df.loc[start_date: end_date]  # Since 2016-01-01, 5y(1238rows till 2020-12-31), + year 2021's rows
tdf = df.copy()  # deep copy
df.reset_index(drop=True, inplace=True)


def number_of_years(y):  # calculates the number of years of the dataset
    p = y.index[0]  # date of first row in the dataset (datetime format)
    q = y.index[len(y) - 1]  # date of last row in the dataset  (datetime format)
    return ((q - p).days + 1) / 365  # the difference give the number of total days (not trading days) over the total number of years in the dataset


trading_days = len(df) / number_of_years(tdf)  # Trading days per year (automated)

returnsh = df.pct_change()  # Here, returnsh would mean return considered for sharpe ratio
returnsh.fillna(0, inplace=True)  # calculating daily returns of the stocks in the portfolio

returnso = returnsh.copy()  # this cell considers only NEGATIVE returns so as to calculate sortino ratio
for cols in returnso.columns.tolist():
    for i in range(0, len(df)):
        if returnso[cols][i] > 0:
            returnso[cols][i] = 0  # Here, returnso would mean return considered for sortino ratio

covmatsh = returnsh.cov() * trading_days  # Annualised covariance matrix calculated wrt returnsh i.e. used to calculate sharpe ratio
covmatso = returnso.cov() * trading_days  # Annualised covariance matrix calculated wrt returnso i.e. used to calculate sortino ratio

num_portfolios = 50000  # initializing number of portfolios to 50000; referred from Wang et al (2020) (science direct)
num_assets = len(df.columns)  # initializing number of stocks/assets considered in the portfolio
risk_free_rate = 0.0358  # initializing risk free rate that will be used in calculating both the ratios (absolute value)
# referred from url: https://www.rbi.org.in/Scripts/BS_NSDPDisplay.aspx?param=4&Id=24292
# In the above url, the 364 (1 year) day treasury bill is 3.58% , when taken absolute value => 0.0358
# (improved)

# 2021_chen etal_Mean–variance portfolio optimization using machine learning-based stock price prediction
# Repeat the process 50,000times. From a statistical point of view, 50,000 random portfolios cover most possible portfolios with different weights and aresufficiently representative

portfolio_returns = []  # initializing an empty list for portfolio returns
portfolio_volatility = []  # initializing an empty list for portfolio risk
stock_weights = []  # initializing an empty list for portfolio weights
semi_deviation = []  # initializing an empty list for portfolio semi-deviation
sharpe = []  # initializing an empty list for portfolio sharpe ratio
sortino = []  # initializing an empty list for portfolio sortino ratio


def ratio(a, b, c):  # function to calculate ratio i.e. "(returns-(risk_free_rate))/deviation"
    return (
                       a - c) / b  # a => annual return, c => risk_free_rate, b => deviation (standard for sharpe, semi for sortino)


for single_portfolio in range(num_portfolios):  # iterating forloop for 50000 times to generate 50000 portfolios
    weights = np.random.random(num_assets)  # initializing random weights
    weights /= np.sum(
        weights)  # No Short Selling Allowed => weights add up to 1   "x = x+y" => "x+=y"    weights = weights/np.sum(weights)
    returns_temp = np.sum(returnsh.mean() * weights) * trading_days  # calculating annulaised portfolio return
    varsh = np.dot(weights.T, np.dot(covmatsh, weights))  # calculating portfolio varience wrt calculating sharpe ratio
    varso = np.dot(weights.T, np.dot(covmatso, weights))  # calculating portfolio varience wrt calculating sortino ratio
    volatility_temp = np.sqrt(varsh)  # portfolio risk
    semi_temp = np.sqrt(varso)  # portfolio semi-deviation
    shtemp = ratio(returns_temp, volatility_temp, risk_free_rate)  # calculating sharpe ratio
    sotemp = ratio(returns_temp, semi_temp, risk_free_rate)  # calculating sortino ratio
    portfolio_returns.append(returns_temp)
    portfolio_volatility.append(volatility_temp)
    stock_weights.append(weights)
    sharpe.append(shtemp)
    sortino.append(sotemp)
    semi_deviation.append(semi_temp)

portfolio = {'Returns': portfolio_returns, 'Standard Deviation': portfolio_volatility, 'Semi-Deviation': semi_deviation,
             'Sharpe Ratio': sharpe,
             'Sortino Ratio': sortino}
# here, 'portfolio' is a dictionary which will be used to create dataframe where each row will be a portfolio

for counter, symbol in enumerate(df.columns):
    portfolio[symbol + " Weight"] = [Weight[counter] for Weight in stock_weights]
# to the dictionary (named 'portfolio'), weights for each symbol are added in so as to be displayed in the dataframe

pc = pd.DataFrame(
    portfolio)  # making the final dataframe where data of 50000 portfolios is appended (subject to be saved, whose code is to be written)

pc = pc * 100  # Converting everything to percentage
pc['Sharpe Ratio'] = pc['Sharpe Ratio'] / 100  # leaving ratios as it is
pc['Sortino Ratio'] = pc['Sortino Ratio'] / 100

#pc.to_csv('portfolios_by_MV.csv')  # saving the portfolios data

max_sharpe = pc['Sharpe Ratio'].max()  # Best optimised portfolio wrt sharpe ratio
max_sharpe_portfolio = pc.loc[pc['Sharpe Ratio'] == max_sharpe]
max_sortino = pc['Sortino Ratio'].max()  # Best optimised portfolio wrt sortino ratio
max_sortino_portfolio = pc.loc[pc['Sortino Ratio'] == max_sortino]

pc_sharpe = pc.drop(columns=['Sortino Ratio', 'Semi-Deviation'])

pc_sharpe_top10 = pc_sharpe.sort_values(by=['Sharpe Ratio'], ascending=False).head(10)

#pc_sharpe_top10.to_csv('Sharpe_Top10_MV.csv')

pc_sharpe_bottom10 = pc_sharpe.sort_values(by=['Sharpe Ratio'], ascending=False).tail(10)

#pc_sharpe_bottom10.to_csv('Sharpe_Bottom10_MV.csv')

sharpe_optimal_portfolio = pc_sharpe_top10.head(1)

sharpe_optimal_portfolio.to_csv('MVO_Sharpe_Optimal.csv')

pc_sortino = pc.drop(columns=['Sharpe Ratio', 'Standard Deviation'])

pc_sortino_top10 = pc_sortino.sort_values(by=['Sortino Ratio'], ascending=False).head(10)

#pc_sortino_top10.to_csv('Sortino_Top10_MV.csv')

pc_sortino_bottom10 = pc_sortino.sort_values(by=['Sortino Ratio'], ascending=False).tail(10)

#pc_sortino_bottom10.to_csv('Sortino_Bottom10_MV.csv')

sortino_optimal_portfolio = pc_sortino_top10.head(1)

sortino_optimal_portfolio.to_csv('MVO_Sortino_Optimal.csv')
