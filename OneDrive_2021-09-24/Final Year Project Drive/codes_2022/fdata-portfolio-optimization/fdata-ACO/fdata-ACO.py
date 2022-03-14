import optuna

import pandas as pd
import numpy as np
import random
import math

def BALANCE(weights):
  #Making sure the total sum of the weights eual to 1
  weights = [w/sum(weights) for w in weights]
  # Making sure all weights represent proportions that add up to 1
  return weights

def ratio(a,b,c):
  #function to calculate ratio i.e. "(returns-(risk_free_rate))/deviation"
  #calculating sharpe ratio
  return (a-c)/b

def number_of_years(y):#calculates the number of years of the dataset
  p=y.index[0]         #date of first row in the dataset (datetime format)
  q=y.index[len(y)-1]  #date of last row in the dataset  (datetime format)
  return ((q-p).days+1)/365

df=pd.read_csv("fdata08_02_22.csv",parse_dates=['Date'],index_col='Date')  #Importing Dataset
df = df.loc["2016-01-01" : ]   #Since 2016-01-01, 5y(1234rows till 2020-12-31)
tdf=df.copy()                  #deep copy
df.reset_index(drop=True, inplace=True)
col=list(df.columns)

trading_days=len(df)/number_of_years(tdf)

returnsh=df.pct_change()
#Here, returnsh would mean return considered for sharpe ratio
returnsh.fillna(0,inplace=True)

returnso = returnsh.copy()  # this cell considers only NEGATIVE returns so as to calculate sortino ratio
for cols in returnso.columns.tolist():
  for i in range(0, len(df)):
    if returnso[cols][i] > 0:
      returnso[cols][i] = 0

covmatsh=returnsh.cov()*trading_days
#Annualised covariance matrix calculated wrt returnsh i.e. used to calculate sharpe ratio
covmatso = returnso.cov() * trading_days

risk_free_rate = 0.0358 #initializing risk free rate that will be used in calculating both the ratios (absolute value)
#referred from url: https://www.rbi.org.in/Scripts/BS_NSDPDisplay.aspx?param=4&Id=24292
#In the above url, the 364 (1 year) day treasury bill is 3.58% , when taken absolute value => 0.0358
# (improved)

stocks=df.shape[1]

# Sharpe

def antcolony_tuning_sharpe(ITERATIONS, Q, EVA_RATE, ANTS):
    sharpe_pbest = -1
    # Initializing sharpe_pbest(the best fitness value   SHARPE)
    # Initializing the current fitness value
    fitness = 0
    # for each iteration
    for iteration in range(ITERATIONS):

        # PREPARAING THE PHEROMONE MATRIX WHERE THE COLS=STOCKS AND  ROWS=ANTS
        pheromon = [[0] * stocks for i in
                    range(ANTS + 1)]  # why (ants+1)?The last ant can update the pheromone values in the last row

        # Initializing the pheromone status
        for i in range(len(pheromon[0])):
            pheromon[0][i] = random.randint(1,
                                            15)  # When input stocks varies, this needs to vary accordingly.(Divide number of stocks / 2)

        # copying the values and storing it in temp_pher
        temp_pher = pheromon[0]

        # Making sure that the total amount of pheromone equals 1
        weights = np.array(BALANCE(temp_pher))

        # calculating annulaised portfolio return
        returns_temp = np.sum(returnsh.mean() * weights) * trading_days

        # calculating portfolio varience wrt calculating sharpe ratio
        varsh = np.dot(weights.T, np.dot(covmatsh, weights))

        # portfolio risk
        volatility_temp = np.sqrt(varsh)

        # Calculating fitness value(ie sharpe ratio)
        fitness = ratio(returns_temp, volatility_temp, risk_free_rate)

        # Initializing the intial fitness value as the best fitness value(sharpe_pbest)
        if sharpe_pbest == -1:
            sharpe_pbest = fitness

        # list
        path = []

        # for each ant
        for ant in range(ANTS - 1):

            # find the total pheromone
            total = sum(pheromon[ant])

            # Initializing probability
            probability = pheromon[ant][:]

            # finding probability of each stocks pheromone
            for p in range(len(probability)):
                probability[p] = (probability[p] / total)

            # Trying to select stocks in decreasing order based on their pheromone level and storing the stock order in a list(path)
            for stock in range(stocks):
                select = probability.index(max(probability))
                probability[select] = -math.inf
                path.append(select)

            # Updating the pheromone level of each stock for the next ant
            # Formula: old pheromone level * (1-eva_rate) + Q * (fitness/sharpe_pbest) where Q is fixed amount of pheromone
            for s in path:
                pheromon[ant + 1][s] = pheromon[ant][s] * (1 - EVA_RATE) + Q * (fitness / sharpe_pbest)

            # making sure that the updated pheromon adds upto 1
            temp_pher = pheromon[ant + 1]
            weights = np.array(BALANCE(temp_pher))
            returns_temp = np.sum(returnsh.mean() * weights) * trading_days  # calculating annulaised portfolio return
            varsh = np.dot(weights.T,
                           np.dot(covmatsh, weights))  # calculating portfolio varience wrt calculating sharpe ratio
            volatility_temp = np.sqrt(varsh)  # portfolio risk
            fitness = ratio(returns_temp, volatility_temp, risk_free_rate)  # calculating sharpe ratio

            # comparing the old fitness value with the updated fitness value
            # explore on scape condition paper ref
            if (fitness > sharpe_pbest):
                # if the updated fitness value is better than the previous, change sharpe_pbest to present fitness value
                sharpe_pbest = fitness

                # remembering the weights of the best portfolio
                global_warr_sharpe = weights.tolist()
        # sharpe_portfolio_return.append(returns_temp)
        # sharpe_portfolio_risk.append(volatility_temp)
        # sharpe_portfolio_shratio.append(fitness)
        # sharpe_portfolio_stockWeights.append(weights)
    return sharpe_pbest

#hyperparameter values from literature survey excel sheet
def objective(trial):
    ITERATIONS=trial.suggest_int('ITERATIONS',2,500)
    Q=trial.suggest_float('Q',0.1,1.0)
    EVA_RATE=trial.suggest_float('EVA_RATE',0.1,1.00)
    ANTS=trial.suggest_int('ANTS',2,500)
    return antcolony_tuning_sharpe(int(ITERATIONS),Q,EVA_RATE,int(ANTS))

sharpe_study=optuna.create_study(direction='maximize')
sharpe_study.optimize(objective,n_trials=100)

sh_hptuning=sharpe_study.trials_dataframe()
best=sharpe_study.best_params

ITERATIONS=int(best['ITERATIONS'])
Q=best['Q']
EVA_RATE=best['EVA_RATE']
ANTS=best['ANTS']

global_warr_sortino=[]
global_war_sharpe=[]
sharpe_portfolio_return=[]
sharpe_portfolio_risk=[]
sharpe_portfolio_shratio=[]
sharpe_portfolio_stockWeights=[]


def antcolony_sharpe(ITERATIONS, Q, EVA_RATE, ANTS):
    sharpe_pbest = -1
    # Initializing sharpe_pbest(the best fitness value   SHARPE)
    # Initializing the current fitness value
    fitness = 0
    # for each iteration
    for iteration in range(ITERATIONS):

        # PREPARAING THE PHEROMONE MATRIX WHERE THE COLS=STOCKS AND  ROWS=ANTS
        pheromon = [[0] * stocks for i in
                    range(ANTS + 1)]  # why (ants+1)?The last ant can update the pheromone values in the last row

        # Initializing the pheromone status
        for i in range(len(pheromon[0])):
            pheromon[0][i] = random.randint(1,
                                            15)  # When input stocks varies, this needs to vary accordingly.(Divide number of stocks / 2)

        # copying the values and storing it in temp_pher
        temp_pher = pheromon[0]

        # Making sure that the total amount of pheromone equals 1
        weights = np.array(BALANCE(temp_pher))

        # calculating annulaised portfolio return
        returns_temp = np.sum(returnsh.mean() * weights) * trading_days

        # calculating portfolio varience wrt calculating sharpe ratio
        varsh = np.dot(weights.T, np.dot(covmatsh, weights))

        # portfolio risk
        volatility_temp = np.sqrt(varsh)

        # Calculating fitness value(ie sharpe ratio)
        fitness = ratio(returns_temp, volatility_temp, risk_free_rate)

        # Initializing the intial fitness value as the best fitness value(sharpe_pbest)
        if sharpe_pbest == -1:
            sharpe_pbest = fitness

        # list
        path = []

        # for each ant
        for ant in range(ANTS - 1):

            # find the total pheromone
            total = sum(pheromon[ant])

            # Initializing probability
            probability = pheromon[ant][:]

            # finding probability of each stocks pheromone
            for p in range(len(probability)):
                probability[p] = (probability[p] / total)

            # Trying to select stocks in decreasing order based on their pheromone level and storing the stock order in a list(path)
            for stock in range(stocks):
                select = probability.index(max(probability))
                probability[select] = -math.inf
                path.append(select)

            # Updating the pheromone level of each stock for the next ant
            # Formula: old pheromone level * (1-eva_rate) + Q * (fitness/sharpe_pbest) where Q is fixed amount of pheromone
            for s in path:
                pheromon[ant + 1][s] = pheromon[ant][s] * (1 - EVA_RATE) + Q * (fitness / sharpe_pbest)

            # making sure that the updated pheromon adds upto 1
            temp_pher = pheromon[ant + 1]
            weights = np.array(BALANCE(temp_pher))
            returns_temp = np.sum(returnsh.mean() * weights) * trading_days  # calculating annulaised portfolio return
            varsh = np.dot(weights.T,
                           np.dot(covmatsh, weights))  # calculating portfolio varience wrt calculating sharpe ratio
            volatility_temp = np.sqrt(varsh)  # portfolio risk
            fitness = ratio(returns_temp, volatility_temp, risk_free_rate)  # calculating sharpe ratio

            # comparing the old fitness value with the updated fitness value
            # explore on scape condition paper ref
            if (fitness > sharpe_pbest):
                # if the updated fitness value is better than the previous, change sharpe_pbest to present fitness value
                sharpe_pbest = fitness

                # remembering the weights of the best portfolio

                global_warr_sharpe = weights.tolist()

            sharpe_portfolio_return.append(returns_temp)
            sharpe_portfolio_risk.append(volatility_temp)
            sharpe_portfolio_shratio.append(fitness)
            sharpe_portfolio_stockWeights.append(weights)

    return sharpe_pbest


tuned=antcolony_sharpe(ITERATIONS,Q,EVA_RATE,ANTS)

sharpe_portfolio = {'Returns' : sharpe_portfolio_return, 'Standard Deviation' : sharpe_portfolio_risk,  'Sharpe Ratio' : sharpe_portfolio_shratio}

for counter,symbol in enumerate(df.columns):
  sharpe_portfolio[symbol + " Weight"] = [Weight[counter] for Weight in sharpe_portfolio_stockWeights]
sharpe_pc = pd.DataFrame(sharpe_portfolio)
sharpe_optimal=sharpe_pc.iloc[sharpe_pc['Sharpe Ratio'].idxmax()]
sharpe_optimal=pd.DataFrame(sharpe_optimal)
sharpe_optimal.to_csv("sharpe_optimal.csv")


sharpe_unsort_top=sharpe_pc.iloc[:,0:3].head(10)
sharpe_unsort_top.to_csv("sharpe_unsort_top.csv",)
sharpe_unsort_top

sharpe_unsort_top_all=sharpe_pc.head(10)
sharpe_unsort_top_all.to_csv("sharpe_unsort_top_all.csv")
sharpe_unsort_top_all

sharpe_unsort_bottom=sharpe_pc.iloc[:,0:3].tail(10)
sharpe_unsort_bottom.to_csv("sharpe_unsort_bottom.csv")
sharpe_unsort_bottom

sharpe_unsort_bottom_all=sharpe_pc.tail(10)
sharpe_unsort_bottom_all.to_csv("sharpe_unsort_bottom_all.csv")
sharpe_unsort_bottom_all

sharpe_pc.to_csv('sharpe_ACO_portfolio.csv')

sharpe_pc_sort=sharpe_pc.copy()

sharpe_pc_sort.sort_values(by=['Sharpe Ratio'], ascending=False, inplace=True)
sharpe_pc_sort.to_csv("sharpe_porfolio_sort.csv")


sharpe_sort_top = sharpe_pc_sort.iloc[1:, 0:3].head(11)
sharpe_sort_top.to_csv("sharpe_sort_top.csv")
sharpe_sort_top



sharpe_sort_top_all = sharpe_pc_sort.iloc[1:, 0:].head(11)
sharpe_sort_top_all.to_csv("sharpe_sort_top_all.csv")
sharpe_sort_top_all



sharpe_sort_bottom = sharpe_pc_sort.iloc[:, 0:3].tail(10)
sharpe_sort_bottom.to_csv("sharpe_sort_bottom.csv")
sharpe_sort_bottom



sharpe_sort_bottom_all = sharpe_pc_sort.iloc[:, 0:3].tail(10)
sharpe_sort_bottom_all.to_csv("sharpe_sort_bottom_all.csv")
sharpe_sort_bottom_all



# Sortino



def antcolony_tuning_sortino(ITERATIONS, Q, EVA_RATE, ANTS):
    sortino_pbest = -1
    # Initializing sortino_pbest(the best fitness value   SHARPE)
    # Initializing the current fitness value
    fitness = 0
    # for each iteration
    for iteration in range(ITERATIONS):

        # PREPARAING THE PHEROMONE MATRIX WHERE THE COLS=STOCKS AND  ROWS=ANTS
        pheromon = [[0] * stocks for i in
                    range(ANTS + 1)]  # why (ants+1)?The last ant can update the pheromone values in the last row

        # Initializing the pheromone status
        for i in range(len(pheromon[0])):
            pheromon[0][i] = random.randint(1,
                                            15)  # When input stocks varies, this needs to vary accordingly.(Divide number of stocks / 2)

        # copying the values and storing it in temp_pher
        temp_pher = pheromon[0]

        # Making sure that the total amount of pheromone equals 1
        weights = np.array(BALANCE(temp_pher))

        # calculating annulaised portfolio return
        returns_temp = np.sum(returnsh.mean() * weights) * trading_days

        # calculating portfolio varience wrt calculating sharpe ratio
        varso = np.dot(weights.T, np.dot(covmatso, weights))

        # portfolio risk
        volatility_temp = np.sqrt(varso)

        # Calculating fitness value(ie sortino ratio)
        fitness = ratio(returns_temp, volatility_temp, risk_free_rate)

        # Initializing the intial fitness value as the best fitness value(sortino_pbest)
        if sortino_pbest == -1:
            sortino_pbest = fitness

        # list
        path = []

        # for each ant
        for ant in range(ANTS - 1):

            # find the total pheromone
            total = sum(pheromon[ant])

            # Initializing probability
            probability = pheromon[ant][:]

            # finding probability of each stocks pheromone
            for p in range(len(probability)):
                probability[p] = (probability[p] / total)

            # Trying to select stocks in decreasing order based on their pheromone level and storing the stock order in a list(path)
            for stock in range(stocks):
                select = probability.index(max(probability))
                probability[select] = -math.inf
                path.append(select)

            # Updating the pheromone level of each stock for the next ant
            # Formula: old pheromone level * (1-eva_rate) + Q * (fitness/pbest) where Q is fixed amount of pheromone
            for s in path:
                pheromon[ant + 1][s] = pheromon[ant][s] * (1 - EVA_RATE) + Q * (fitness / sortino_pbest)

            # making sure that the updated pheromon adds upto 1
            temp_pher = pheromon[ant + 1]
            weights = np.array(BALANCE(temp_pher))
            returns_temp = np.sum(returnsh.mean() * weights) * trading_days  # calculating annulaised portfolio return
            varso = np.dot(weights.T,
                           np.dot(covmatso, weights))  # calculating portfolio varience wrt calculating sharpe ratio
            volatility_temp = np.sqrt(varso)  # portfolio risk
            fitness = ratio(returns_temp, volatility_temp, risk_free_rate)  # calculating sharpe ratio

            # comparing the old fitness value with the updated fitness value
            # explore on scape condition paper ref
            if (fitness > sortino_pbest):
                # if the updated fitness value is better than the previous, change sortino_pbest to present fitness value
                sortino_pbest = fitness

                # remembering the weights of the best portfolio
                global_warr_sortino = weights.tolist()
        # sortino_portfolio_return.append(returns_temp)
        # sortino_portfolio_risk.append(volatility_temp)
        # sortino_portfolio_soratio.append(fitness)
        # sortino_portfolio_stockWeights.append(weights)
    return sortino_pbest


# parameter values from excel sheet literature survey
def objective(trial):
    ITERATIONS = trial.suggest_int('ITERATIONS', 2, 550)
    Q = trial.suggest_float('Q', 0.0, 1.0)
    EVA_RATE = trial.suggest_float('EVA_RATE', 0.00, 1.00)
    ANTS = trial.suggest_int('ANTS', 2, 550)
    return antcolony_tuning_sortino(int(ITERATIONS), Q, EVA_RATE, int(ANTS))



sortino_study = optuna.create_study(direction='maximize')
sortino_study.optimize(objective, n_trials=100)


hptuning = sortino_study.trials_dataframe()
hptuning.to_csv("sortino_trial0.csv")
best = sortino_study.best_params


ITERATIONS = int(best['ITERATIONS'])
Q = best['Q']
EVA_RATE = best['EVA_RATE']
ANTS = int(best['ANTS'])



global_warr_sortino = []
sortino_portfolio_return = []
sortino_portfolio_risk = []
sortino_portfolio_soratio = []
sortino_portfolio_stockWeights = []



def antcolony_sortino(ITERATIONS, Q, EVA_RATE, ANTS):
    sortino_pbest = -1
    # Initializing sortino_pbest(the best fitness value   SHARPE)
    # Initializing the current fitness value
    fitness = 0
    # for each iteration
    for iteration in range(ITERATIONS):

        # PREPARAING THE PHEROMONE MATRIX WHERE THE COLS=STOCKS AND  ROWS=ANTS
        pheromon = [[0] * stocks for i in
                    range(ANTS + 1)]  # why (ants+1)?The last ant can update the pheromone values in the last row

        # Initializing the pheromone status
        for i in range(len(pheromon[0])):
            pheromon[0][i] = random.randint(1,
                                            15)  # When input stocks varies, this needs to vary accordingly.(Divide number of stocks / 2)

        # copying the values and storing it in temp_pher
        temp_pher = pheromon[0]

        # Making sure that the total amount of pheromone equals 1
        weights = np.array(BALANCE(temp_pher))

        # calculating annulaised portfolio return
        returns_temp = np.sum(returnsh.mean() * weights) * trading_days

        # calculating portfolio varience wrt calculating sharpe ratio
        varso = np.dot(weights.T, np.dot(covmatso, weights))

        # portfolio risk
        volatility_temp = np.sqrt(varso)

        # Calculating fitness value(ie sortino ratio)
        fitness = ratio(returns_temp, volatility_temp, risk_free_rate)

        # Initializing the intial fitness value as the best fitness value(sortino_pbest)
        if sortino_pbest == -1:
            sortino_pbest = fitness

        # list
        path = []

        # for each ant
        for ant in range(ANTS - 1):

            # find the total pheromone
            total = sum(pheromon[ant])

            # Initializing probability
            probability = pheromon[ant][:]

            # finding probability of each stocks pheromone
            for p in range(len(probability)):
                probability[p] = (probability[p] / total)

            # Trying to select stocks in decreasing order based on their pheromone level and storing the stock order in a list(path)
            for stock in range(stocks):
                select = probability.index(max(probability))
                probability[select] = -math.inf
                path.append(select)

            # Updating the pheromone level of each stock for the next ant
            # Formula: old pheromone level * (1-eva_rate) + Q * (fitness/pbest) where Q is fixed amount of pheromone
            for s in path:
                pheromon[ant + 1][s] = pheromon[ant][s] * (1 - EVA_RATE) + Q * (fitness / sortino_pbest)

            # making sure that the updated pheromon adds upto 1
            temp_pher = pheromon[ant + 1]
            weights = np.array(BALANCE(temp_pher))
            returns_temp = np.sum(returnsh.mean() * weights) * trading_days  # calculating annulaised portfolio return
            varso = np.dot(weights.T,
                           np.dot(covmatso, weights))  # calculating portfolio varience wrt calculating sharpe ratio
            volatility_temp = np.sqrt(varso)  # portfolio risk
            fitness = ratio(returns_temp, volatility_temp, risk_free_rate)  # calculating sharpe ratio

            # comparing the old fitness value with the updated fitness value
            # explore on scape condition paper ref
            if (fitness > sortino_pbest):
                # if the updated fitness value is better than the previous, change sortino_pbest to present fitness value
                sortino_pbest = fitness

                # remembering the weights of the best portfolio
                global_warr_sortino = weights.tolist()
            sortino_portfolio_return.append(returns_temp)
            sortino_portfolio_risk.append(volatility_temp)
            sortino_portfolio_soratio.append(fitness)
            sortino_portfolio_stockWeights.append(weights)
    return sortino_pbest


sortino_tuned = antcolony_sortino(ITERATIONS, Q, EVA_RATE, ANTS)


sortino_portfolio = {'Returns': sortino_portfolio_return, 'Standard Deviation': sortino_portfolio_risk,
                     'Sortino Ratio': sortino_portfolio_soratio}

for counter, symbol in enumerate(df.columns):
    sortino_portfolio[symbol + " Weight"] = [Weight[counter] for Weight in sortino_portfolio_stockWeights]
sortino_pc = pd.DataFrame(sortino_portfolio)
sortino_optimal = sortino_pc.iloc[sortino_pc['Sortino Ratio'].idxmax()]
sortino_optimal = pd.DataFrame(sortino_optimal)
sortino_optimal.to_csv("sortino_optimal.csv")
sortino_optimal


## not sort


sortino_unsort_top = sortino_pc.iloc[:, 0:3].head(10)
sortino_unsort_top.to_csv("sortino_unsort_top.csv")


sortino_unsort_top_all = sortino_pc.head(10)
sortino_unsort_top_all.to_csv("sortino_unsort_top_all.csv")



sortino_unsort_bottom = sortino_pc.iloc[:, 0:3].tail(10)
sortino_unsort_bottom.to_csv("sortino_unsort_bottom.csv")


sortino_unsort_bottom_all = sortino_pc.tail(10)
sortino_unsort_bottom_all.to_csv("sortino_unsort_bottom_all.csv")


sortino_pc.to_csv('sortino_ACO_portfolio.csv')

sortino_pc_sort = sortino_pc.copy()



## sort sortino



sortino_pc_sort.sort_values(by=['Sortino Ratio'], ascending=False, inplace=True)
sortino_pc_sort.to_csv("sortino_porfolio_sort.csv")



sortino_sort_top = sortino_pc_sort.iloc[1:, 0:3].head(11)
sortino_sort_top.to_csv("sortino_sort_top.csv")
sortino_sort_top



sortino_sort_top_all = sortino_pc_sort.iloc[1:, 0:].head(11)
sortino_sort_top_all.to_csv("sortino_sort_top_all.csv")
sortino_sort_top_all



sortino_sort_bottom = sortino_pc_sort.iloc[:, 0:3].tail(10)
sortino_sort_bottom.to_csv("sortino_sort_bottom.csv")
sortino_sort_bottom


sortino_sort_bottom_all = sortino_pc_sort.iloc[:, 0:3].tail(10)
sortino_sort_bottom_all.to_csv("sortino_sort_bottom_all.csv")
sortino_sort_bottom_all