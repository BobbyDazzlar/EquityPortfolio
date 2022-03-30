import pandas as pd
import numpy as np
import random
import optuna

df = pd.read_csv("/home/pn_kumar/Karthik/window-sliding/n50.csv", parse_dates=['Date'],
                 index_col='Date')  # Importing Dataset
df = df.loc["2016-01-05":"2021-01-04"]  # Since 2016-01-01, 5y(1234rows till 2020-12-31)
tdf = df.copy()  # deep copy
df.reset_index(drop=True, inplace=True)
col = list(df.columns)
tdf.shape
df1 = tdf.reset_index()
df1 = pd.DataFrame(df1)

df1 = list(df1.iloc[-1:])[1:]
stock_names = {}
for i in range(len(df1)):
    stock_names[df1[i]] = i + 1

df = pd.read_csv("/home/pn_kumar/Karthik/window-sliding/n50.csv", parse_dates=['Date'],
                 index_col='Date')  # Importing Dataset
df = df.loc["2016-01-05":"2021-01-04"]

n50 = df

trading_days = len(n50)
n50

## CALCULATING THE NUMBER OF YEARS

start_date = n50.index[0]  # date of first row in the dataset (datetime format)
end_date = n50.index[len(n50) - 1]  # date of last row in the dataset  (datetime format)
no_of_years = int((((
                                end_date - start_date).days + 1) / 365))  # the difference give the number of total days (not trading days) over the total number of years in the dataset

close = []
name = []
stock_no = []
daily_returns = []
for i in n50:
    t = list(n50[i])  # finding the close price col for each asset
    close.append(t)
    name.append(i)
    t = list(n50[i].pct_change())  # daily return for stock
    daily_returns.append(t)

closed = []
for i in close:
    for j in i:
        closed.append(j)
close = closed

daily_returns_arr = []

for i in daily_returns:  # HERE IN THIS CODE ,WE ARE TRYING TO PLACE ALL THE DAILY RETURNS IN A SINGLE LIST Daily_Returnss----><2>
    for j in i:
        daily_returns_arr.append(j)
daily_returns = daily_returns_arr

name_arr = []

x = n50.shape[0]  # FINDING NO OF ROWS, THE REASON IS EACH STOCK IS REPEATING X TIMES

for i in stock_names:  # ------><3>
    for j in range(x):
        name_arr.append(i)

name = name_arr

data = {}
data['Close'] = close
data['Daily_returns'] = daily_returns
data['Name'] = name
data = pd.DataFrame(data)
data

data.isna().sum()

## REMOVING NAN VALUES


data = data.fillna(0)  # .dropna(),it removes rows containing nan values

data

## DIMENSIONS REPRESENT THE NUMBER OF ASSETS


dimensions = len(stock_names)  # No of stocks


# iterations = 500
# swarm_size = 100 #No of portfolios


# SHARPE


def SHARPE(weights, mean_daily_returns, cov_matrix):
    weights = [w / sum(weights) for w in weights]  # Making sure all weights represent proportions that add up to 1
    weights = np.matrix(weights)
    port_return = np.round(np.sum(weights * mean_daily_returns.T) * trading_days, 2) / (
        no_of_years)  # 1259 trading days over 5 year period
    port_std_dev = np.round(np.sqrt(weights * cov_matrix * weights.T) * np.sqrt(trading_days), 2) / np.sqrt(no_of_years)
    port_std_dev = float(port_std_dev)
    sharpe_ratio = (
                           port_return - 0.0358) / port_std_dev  # 2.57 represents annual return of risk free security - 5-year US Treasury

    return sharpe_ratio


def PSO_HPTuning(swarm_size, iterations):
    mean_daily_returns = []  # IT WOULD CONTAIN THE AVG RETURS OF THE ASSETS
    all_return_input = []  # cov_input # IT WOULD CONTAIN THE [DAILY RETURNS] OF THE ASSETS
    neg_return = []

    for stock in stock_names:
        indv_stock = data[data.Name == stock]
        avg_return = indv_stock.Daily_returns.mean()
        mean_daily_returns.append(avg_return)
        all_return_input.append(indv_stock.Daily_returns.tolist())

    neg_return = all_return_input[:]
    mean_daily_returns = np.matrix(mean_daily_returns)  # CONVERTING 1D LIST(mean_daily_returns)INTO A MATRIX
    all_return_input = np.matrix(all_return_input)  # CONVERTING A 2D LIST(cov_input)INTO A MATRIX
    cov_matrix = np.cov(all_return_input)

    for i in range(len(neg_return)):
        for j in range(len(neg_return[i])):
            if neg_return[i][j] > 0:
                neg_return[i][j] = 0
    sortino_input = np.matrix(neg_return)
    cov_matrix_neg = np.cov(sortino_input)

    ''' w,c1,c2 ARE PARAMETERS WHICH WOULD HELP IN CHANGING THE PARTICLE(ASSET) POSITIONS'''
    '''INITIALIZING RANDOM POSITIONS(WEIGHTS) FOR ALL THE 100
    *11 ASSETS'''
    swarm_position = []
    swarm_velocity = []
    for particle in range(swarm_size):  # SWARM SIZE->PORTFOLIO
        '''FOR EACH NEW PORTFOLIO, WE WOULD CREATING A LIST (position)'''
        position = [0] * dimensions
        velocity = [0] * dimensions
        for dimension in range(dimensions):  # DIMENSIONS->ASSET

            position[dimension] = random.random()  # random.random() would assign some random number between 0 and 1
            velocity[dimension] = random.random()
        '''position LIST OF EVERY NEW PORTFOLIO WOULD BE KEPT IN A SEPARATE LIST(swarm_position)'''
        swarm_position.append(position)
        swarm_velocity.append(velocity)

    swarm_position = np.array([np.array(p) for p in swarm_position])  # COVERTING  swarm_position  list to an array
    swarm_velocity = np.array([np.array(v) for v in swarm_velocity])  # COVERTING  swarm_velocities  list to an array

    '''initial_swarm_positions IS A LIST , CONTAINING ALL THE INITIAL POSITIONS OF ALL THE ASSETS OF ALL THE PORTFOLIOS'''
    initial_swarm_positions = swarm_position  # HERE IS THE MASTER FILE WHICH WE WOULD BE USING FOR COMPARISON
    '''swarm_gbest,WE ARE ASSUMING THAT THE FIRST PORTFOLIO IS THE BEST OPTION AVAILABLE '''
    swarm_gbest = initial_swarm_positions[0]

    avg_sharpe_list = []
    portfolio_return = []
    portfolio_standard_deviation = []
    portfolio_sortino = []
    portfolio_semi_deviation = []
    portfolio_sharpe = []
    portfolio_weights = []

    for iteration in range(iterations):
        '''FOR EACH ITERATION'''
        '''sharpe_pbest_all is a list which would contain the sharpe ratio of the portfolios,Finally at the end it would containing 100 sharpe ratios'''
        sharpe_pbest_all = []
        for particle in range(swarm_size):  # SWARM_SIZE->PORTFOLIOS
            '''FOR EACH PORTFOLIO'''
            for dimension in range(dimensions):  # DIMENSIONS->ASSETS
                '''FOR EACH ASSET'''
                '''HERE , WE WOULD BE UPDATING THE POSITIONS AND VELOCITES OF EACH OF THE ASSET PRESENT IN A PARTICULAR PORTFOLIO'''
                r1 = random.random()  # random.random() WOULD BASICALLY GIVE YOU A VALUE BETWEEN 0 AND 1
                r2 = random.random()
                '''HERE WE WOULD BE CHANGING THE PRESENT PARTICLE(ASSET) POSITION 
                   AND FOR CHANGING THE POSITION,WE NEED TO CALCULATE THE NEW VELOCITY'''

                '''BASIC FORMULA 
                   NEW VELOCITY->w*PRESENT VELOCITY + c1 * r1 * abs(INTIAL POSITION - PRESENT POSITION)+c2*r2*abs(GBEST POSITION - PRESENT POSITION)
                   NEW POSITION->PRESENT POSITION + NEW VELOCITY
                '''
                w = (0.9 - 0.4) * ((
                                           iterations - iteration) / iterations) + 0.4  # w at iteration, where initial_w = 0.9 & final_w = 0.4
                c1 = (0.5 - 2.5) * (iteration / iterations) + 2.5  # c1 at iteration, where min_c1 = 0.5 & max_c1 = 2.5
                c2 = (2.5 - 0.5) * (iteration / iterations) + 0.5  # c2 at iteration, where max_c2 = 2.5 & min_c2 = 0.5
                swarm_velocity[particle][dimension] = w * swarm_velocity[particle][dimension] + c1 * r1 * abs(
                    initial_swarm_positions[particle][dimension] - swarm_position[particle][dimension]) + c2 * r2 * abs(
                    swarm_gbest[dimension] - swarm_position[particle][dimension])  # Update velocity in every dimension
                swarm_position[particle][dimension] = swarm_position[particle][dimension] + swarm_velocity[particle][
                    dimension]  # Update position in every direction

            # AFTER 1 COMPLETE PORTFOLIO

            '''AFTER CHANGING THE POSITIONS OF ALL THE ASSETS IN A PARTICULAR PORTFOLIO, WE WOULD FIND THE SHARPE RATIO OF THIS PORTFOLIO
               AND WE WOULD STORE IT IN sharpe_pbest'''
            sharpe_pbest = SHARPE(initial_swarm_positions[particle], mean_daily_returns,
                                  cov_matrix)  # Evaluating sharpe of existing pbest position

            '''THEN WE WOULD BE COMPARING THE sharpe_pbest WITH THE SHARPE RATIO OF THE INITIAL PORTFOLIO WHICH IS STORED IN initial_swarm_positions
               ie COMPARING SHARPE RATIO OF THE PORTFOLIO OLD POSITION WITH SHARPE RATIO OF THE PORTFOLIO NEW POSITION'''

            if SHARPE(swarm_position[particle], mean_daily_returns, cov_matrix) > sharpe_pbest:
                '''IS THE NEW POSITION BETTER THAN THE EXISTING ONE'''
                '''IF THE NEW POSITION IS BETTER, THEN UPDATE sharpe_pbest  and initial_swarm_positions[particle]'''
                initial_swarm_positions[particle] = swarm_position[particle]  # Update pbest to new position
                sharpe_pbest = SHARPE(initial_swarm_positions[particle], mean_daily_returns,
                                      cov_matrix)  # Update sharpe of pbest
            '''IF THE PORTFOLIO WITH NEW POSITION HAS HIGHER SHARPE RATIO THAN
               THE OLD POSITION,APPEND SHARPE RATIO OF THE PORTFOLIO WRT TO NEW POSTION,
               ELSE APPEND SHARPE RATIO OF THE PORTFOLIO WRT TO OLD POSTION '''
            sharpe_pbest_all.append(sharpe_pbest)
            '''AFTER ALL THE PORTFOLIOS(ie,AFTER ONE ITERATION) ARE DONE, WE WOULD SELECT THE BEST PORTFOLIO
               WRT OF SHARPE RATIO AND COMPARE IT WITH THE SHARPE RATIO OF GLOBAL BEST(swarm_gbest PORTFOLIO)'''

        # AFTER 1 COMPLETE ITERATION

        if max(sharpe_pbest_all) > SHARPE(swarm_gbest, mean_daily_returns, cov_matrix):
            ''' IS THE LARGEST SHARPE RATIO OF ALL THE PORTFOLIOS BETTER THEN gbest?'''
            ''' IF YES, CHANGE THE gbest to max(sharpe_pbest_all)'''
            max_index = sharpe_pbest_all.index(max(sharpe_pbest_all))
            swarm_gbest = initial_swarm_positions[max_index]

    return SHARPE(swarm_gbest, mean_daily_returns, cov_matrix)


### SETTING THE HYPERPARAMETERS FROM EXCEL SHEET OF SURVEY OF PSO PAPERS


def objective(trial):
    swarm_size = trial.suggest_uniform('swarm_size', 10, 100)
    iterations = trial.suggest_uniform('iterations', 100, 500)
    return PSO_HPTuning(int(swarm_size), int(iterations))


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

best = study.best_params
### BEST PARAMETERS FOR PSO


swarm_size = int(best['swarm_size'])
iterations = int(best['iterations'])


## PSO FOR SHARPE RATIO


def PSO_OPTIMIZER(swarm_size, iterations):
    mean_daily_returns = []  # IT WOULD CONTAIN THE AVG RETURS OF THE ASSETS
    all_return_input = []  # cov_input # IT WOULD CONTAIN THE [DAILY RETURNS] OF THE ASSETS
    neg_return = []

    for stock in stock_names:
        indv_stock = data[data.Name == stock]
        avg_return = indv_stock.Daily_returns.mean()
        mean_daily_returns.append(avg_return)
        all_return_input.append(indv_stock.Daily_returns.tolist())

    neg_return = all_return_input[:]
    mean_daily_returns = np.matrix(mean_daily_returns)  # CONVERTING 1D LIST(mean_daily_returns)INTO A MATRIX
    all_return_input = np.matrix(all_return_input)  # CONVERTING A 2D LIST(cov_input)INTO A MATRIX
    cov_matrix = np.cov(all_return_input)

    for i in range(len(neg_return)):
        for j in range(len(neg_return[i])):
            if neg_return[i][j] > 0:
                neg_return[i][j] = 0
    sortino_input = np.matrix(neg_return)
    cov_matrix_neg = np.cov(sortino_input)

    ''' w,c1,c2 ARE PARAMETERS WHICH WOULD HELP IN CHANGING THE PARTICLE(ASSET) POSITIONS'''
    '''INITIALIZING RANDOM POSITIONS(WEIGHTS) FOR ALL THE 100
    *11 ASSETS'''
    swarm_position = []
    swarm_velocity = []
    for particle in range(swarm_size):  # SWARM SIZE->PORTFOLIO
        '''FOR EACH NEW PORTFOLIO, WE WOULD CREATING A LIST (position)'''
        position = [0] * dimensions
        velocity = [0] * dimensions
        for dimension in range(dimensions):  # DIMENSIONS->ASSET

            position[dimension] = random.random()  # random.random() would assign some random number between 0 and 1
            velocity[dimension] = random.random()
        '''position LIST OF EVERY NEW PORTFOLIO WOULD BE KEPT IN A SEPARATE LIST(swarm_position)'''
        swarm_position.append(position)
        swarm_velocity.append(velocity)

    swarm_position = np.array([np.array(p) for p in swarm_position])  # COVERTING  swarm_position  list to an array
    swarm_velocity = np.array([np.array(v) for v in swarm_velocity])  # COVERTING  swarm_velocities  list to an array

    '''initial_swarm_positions IS A LIST , CONTAINING ALL THE INITIAL POSITIONS OF ALL THE ASSETS OF ALL THE PORTFOLIOS'''
    initial_swarm_positions = swarm_position  # HERE IS THE MASTER FILE WHICH WE WOULD BE USING FOR COMPARISON
    '''swarm_gbest,WE ARE ASSUMING THAT THE FIRST PORTFOLIO IS THE BEST OPTION AVAILABLE '''
    swarm_gbest = initial_swarm_positions[0]

    avg_sharpe_list = []
    portfolio_return = []
    portfolio_standard_deviation = []
    portfolio_sortino = []
    portfolio_semi_deviation = []
    portfolio_sharpe = []
    portfolio_weights = []

    for iteration in range(iterations):
        '''FOR EACH ITERATION'''
        '''sharpe_pbest_all is a list which would contain the sharpe ratio of the portfolios,Finally at the end it would containing 100 sharpe ratios'''
        sharpe_pbest_all = []
        for particle in range(swarm_size):  # SWARM_SIZE->PORTFOLIOS
            '''FOR EACH PORTFOLIO'''
            for dimension in range(dimensions):  # DIMENSIONS->ASSETS
                '''FOR EACH ASSET'''
                '''HERE , WE WOULD BE UPDATING THE POSITIONS AND VELOCITES OF EACH OF THE ASSET PRESENT IN A PARTICULAR PORTFOLIO'''
                r1 = random.random()  # random.random() WOULD BASICALLY GIVE YOU A VALUE BETWEEN 0 AND 1
                r2 = random.random()
                '''HERE WE WOULD BE CHANGING THE PRESENT PARTICLE(ASSET) POSITION 
                   AND FOR CHANGING THE POSITION,WE NEED TO CALCULATE THE NEW VELOCITY'''

                '''BASIC FORMULA 
                   NEW VELOCITY->w*PRESENT VELOCITY + c1 * r1 * abs(INTIAL POSITION - PRESENT POSITION)+c2*r2*abs(GBEST POSITION - PRESENT POSITION)
                   NEW POSITION->PRESENT POSITION + NEW VELOCITY
                '''
                w = (0.9 - 0.4) * ((
                                           iterations - iteration) / iterations) + 0.4  # w at iteration, where initial_w = 0.9 & final_w = 0.4
                c1 = (0.5 - 2.5) * (iteration / iterations) + 2.5  # c1 at iteration, where min_c1 = 0.5 & max_c1 = 2.5
                c2 = (2.5 - 0.5) * (iteration / iterations) + 0.5  # c2 at iteration, where max_c2 = 2.5 & min_c2 = 0.5
                swarm_velocity[particle][dimension] = w * swarm_velocity[particle][dimension] + c1 * r1 * abs(
                    initial_swarm_positions[particle][dimension] - swarm_position[particle][dimension]) + c2 * r2 * abs(
                    swarm_gbest[dimension] - swarm_position[particle][dimension])  # Update velocity in every dimension
                swarm_position[particle][dimension] = swarm_position[particle][dimension] + swarm_velocity[particle][
                    dimension]  # Update position in every direction

            # AFTER 1 COMPLETE PORTFOLIO

            '''AFTER CHANGING THE POSITIONS OF ALL THE ASSETS IN A PARTICULAR PORTFOLIO, WE WOULD FIND THE SHARPE RATIO OF THIS PORTFOLIO
               AND WE WOULD STORE IT IN sharpe_pbest'''
            sharpe_pbest = SHARPE(initial_swarm_positions[particle], mean_daily_returns,
                                  cov_matrix)  # Evaluating sharpe of existing pbest position

            '''THEN WE WOULD BE COMPARING THE sharpe_pbest WITH THE SHARPE RATIO OF THE INITIAL PORTFOLIO WHICH IS STORED IN initial_swarm_positions
               ie COMPARING SHARPE RATIO OF THE PORTFOLIO OLD POSITION WITH SHARPE RATIO OF THE PORTFOLIO NEW POSITION'''

            if SHARPE(swarm_position[particle], mean_daily_returns, cov_matrix) > sharpe_pbest:
                '''IS THE NEW POSITION BETTER THAN THE EXISTING ONE'''
                '''IF THE NEW POSITION IS BETTER, THEN UPDATE sharpe_pbest  and initial_swarm_positions[particle]'''
                initial_swarm_positions[particle] = swarm_position[particle]  # Update pbest to new position
                sharpe_pbest = SHARPE(initial_swarm_positions[particle], mean_daily_returns,
                                      cov_matrix)  # Update sharpe of pbest
            '''IF THE PORTFOLIO WITH NEW POSITION HAS HIGHER SHARPE RATIO THAN
               THE OLD POSITION,APPEND SHARPE RATIO OF THE PORTFOLIO WRT TO NEW POSTION,
               ELSE APPEND SHARPE RATIO OF THE PORTFOLIO WRT TO OLD POSTION '''
            sharpe_pbest_all.append(sharpe_pbest)
            '''AFTER ALL THE PORTFOLIOS(ie,AFTER ONE ITERATION) ARE DONE, WE WOULD SELECT THE BEST PORTFOLIO
               WRT OF SHARPE RATIO AND COMPARE IT WITH THE SHARPE RATIO OF GLOBAL BEST(swarm_gbest PORTFOLIO)'''

        # AFTER 1 COMPLETE ITERATION

        if max(sharpe_pbest_all) > SHARPE(swarm_gbest, mean_daily_returns, cov_matrix):
            ''' IS THE LARGEST SHARPE RATIO OF ALL THE PORTFOLIOS BETTER THEN gbest?'''
            ''' IF YES, CHANGE THE gbest to max(sharpe_pbest_all)'''
            max_index = sharpe_pbest_all.index(max(sharpe_pbest_all))
            swarm_gbest = initial_swarm_positions[max_index]

        '''AFTER 1 ITERATION IS DONE,ie GOING THROUGH ALL THE PORTFOLIOS AND FINDING THE BEST PORTFOLIO,
           WE WOULD NOW PRINT THE VALUES
        '''
        ''' CALCULATING THE AVG SHARPE RATIO FOR THE ITERATION '''
        avg_sharpe = sum(sharpe_pbest_all) / len(sharpe_pbest_all)

        avg_sharpe_list.append(avg_sharpe)
        '''NOW FOR PRINTING THE IMPORTANT DATA FROM EACH ITERATION, WE HAVE CREATED A SEPARATE FUNCTION(PRINT_EACH_ITERATION)'''
        # portfolio_return, portfolio_standard_deviation, portfolio_sharpe,portfolio_semi_deviation, portfolio_sortino,  portfolio_weights =
        # PRINT_EACH_ITERATION(iteration, swarm_gbest, mean_daily_returns, cov_matrix,cov_matrix_neg, portfolio_return, portfolio_standard_deviation, portfolio_sharpe,portfolio_semi_deviation, portfolio_sortino,  portfolio_weights)

        weights_arr = [w / sum(swarm_gbest) for w in
                       swarm_gbest]  # Making sure all weights represent proportions that add up to 1
        weights = np.matrix(weights_arr)
        port_return = np.round(np.sum(weights * mean_daily_returns.T) * trading_days, 2) / (
            no_of_years)  # 1259 trading days over 5 year period
        port_std_dev = np.round(np.sqrt(weights * cov_matrix * weights.T) * np.sqrt(trading_days), 2) / np.sqrt(
            no_of_years)
        port_std_dev = float(port_std_dev)
        sharpe_ratio = (
                               port_return - 0.0358) / port_std_dev  # 3.58 represents annual return of risk free security - 5-year US Treasury
        port_semi_dev = np.round(np.sqrt(weights * cov_matrix_neg * weights.T) * np.sqrt(trading_days), 2) / np.sqrt(
            no_of_years)
        port_semi_dev = float(port_semi_dev)
        sortino_ratio = (port_return - 0.0358) / (port_semi_dev)
        portfolio_return.append(port_return)  # Adding portfolio return of a given PORTFOLIO  to  portfolio_return
        portfolio_standard_deviation.append(
            port_std_dev)  # Adding portfolio standard deviation of a given PORTFOLIO to a list of standard deviations to portfolio_vol
        portfolio_sharpe.append(
            sharpe_ratio)  # Adding portfolio sharpe ratio of a given PORTFOLIO  to   portfolio_sharpe
        portfolio_semi_deviation.append(
            port_semi_dev)  # Adding portfolio standard deviation of a given PORTFOLIO to a list of standard deviations to portfolio_vol
        portfolio_sortino.append(
            sortino_ratio)  # Adding portfolio sharpe ratio of a given PORTFOLIO  to   portfolio_sharpe
        portfolio_weights.append(weights_arr)  # Adding portfolio weights of a given PORTFOLIO  to   portfolio_sharpe

    # AFTER ALL THE ITERATIONS
    return portfolio_return, portfolio_standard_deviation, portfolio_sharpe, portfolio_semi_deviation, portfolio_sortino, portfolio_weights


avg_sharpe_list = []
portfolio_return = []
portfolio_standard_deviation = []
portfolio_sortino = []
portfolio_semi_deviation = []
portfolio_sharpe = []
portfolio_weights = []

portfolio_return, portfolio_standard_deviation, portfolio_sharpe, portfolio_semi_deviation, portfolio_sortino, portfolio_weights = PSO_OPTIMIZER(
    swarm_size, iterations)

sharpe_portfolio = {'Returns': portfolio_return, 'Standard Deviation': portfolio_standard_deviation,
                    'Sharpe Ratio': portfolio_sharpe}

for counter, symbol in enumerate(n50.columns):
    sharpe_portfolio[symbol + " Weight"] = [Weight[counter] for Weight in portfolio_weights]
sharpe_pc = pd.DataFrame(sharpe_portfolio)

sharpe_pc.loc[:, :] *= 100
sharpe_pc.loc[:, 'Sharpe Ratio'] /= 100

sharpe_pc

sorted_sharpe = sharpe_pc.sort_values(by=['Sharpe Ratio'], ascending=False)

sorted_sharpe

optimal_portfolio = sorted_sharpe.head(1)

optimal_portfolio.T

optimal_portfolio.T.to_csv('/home/pn_kumar/Karthik/window-sliding/PSO(sharpe)_optimal_portfolio.csv')


# SORTINO


def SORTINO(weights, mean_daily_returns, cov_matrix_neg):
    ''' MAKING SURE THAT ALL THE WEIGHTS OF THE PORTFOLIO EQUALS TO 1'''
    weights = [w / sum(weights) for w in weights]
    ''' CONVERTING weights TO A  MATRIX'''
    weights = np.matrix(weights)
    ''' CALCULATING THE PORTFOLIO RETURNS'''
    port_return = np.round(np.sum(weights * mean_daily_returns.T) * trading_days,
                           2) / no_of_years  # 5yrs+91 days forecasted data from dhiraj's output,thus 5.25 yrs in total

    port_semi_dev = np.round(np.sqrt(weights * cov_matrix_neg * weights.T) * np.sqrt(trading_days), 2) / np.sqrt(
        no_of_years)
    port_semi_dev = float(port_semi_dev)
    # referred from url: https://www.rbi.org.in/Scripts/BS_NSDPDisplay.aspx?param=4&Id=24292
    # In the above url, the 364 (1 year) day treasury bill is 3.58% , when taken absolute value => 0.0358
    sortino_ratio = (port_return - 0.0358) / (port_semi_dev)  # RFR FOR 2020->3.58%

    return sortino_ratio


def PSO_HPTuning_Sortino(swarm_size, iterations):
    mean_daily_returns = []  # IT WOULD CONTAIN THE AVG RETURS OF THE ASSETS
    all_return_input = []  # cov_input # IT WOULD CONTAIN THE [DAILY RETURNS] OF THE ASSETS
    neg_return = []

    for stock in stock_names:
        indv_stock = data[data.Name == stock]
        avg_return = indv_stock.Daily_returns.mean()
        mean_daily_returns.append(avg_return)
        all_return_input.append(indv_stock.Daily_returns.tolist())

    neg_return = all_return_input[:]
    mean_daily_returns = np.matrix(mean_daily_returns)  # CONVERTING 1D LIST(mean_daily_returns)INTO A MATRIX
    all_return_input = np.matrix(all_return_input)  # CONVERTING A 2D LIST(cov_input)INTO A MATRIX
    cov_matrix = np.cov(all_return_input)

    for i in range(len(neg_return)):
        for j in range(len(neg_return[i])):
            if neg_return[i][j] > 0:
                neg_return[i][j] = 0
    sortino_input = np.matrix(neg_return)
    cov_matrix_neg = np.cov(sortino_input)

    swarm_position = []
    swarm_velocity = []
    for particle in range(swarm_size):  # SWARM SIZE->PORTFOLIO
        '''FOR EACH NEW PORTFOLIO, WE WOULD CREATING A LIST (position)'''
        position = [0] * dimensions
        velocity = [0] * dimensions
        for dimension in range(dimensions):  # DIMENSIONS->ASSET

            position[dimension] = random.random()  # random.random() would assign some random number between 0 and 1
            velocity[dimension] = random.random()
        '''position LIST OF EVERY NEW PORTFOLIO WOULD BE KEPT IN A SEPARATE LIST(swarm_position)'''
        swarm_position.append(position)
        swarm_velocity.append(velocity)

    swarm_position = np.array([np.array(p) for p in swarm_position])  # COVERTING  swarm_position  list to an array
    swarm_velocity = np.array([np.array(v) for v in swarm_velocity])  # COVERTING  swarm_velocities  list to an array

    initial_swarm_positions = swarm_position  # HERE IS THE MASTER FILE WHICH WE WOULD BE USING FOR COMPARISON
    '''swarm_gbest,WE ARE ASSUMING THAT THE FIRST PORTFOLIO IS THE BEST OPTION AVAILABLE '''
    swarm_gbest = initial_swarm_positions[0]

    avg_sortino_list = []
    portfolio_return = []
    portfolio_sortino = []
    portfolio_semi_deviation = []
    portfolio_weights = []

    for iteration in range(iterations):
        '''FOR EACH ITERATION'''
        '''sharpe_pbest_all is a list which would contain the sharpe ratio of the portfolios,Finally at the end it would containing 100 sharpe ratios'''
        sortino_pbest_all = []
        for particle in range(swarm_size):  # SWARM_SIZE->PORTFOLIOS
            '''FOR EACH PORTFOLIO'''
            for dimension in range(dimensions):  # DIMENSIONS->ASSETS
                '''FOR EACH ASSET'''
                '''HERE , WE WOULD BE UPDATING THE POSITIONS AND VELOCITES OF EACH OF THE ASSET PRESENT IN A PARTICULAR PORTFOLIO'''
                r1 = random.random()  # random.random() WOULD BASICALLY GIVE YOU A VALUE BETWEEN 0 AND 1
                r2 = random.random()
                '''HERE WE WOULD BE CHANGING THE PRESENT PARTICLE(ASSET) POSITION 
                   AND FOR CHANGING THE POSITION,WE NEED TO CALCULATE THE NEW VELOCITY'''

                '''BASIC FORMULA 
                   NEW VELOCITY->w*PRESENT VELOCITY + c1 * r1 * abs(INTIAL POSITION - PRESENT POSITION)+c2*r2*abs(GBEST POSITION - PRESENT POSITION)
                   NEW POSITION->PRESENT POSITION + NEW VELOCITY
                '''
                w = (0.9 - 0.4) * ((
                                           iterations - iteration) / iterations) + 0.4  # w at iteration, where initial_w = 0.9 & final_w = 0.4
                c1 = (0.5 - 2.5) * (iteration / iterations) + 2.5  # c1 at iteration, where min_c1 = 0.5 & max_c1 = 2.5
                c2 = (2.5 - 0.5) * (iteration / iterations) + 0.5  # c2 at iteration, where max_c2 = 2.5 & min_c2 = 0.5
                swarm_velocity[particle][dimension] = w * swarm_velocity[particle][dimension] + c1 * r1 * abs(
                    initial_swarm_positions[particle][dimension] - swarm_position[particle][dimension]) + c2 * r2 * abs(
                    swarm_gbest[dimension] - swarm_position[particle][dimension])  # Update velocity in every dimension
                swarm_position[particle][dimension] = swarm_position[particle][dimension] + swarm_velocity[particle][
                    dimension]  # Update position in every direction

            # AFTER 1 COMPLETE PORTFOLIO

            '''AFTER CHANGING THE POSITIONS OF ALL THE ASSETS IN A PARTICULAR PORTFOLIO, WE WOULD FIND THE SHARPE RATIO OF THIS PORTFOLIO
               AND WE WOULD STORE IT IN sharpe_pbest'''
            sortino_pbest = SORTINO(initial_swarm_positions[particle], mean_daily_returns,
                                    cov_matrix_neg)  # Evaluating sharpe of existing pbest position

            '''THEN WE WOULD BE COMPARING THE sharpe_pbest WITH THE SHARPE RATIO OF THE INITIAL PORTFOLIO WHICH IS STORED IN initial_swarm_positions
               ie COMPARING SHARPE RATIO OF THE PORTFOLIO OLD POSITION WITH SHARPE RATIO OF THE PORTFOLIO NEW POSITION'''

            if SORTINO(swarm_position[particle], mean_daily_returns, cov_matrix_neg) > sortino_pbest:
                '''IS THE NEW POSITION BETTER THAN THE EXISTING ONE'''
                '''IF THE NEW POSITION IS BETTER, THEN UPDATE sharpe_pbest  and initial_swarm_positions[particle]'''
                initial_swarm_positions[particle] = swarm_position[particle]  # Update pbest to new position
                sortino_pbest = SORTINO(initial_swarm_positions[particle], mean_daily_returns,
                                        cov_matrix_neg)  # Update sharpe of pbest
            '''IF THE PORTFOLIO WITH NEW POSITION HAS HIGHER SHARPE RATIO THAN
               THE OLD POSITION,APPEND SHARPE RATIO OF THE PORTFOLIO WRT TO NEW POSTION,
               ELSE APPEND SHARPE RATIO OF THE PORTFOLIO WRT TO OLD POSTION '''
            sortino_pbest_all.append(sortino_pbest)
            '''AFTER ALL THE PORTFOLIOS(ie,AFTER ONE ITERATION) ARE DONE, WE WOULD SELECT THE BEST PORTFOLIO
               WRT OF SHARPE RATIO AND COMPARE IT WITH THE SHARPE RATIO OF GLOBAL BEST(swarm_gbest PORTFOLIO)'''

        # AFTER 1 COMPLETE ITERATION

        if max(sortino_pbest_all) > SORTINO(swarm_gbest, mean_daily_returns, cov_matrix_neg):
            max_index = sortino_pbest_all.index(max(sortino_pbest_all))
            swarm_gbest = initial_swarm_positions[max_index]

    return SORTINO(swarm_gbest, mean_daily_returns, cov_matrix_neg)


def objective(trial):
    swarm_size = trial.suggest_uniform('swarm_size', 10, 100)
    iterations = trial.suggest_uniform('iterations', 100, 200)
    return PSO_HPTuning_Sortino(int(swarm_size), int(iterations))


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

best = study.best_params

swarm_size = int(best['swarm_size'])
iterations = int(best['iterations'])


def PSO_OPTIMIZER_SORTINO(swarm_size, iterations):
    mean_daily_returns = []  # IT WOULD CONTAIN THE AVG RETURS OF THE ASSETS
    all_return_input = []  # cov_input # IT WOULD CONTAIN THE [DAILY RETURNS] OF THE ASSETS
    neg_return = []

    for stock in stock_names:
        indv_stock = data[data.Name == stock]
        avg_return = indv_stock.Daily_returns.mean()
        mean_daily_returns.append(avg_return)
        all_return_input.append(indv_stock.Daily_returns.tolist())

    neg_return = all_return_input[:]
    mean_daily_returns = np.matrix(mean_daily_returns)  # CONVERTING 1D LIST(mean_daily_returns)INTO A MATRIX
    all_return_input = np.matrix(all_return_input)  # CONVERTING A 2D LIST(cov_input)INTO A MATRIX
    cov_matrix = np.cov(all_return_input)

    for i in range(len(neg_return)):
        for j in range(len(neg_return[i])):
            if neg_return[i][j] > 0:
                neg_return[i][j] = 0
    sortino_input = np.matrix(neg_return)
    cov_matrix_neg = np.cov(sortino_input)

    swarm_position = []
    swarm_velocity = []
    for particle in range(swarm_size):  # SWARM SIZE->PORTFOLIO
        '''FOR EACH NEW PORTFOLIO, WE WOULD CREATING A LIST (position)'''
        position = [0] * dimensions
        velocity = [0] * dimensions
        for dimension in range(dimensions):  # DIMENSIONS->ASSET

            position[dimension] = random.random()  # random.random() would assign some random number between 0 and 1
            velocity[dimension] = random.random()
        '''position LIST OF EVERY NEW PORTFOLIO WOULD BE KEPT IN A SEPARATE LIST(swarm_position)'''
        swarm_position.append(position)
        swarm_velocity.append(velocity)

    swarm_position = np.array([np.array(p) for p in swarm_position])  # COVERTING  swarm_position  list to an array
    swarm_velocity = np.array([np.array(v) for v in swarm_velocity])  # COVERTING  swarm_velocities  list to an array

    initial_swarm_positions = swarm_position  # HERE IS THE MASTER FILE WHICH WE WOULD BE USING FOR COMPARISON
    '''swarm_gbest,WE ARE ASSUMING THAT THE FIRST PORTFOLIO IS THE BEST OPTION AVAILABLE '''
    swarm_gbest = initial_swarm_positions[0]

    avg_sortino_list = []
    portfolio_return = []
    portfolio_sortino = []
    portfolio_semi_deviation = []
    portfolio_weights = []

    for iteration in range(iterations):
        '''FOR EACH ITERATION'''
        '''sharpe_pbest_all is a list which would contain the sharpe ratio of the portfolios,Finally at the end it would containing 100 sharpe ratios'''
        sortino_pbest_all = []
        for particle in range(swarm_size):  # SWARM_SIZE->PORTFOLIOS
            '''FOR EACH PORTFOLIO'''
            for dimension in range(dimensions):  # DIMENSIONS->ASSETS
                '''FOR EACH ASSET'''
                '''HERE , WE WOULD BE UPDATING THE POSITIONS AND VELOCITES OF EACH OF THE ASSET PRESENT IN A PARTICULAR PORTFOLIO'''
                r1 = random.random()  # random.random() WOULD BASICALLY GIVE YOU A VALUE BETWEEN 0 AND 1
                r2 = random.random()
                '''HERE WE WOULD BE CHANGING THE PRESENT PARTICLE(ASSET) POSITION 
                   AND FOR CHANGING THE POSITION,WE NEED TO CALCULATE THE NEW VELOCITY'''

                '''BASIC FORMULA 
                   NEW VELOCITY->w*PRESENT VELOCITY + c1 * r1 * abs(INTIAL POSITION - PRESENT POSITION)+c2*r2*abs(GBEST POSITION - PRESENT POSITION)
                   NEW POSITION->PRESENT POSITION + NEW VELOCITY
                '''
                w = (0.9 - 0.4) * ((
                                           iterations - iteration) / iterations) + 0.4  # w at iteration, where initial_w = 0.9 & final_w = 0.4
                c1 = (0.5 - 2.5) * (iteration / iterations) + 2.5  # c1 at iteration, where min_c1 = 0.5 & max_c1 = 2.5
                c2 = (2.5 - 0.5) * (iteration / iterations) + 0.5  # c2 at iteration, where max_c2 = 2.5 & min_c2 = 0.5
                swarm_velocity[particle][dimension] = w * swarm_velocity[particle][dimension] + c1 * r1 * abs(
                    initial_swarm_positions[particle][dimension] - swarm_position[particle][dimension]) + c2 * r2 * abs(
                    swarm_gbest[dimension] - swarm_position[particle][dimension])  # Update velocity in every dimension
                swarm_position[particle][dimension] = swarm_position[particle][dimension] + swarm_velocity[particle][
                    dimension]  # Update position in every direction

            # AFTER 1 COMPLETE PORTFOLIO

            '''AFTER CHANGING THE POSITIONS OF ALL THE ASSETS IN A PARTICULAR PORTFOLIO, WE WOULD FIND THE SHARPE RATIO OF THIS PORTFOLIO
               AND WE WOULD STORE IT IN sharpe_pbest'''
            sortino_pbest = SORTINO(initial_swarm_positions[particle], mean_daily_returns,
                                    cov_matrix_neg)  # Evaluating sharpe of existing pbest position

            '''THEN WE WOULD BE COMPARING THE sharpe_pbest WITH THE SHARPE RATIO OF THE INITIAL PORTFOLIO WHICH IS STORED IN initial_swarm_positions
               ie COMPARING SHARPE RATIO OF THE PORTFOLIO OLD POSITION WITH SHARPE RATIO OF THE PORTFOLIO NEW POSITION'''

            if SORTINO(swarm_position[particle], mean_daily_returns, cov_matrix_neg) > sortino_pbest:
                '''IS THE NEW POSITION BETTER THAN THE EXISTING ONE'''
                '''IF THE NEW POSITION IS BETTER, THEN UPDATE sharpe_pbest  and initial_swarm_positions[particle]'''
                initial_swarm_positions[particle] = swarm_position[particle]  # Update pbest to new position
                sortino_pbest = SORTINO(initial_swarm_positions[particle], mean_daily_returns,
                                        cov_matrix_neg)  # Update sharpe of pbest
            '''IF THE PORTFOLIO WITH NEW POSITION HAS HIGHER SHARPE RATIO THAN
               THE OLD POSITION,APPEND SHARPE RATIO OF THE PORTFOLIO WRT TO NEW POSTION,
               ELSE APPEND SHARPE RATIO OF THE PORTFOLIO WRT TO OLD POSTION '''
            sortino_pbest_all.append(sortino_pbest)
            '''AFTER ALL THE PORTFOLIOS(ie,AFTER ONE ITERATION) ARE DONE, WE WOULD SELECT THE BEST PORTFOLIO
               WRT OF SHARPE RATIO AND COMPARE IT WITH THE SHARPE RATIO OF GLOBAL BEST(swarm_gbest PORTFOLIO)'''

        # AFTER 1 COMPLETE ITERATION

        if max(sortino_pbest_all) > SORTINO(swarm_gbest, mean_daily_returns, cov_matrix_neg):
            max_index = sortino_pbest_all.index(max(sortino_pbest_all))
            swarm_gbest = initial_swarm_positions[max_index]

        avg_sortino = sum(sortino_pbest_all) / len(sortino_pbest_all)

        avg_sortino_list.append(avg_sortino)
        '''NOW FOR PRINTING THE IMPORTANT DATA FROM EACH ITERATION, WE HAVE CREATED A SEPARATE FUNCTION(PRINT_EACH_ITERATION)'''
        # portfolio_return, portfolio_standard_deviation, portfolio_sharpe,portfolio_semi_deviation, portfolio_sortino,  portfolio_weights =
        # PRINT_EACH_ITERATION(iteration, swarm_gbest, mean_daily_returns, cov_matrix,cov_matrix_neg, portfolio_return, portfolio_standard_deviation, portfolio_sharpe,portfolio_semi_deviation, portfolio_sortino,  portfolio_weights)

        weights_arr = [w / sum(swarm_gbest) for w in
                       swarm_gbest]  # Making sure all weights represent proportions that add up to 1
        weights = np.matrix(weights_arr)
        port_return = np.round(np.sum(weights * mean_daily_returns.T) * trading_days, 2) / (
            no_of_years)  # 1259 trading days over 5 year period
        port_std_dev = np.round(np.sqrt(weights * cov_matrix * weights.T) * np.sqrt(trading_days), 2) / np.sqrt(
            no_of_years)
        port_std_dev = float(port_std_dev)
        sharpe_ratio = (
                               port_return - 0.0358) / port_std_dev  # 3.58 represents annual return of risk free security - 5-year US Treasury
        port_semi_dev = np.round(np.sqrt(weights * cov_matrix_neg * weights.T) * np.sqrt(trading_days), 2) / np.sqrt(
            no_of_years)
        port_semi_dev = float(port_semi_dev)
        sortino_ratio = (port_return - 0.0358) / (port_semi_dev)
        portfolio_return.append(port_return)  # Adding portfolio return of a given PORTFOLIO  to  portfolio_return
        portfolio_standard_deviation.append(
            port_std_dev)  # Adding portfolio standard deviation of a given PORTFOLIO to a list of standard deviations to portfolio_vol
        portfolio_sharpe.append(
            sharpe_ratio)  # Adding portfolio sharpe ratio of a given PORTFOLIO  to   portfolio_sharpe
        portfolio_semi_deviation.append(
            port_semi_dev)  # Adding portfolio standard deviation of a given PORTFOLIO to a list of standard deviations to portfolio_vol
        portfolio_sortino.append(
            sortino_ratio)  # Adding portfolio sharpe ratio of a given PORTFOLIO  to   portfolio_sharpe
        portfolio_weights.append(weights_arr)  # Adding portfolio weights of a given PORTFOLIO  to   portfolio_sharpe

    # AFTER ALL THE ITERATIONS
    return portfolio_return, portfolio_standard_deviation, portfolio_sharpe, portfolio_semi_deviation, portfolio_sortino, portfolio_weights


avg_sharpe_list = []
portfolio_return = []
portfolio_standard_deviation = []
portfolio_sortino = []
portfolio_semi_deviation = []
portfolio_sharpe = []
portfolio_weights = []

portfolio_return, portfolio_standard_deviation, portfolio_sharpe, portfolio_semi_deviation, portfolio_sortino, portfolio_weights = PSO_OPTIMIZER_SORTINO(
    swarm_size, iterations)

sortino_portfolio = {'Returns': portfolio_return, 'Semi Deviation': portfolio_semi_deviation,
                     'Sortino Ratio': portfolio_sortino}

for counter, symbol in enumerate(n50.columns):
    sortino_portfolio[symbol + " Weight"] = [Weight[counter] for Weight in portfolio_weights]
sortino_pc = pd.DataFrame(sortino_portfolio)

sortino_pc.loc[:, :] *= 100
sortino_pc.loc[:, 'Sortino Ratio'] /= 100

sorted_sortino = sortino_pc.sort_values(by=['Sortino Ratio'], ascending=False)

optimal_portfolio_sortino = sorted_sortino.head(1)

optimal_portfolio_sortino.T.to_csv('/home/pn_kumar/Karthik/window-sliding/PSO(sortino)_optimal_portfolio.csv')
