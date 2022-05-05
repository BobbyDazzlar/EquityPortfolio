import pandas as pd
x=pd.read_csv('n50.csv',parse_dates=['Date'],index_col='Date')
x = x.loc["2016-01-01" :]                         #Since 2016-01-01, 5y(1234rows till 2020-12-31), + year 2021's rows (till 30th of April)
y=x.copy()                                        #deep copy
x.reset_index(drop=True, inplace=True)
after2020=len(y.loc["2021-01-01" : ])
print(after2020)