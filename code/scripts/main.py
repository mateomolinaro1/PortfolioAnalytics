# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 23:09:30 2024

@author: mateo
"""
import os
os.chdir(r'C:\Users\mateo\Code\Python POO\Projet\code')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from my_package.analysis import FinancialAssetUtilities, DescriptiveStats, FactorAnalysis, CreateVisuals

# NAV
df = pd.read_excel(r'C:\Users\mateo\Code\Python POO\Projet\data\AQR Large Cap Multi-Style Fund Daily Price History.xls', engine='xlrd')
df = df.iloc[16:len(df), [1,2]]
df = df.rename(columns={'Unnamed: 1': 'date', 'Unnamed: 2':'price'})
df['date'] = pd.to_datetime(df['date'])
df_cum_ret = DescriptiveStats.calculate_cumulative_perf(df)
df_cum_ret = df_cum_ret.reset_index(drop=True)
df_ret = df_cum_ret.iloc[:,[0,2]]
df_ret = df_ret.rename(columns={'return': 'return_nav'})

CreateVisuals.plot_line_chart(df_cum_ret.iloc[:,[0,1]], x_col='date', title='NAV of AQR')
CreateVisuals.plot_line_chart(df_cum_ret.iloc[:,[0,3]], x_col='date', title='cumulative_return of AQR')
CreateVisuals.plot_line_chart(df_cum_ret.iloc[:,[0,2]], x_col='date', title='return of AQR')

# BAB factor
df_bab = pd.read_excel(r'C:\Users\mateo\Code\Python POO\Projet\data\Betting Against Beta Equity Factors Daily.xlsx')
df_bab = df_bab.iloc[18:len(df_bab), [0,24]]
df_bab = df_bab.rename(columns={'AQR Capital Management, LLC — Betting Against Beta: Equity Factors, Daily': 'date', 'Unnamed: 24':'return'})
df_bab['date'] = pd.to_datetime(df_bab['date'], format='%m/%d/%Y')
df_cum_ret_bab = DescriptiveStats.calculate_cumulative_perf(df_bab)
df_cum_ret_bab = df_cum_ret_bab.reset_index(drop=True)

CreateVisuals.plot_line_chart(df_cum_ret_bab.iloc[:,[0,2]], x_col='date', title='cumulative_return of BAB factor')
CreateVisuals.plot_line_chart(df_cum_ret_bab.iloc[:,[0,1]], x_col='date', title='return of BAB factor')

# QMJ factor
df_qmj = pd.read_excel(r'C:\Users\mateo\Code\Python POO\Projet\data\Quality Minus Junk Factors Daily.xlsx', sheet_name='QMJ Factors')
df_qmj = df_qmj.iloc[18:len(df_qmj), [0,24]]
df_qmj = df_qmj.rename(columns={'AQR Capital Management, LLC — Quality Minus Junk: Factors, Daily': 'date', 'Unnamed: 24':'return'})
df_qmj['date'] = pd.to_datetime(df_qmj['date'], format='%m/%d/%Y')
df_cum_ret_qmj = DescriptiveStats.calculate_cumulative_perf(df_qmj)
df_cum_ret_qmj = df_cum_ret_qmj.reset_index(drop=True)

CreateVisuals.plot_line_chart(df_cum_ret_qmj.iloc[:,[0,2]], x_col='date', title='cumulative_return of QMJ factor')
CreateVisuals.plot_line_chart(df_cum_ret_qmj.iloc[:,[0,1]], x_col='date', title='return of QMJ factor')

# Market factor
df_mkt = pd.read_csv(r'C:\Users\mateo\Code\Python POO\Projet\data\F-F_Research_Data_5_Factors_2x3_daily.csv', delimiter=',', header=2, parse_dates=['date'])
df_mkt = df_mkt.iloc[:,:-1]  # df_mkt = df_mkt.drop(df_mkt.columns[-1], axis=1)
df_mkt.iloc[:,1:] = df_mkt.iloc[:,1:]/100 
df_mkt.columns = df_mkt.columns.str.lower()

# Momentum factor
df_mom = pd.read_csv(r'C:\Users\mateo\Code\Python POO\Projet\data\F-F_Momentum_Factor_daily.csv', delimiter=',', header=13, parse_dates=['date'])
df_mom = df_mom.iloc[:-2, :]
df_mom.iloc[:, -1] = df_mom.iloc[:, -1].str.rstrip(' ;')
df_mom.columns = df_mom.columns.str.rstrip(' ;')
df_mom.iloc[:, 0] = pd.to_datetime(df_mom.iloc[:, 0], format='%Y%m%d')
df_mom['date'] = pd.to_datetime(df_mom['date'], errors='coerce')
df_mom['mom'] = pd.to_numeric(df_mom['mom'], errors='coerce')
df_mom['mom'] = df_mom['mom'].replace([-99.99, -999], np.nan)
df_mom.iloc[:,1:] = df_mom.iloc[:,1:]/100 

df_factors = pd.merge(df_mkt, df_mom, on='date', how='inner')
df_factors2 = pd.merge(df_bab.iloc[:,[0,1]], df_qmj.iloc[:,[0,1]], on='date', how='inner', suffixes=('_bab', '_qmj'))
df_factors = pd.merge(df_factors, df_factors2, on='date', how='inner')
df_fd = pd.merge(df_ret, df_factors, on='date', how='inner').dropna()
df_fd = df_fd.rename(columns={'return_nav': 'nav', 'return_bab': 'bab', 'return_qmj': 'qmj'})
df_fd['nav'] = pd.to_numeric(df_fd['nav'], errors='coerce')
df_fd['bab'] = pd.to_numeric(df_fd['bab'], errors='coerce')
df_fd['qmj'] = pd.to_numeric(df_fd['qmj'], errors='coerce')
df_fd = df_fd.reset_index(drop=True)
CreateVisuals.plot_line_chart(df_fd, x_col='date', title='returns')

# Df_cut for speed and visualisation
df_fd = df_fd.iloc[:,0:5]

# Factor Analysis

# Factor loadings
window = 252
x = df_fd.iloc[:,2:]
x = x.reset_index(drop=True)
y = df_fd.iloc[:,0:2]
add_const = False

df_rolling_regression = FactorAnalysis.rolling_regression(y, x, window, add_const)
CreateVisuals.plot_line_chart(df_rolling_regression, x_col='date', title='factor loadings')
# Faire tests unitaires

# Factor decomposition
df_factor_decomposition = FactorAnalysis.calculate_factor_decomposition(y, x, window, add_const)
# erreur division par 0
check = df_factor_decomposition.iloc[:,1:].sum(axis=1)
print(check)
df_plot = df_factor_decomposition.set_index(df_factor_decomposition.columns[0])
df_plot = df_plot.iloc[2600:,:]
df_plot.plot(kind='bar', stacked=True, figsize=(36, 20), grid=False)
