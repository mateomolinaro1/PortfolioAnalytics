# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 23:09:30 2024

@author: mateo
"""
import os
os.chdir(r'C:\Users\mateo\Code\Python POO\Projet\code')
import pandas as pd

from my_package.analysis import FinancialAssetUtilities, DescriptiveStats, CreateVisuals

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

df_factors = pd.merge(df_bab.iloc[:,[0,1]], df_qmj.iloc[:,[0,1]], on='date', how='inner', suffixes=('_bab', '_qmj'))
df_fd = pd.merge(df_ret, df_factors, on='date', how='inner').dropna()
CreateVisuals.plot_line_chart(df_fd, x_col='date', title='returns')

# Return decompisition into factors