# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 21:24:48 2024

@author: mateo
"""

import unittest
import numpy as np
import pandas as pd

from my_package.analysis import FinancialAssetUtilities, DescriptiveStats
from my_package.exceptions import FinancialAssetError

# Tests for FinancialAssetUtilities.calculate_return
class TestCalculateReturn(unittest.TestCase):
    
    # Tests for numeric inputs
    def test_numeric_simple_positive(self):
        result = FinancialAssetUtilities.calculate_return(100, 110)
        self.assertAlmostEqual(result, 0.1, delta=0.001)

    def test_numeric_simple_zero(self):
        with self.assertRaises(FinancialAssetError.InvalidPriceError):  # Fixed exception reference
            FinancialAssetUtilities.calculate_return(100, 0)
        
    def test_numeric_simple_negative(self):
        with self.assertRaises(FinancialAssetError.InvalidPriceError):  # Fixed exception reference
            FinancialAssetUtilities.calculate_return(-100, 110)
        
    def test_numeric_log_positive(self):
        result = FinancialAssetUtilities.calculate_return(100, 110, method='log')
        self.assertAlmostEqual(result, 0.0953, delta=0.001)
    
    def test_numeric_log_zero(self):
        with self.assertRaises(FinancialAssetError.InvalidPriceError):  # Fixed exception reference
            FinancialAssetUtilities.calculate_return(0, 110, method='log')
    
    def test_numeric_log_negative(self):
        with self.assertRaises(FinancialAssetError.InvalidPriceError):  # Fixed exception reference
            FinancialAssetUtilities.calculate_return(-100, 110, method='log')
        
    # Tests for DataFrame inputs
    def test_dataframe_simple_positive(self):
        dates = pd.date_range(start="2024-11-01", periods=5, freq="D")
        prices = [100, 105, 103, 108, 110]
        df = pd.DataFrame({"date": dates, "price": prices})
        
        result_df = FinancialAssetUtilities.calculate_return(df, method='simple')
        expected_returns = [np.nan, 0.05, -0.019, 0.0485, 0.0185]
        
        for i, expected in enumerate(expected_returns):
            if pd.notna(expected):
                self.assertAlmostEqual(result_df['return'].iloc[i], expected, delta=0.001)
    
    def test_dataframe_log_positive(self):
        dates = pd.date_range(start="2024-11-01", periods=5, freq="D")
        prices = [100, 105, 103, 108, 110]
        df = pd.DataFrame({"date": dates, "price": prices})
        
        result_df = FinancialAssetUtilities.calculate_return(df, method='log')
        expected_returns = [np.nan, 0.0488, -0.0192, 0.0474, 0.0183]
        
        for i, expected in enumerate(expected_returns):
            if pd.notna(expected):
                self.assertAlmostEqual(result_df['return'].iloc[i], expected, delta=0.0001)

    def test_dataframe_missing_price_column(self):
        dates = pd.date_range(start="2024-11-01", periods=5, freq="D")
        values = [100, 105, 103, 108, 110]
        df = pd.DataFrame({"date": dates, "value": values})
        
        with self.assertRaises(FinancialAssetError.MissingPriceColumnError):  # Fixed exception reference
            FinancialAssetUtilities.calculate_return(df, method='simple')
    
    def test_dataframe_negative_prices(self):
        dates = pd.date_range(start="2024-11-01", periods=5, freq="D")
        prices = [100, -105, 103, 108, 110]
        df = pd.DataFrame({"date": dates, "price": prices})
        
        with self.assertRaises(FinancialAssetError.InvalidPriceError):  # Fixed exception reference
            FinancialAssetUtilities.calculate_return(df, method='simple')
    
    def test_dataframe_single_price(self):
        dates = pd.date_range(start="2024-11-01", periods=1, freq="D")
        prices = [100]
        df = pd.DataFrame({"date": dates, "price": prices})
        
        with self.assertRaises(ValueError):
            FinancialAssetUtilities.calculate_return(df, method='simple')


# Tests for DescriptiveStats.calculate_cumulative_perf
class TestCalculateCumulativePerf(unittest.TestCase):
    
    def test_cumulative_perf_simple(self):
        # Test a basic dataframe with 'price' and 'return'
        dates = pd.date_range(start="2024-11-01", periods=5, freq="D")
        prices = [100, 105, 103, 108, 110]
        df = pd.DataFrame({"date": dates, "price": prices})
        df = FinancialAssetUtilities.calculate_return(df, method='simple')
        
        result_df = DescriptiveStats.calculate_cumulative_perf(df)
        expected_cumulative_return = [np.nan, 0.05, 0.03095, 0.08011, 0.10193]
        
        for i, expected in enumerate(expected_cumulative_return):
            if pd.notna(expected):
                self.assertAlmostEqual(result_df['cumulative_return'].iloc[i], expected, delta=0.01)
                
    

    def test_cumulative_perf_missing_price_column(self):
        dates = pd.date_range(start="2024-11-01", periods=5, freq="D")
        values = [100, 105, 103, 108, 110]
        df = pd.DataFrame({"date": dates, "value": values})
        
        with self.assertRaises(FinancialAssetError.MissingPriceColumnError):  # Fixed exception reference
            DescriptiveStats.calculate_cumulative_perf(df)
    
    def test_cumulative_perf_negative_prices(self):
        dates = pd.date_range(start="2024-11-01", periods=5, freq="D")
        prices = [100, -105, 103, 108, 110]
        df = pd.DataFrame({"date": dates, "price": prices})
        
        with self.assertRaises(FinancialAssetError.InvalidPriceError):  # Fixed exception reference
            DescriptiveStats.calculate_cumulative_perf(df)
    
    def test_cumulative_perf_single_price(self):
        dates = pd.date_range(start="2024-11-01", periods=1, freq="D")
        prices = [100]
        df = pd.DataFrame({"date": dates, "price": prices})
        
        with self.assertRaises(ValueError):
            DescriptiveStats.calculate_cumulative_perf(df)

    def test_cumulative_perf_missing_return_column(self):
        dates = pd.date_range(start="2024-11-01", periods=5, freq="D")
        prices = [100, 105, 103, 108, 110]
        df = pd.DataFrame({"date": dates, "price": prices})
        
        result_df = DescriptiveStats.calculate_cumulative_perf(df)
        self.assertIn('cumulative_return', result_df.columns)

# Tests for DescriptiveStats.calculate_annualized_perf
class TestCalculateAnnualizedPerf(unittest.TestCase):
    
    def test_geo_perf_with_prices_daily(self):
        # Test with daily prices, geometric return
        series = pd.Series([100.0, 105.0, 110.0, 120.0])
        annualized_ret = DescriptiveStats.calculate_annualized_perf(series, freq='d', prc_or_ret='prc', geo_or_arith='geo')
        expected_ret = ((120 / 100) ** (252 / 3) - 1)
        self.assertAlmostEqual(annualized_ret, expected_ret, delta=0.001)

    def test_arith_perf_with_prices_monthly(self):
        # Test with monthly prices, arithmetic return
        series = pd.Series([100.0, 105.0, 110.0, 120.0])
        annualized_ret = DescriptiveStats.calculate_annualized_perf(series, freq='m', prc_or_ret='prc', geo_or_arith='arith')
        expected_ret = ((120 / 100 - 1) * (12 / 3))
        self.assertAlmostEqual(annualized_ret, expected_ret, delta=0.001)

    def test_geo_perf_with_returns_daily(self):
        # Test with simple daily returns
        series = pd.Series([0.05, 0.047619, 0.090909])
        annualized_ret = DescriptiveStats.calculate_annualized_perf(series, freq='d', prc_or_ret='ret', ret_log=False)
        expected_ret = ((1 + series).prod() ** (252 / 3) - 1)
        self.assertAlmostEqual(annualized_ret, expected_ret, delta=0.001)

    def test_geo_perf_with_log_returns_daily(self):
        # Test with daily log-ret 
        series = pd.Series([np.log(1.05), np.log(1.047619), np.log(1.090909)])
        annualized_ret = DescriptiveStats.calculate_annualized_perf(series, freq='d', prc_or_ret='ret', ret_log=True)
        mean_log_ret = series.mean()
        expected_ret = np.exp(mean_log_ret * 252) - 1
        self.assertAlmostEqual(annualized_ret, expected_ret, delta=0.001)

    def test_error_on_nan_values(self):
        # Test with NaN values
        series = pd.Series([100.0, np.nan, 110.0])
        with self.assertRaises(ValueError):
            DescriptiveStats.calculate_annualized_perf(series, freq='d', prc_or_ret='prc')

    def test_error_on_invalid_frequency(self):
        # Test with an invalid frequency
        series = pd.Series([100.0, 105.0, 110.0])
        with self.assertRaises(ValueError):
            DescriptiveStats.calculate_annualized_perf(series, freq='h', prc_or_ret='prc')

    def test_error_on_invalid_prc_or_ret(self):
        # Test with argument 'prc_or_ret' invalid
        series = pd.Series([100.0, 105.0, 110.0])
        with self.assertRaises(ValueError):
            DescriptiveStats.calculate_annualized_perf(series, freq='d', prc_or_ret='invalid')

    def test_error_on_zero_start_price(self):
        # Test with a zero initial price
        series = pd.Series([0.0, 105.0, 110.0])
        with self.assertRaises(ZeroDivisionError):
            DescriptiveStats.calculate_annualized_perf(series, freq='d', prc_or_ret='prc')

    def test_error_on_non_numeric_series(self):
        # Test with a non numeric serie
        series = pd.Series(['100', '105', '110'])
        with self.assertRaises(FinancialAssetError.IncorrectInputType):
            DescriptiveStats.calculate_annualized_perf(series, freq='d', prc_or_ret='prc')

    def test_error_on_insufficient_data(self):
        # Test for a series with less than two data points
        series = pd.Series([100.0])
        with self.assertRaises(FinancialAssetError.NotEnoughData):
            DescriptiveStats.calculate_annualized_perf(series, freq='d', prc_or_ret='prc')
