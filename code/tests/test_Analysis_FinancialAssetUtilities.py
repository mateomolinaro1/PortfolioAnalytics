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
