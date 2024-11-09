# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 20:33:47 2024

@author: mateo
"""
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Optional, Tuple

from my_package.exceptions import FinancialAssetError


class FinancialAssetUtilities:
    """
    A utility class for performing financial calculations on price data.

    This class provides methods for calculating returns based on either two price values
    or a list of prices contained in a pandas DataFrame. It supports both simple and
    logarithmic returns, with appropriate error handling for invalid inputs such as zero
    or negative prices, and missing or incorrect columns in DataFrames.

    Methods
    -------
    calculate_return(*args, method='simple')
        Calculates the return based on two prices or a series of prices in a DataFrame.
        Supports both simple and logarithmic return calculations.
    """
    
    @staticmethod
    def calculate_return(*args: Union[int, float, pd.DataFrame], method: str ='simple') -> Union[float, pd.DataFrame]:
        
        """
        Calculate the return based on provided price values or a series of prices.
    
        Parameters
        ----------
        *args : numeric or pd.DataFrame
            - If two numeric values are provided, computes either 'simple' or 'log'
              return between them.
            - If a pd.DataFrame is provided, computes consecutive 'simple' or 'log' 
              returns over the list. The DataFrame must contain at least two columns,
              with one named 'price' containing only positive prices.
        
        method : str, optional
            Specifies the type of return to calculate: 'simple' or 'log'. Default is 'simple'.
    
        Returns
        -------
        float or pd.DataFrame
            - If two numeric values are provided, returns the calculated return as a float.
            - If a DataFrame is provided, returns the original DataFrame with an additional 
              'Return' column containing the calculated returns for each consecutive price pair.
    
        Raises
        ------
        FinancialAssetError
            If input values are invalid, such as:
            - Zero or negative prices (InvalidPriceError).
            - DataFrame not containing a 'price' column (MissingPriceColumnError).
            - Invalid method specified (InvalidMethodError).
        """
    
        if len(args) == 2 and all(isinstance(arg, (int, float)) for arg in args):
            
            initial_value, final_value = args
            if initial_value <= 0 or final_value <= 0:
                raise FinancialAssetError.InvalidPriceError(price=initial_value if initial_value <= 0 else final_value)
                
            if method == 'simple':
                return (final_value / initial_value - 1)
            elif method == 'log':
                return math.log(final_value / initial_value)
            else:
                raise FinancialAssetError.InvalidMethodError(method)
                
        elif len(args) == 1 and isinstance(args[0], pd.DataFrame):
            
            df = args[0]
            
            # Column checks
            if df.shape[1] < 2:
                raise ValueError("DataFrame must have at least two columns, including a 'price' column.")
            if 'price' not in df.columns:
                raise FinancialAssetError.MissingPriceColumnError()
            
            prices = df['price']
            # Check if there is at leat 2 prices
            if len(prices) < 2:
                raise ValueError("price list must contain at least 2 prices.")
            
            # Check if prices are positive
            if (prices <= 0).any():
                raise FinancialAssetError.InvalidPriceError(price=prices[prices <= 0].iloc[0])
                
            # Calculate returns using shift() for consecutive price pairs
            if method == 'simple':
                df['return'] = (prices / prices.shift(1) - 1)
            elif method == 'log':
                df['return'] = (prices / prices.shift(1)).apply(math.log)
            else:
                raise FinancialAssetError.InvalidMethodError(method)
            
            return df
        
        else:
            raise ValueError("Invalid arguments provided.")
           
           
class DescriptiveStats:
    @staticmethod
    def calculate_cumulative_perf(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the cumulative performance based on returns or prices provided.
    
        Parameters
        ----------
        df : pd.DataFrame
            A DataFrame with at least two columns: 'date', 'price', and optionally 'return'. 
            If 'return' column is not provided, the function will compute it based on 'price'.
    
        Returns
        -------
        pd.DataFrame
            The original DataFrame with an additional 'cumulative_return' column.
    
        Raises
        ------
        FinancialAssetError
            - MissingPriceColumnError if 'price' column is missing and 'return' is also missing.
            - InvalidPriceError if price values are invalid (zero or negative).
            - ValueError if DataFrame does not contain required columns.
        """
        
        # Normalize column names to lower case to avoid case-sensitive issues
        df.columns = df.columns.str.lower()
        
        # Column checks
        if not isinstance(df, pd.DataFrame):  # Even if input is typed, additional guarantee
            raise ValueError("Input must be a pandas DataFrame.")
        
        if df.shape[1] < 2:
            raise ValueError("DataFrame must have at least two columns, including 'price' or 'return'.")
        
        # If 'price' column is missing, check if 'return' exists
        if 'price' not in df.columns:
            if 'return' not in df.columns:
                raise FinancialAssetError.MissingPriceColumnError()  # Missing both 'price' and 'return'
            
            # If 'return' exists, calculate cumulative return
            cumulative_return = (1 + df['return']).cumprod() - 1
            df['cumulative_return'] = cumulative_return
            return df
        
        # If 'price' is present, ensure it's valid
        prices = df['price']
        if len(prices) < 2:
            raise ValueError("Price list must contain at least 2 prices.")
        
        # Check if prices are positive
        if (prices <= 0).any():
            invalid_price = prices[prices <= 0].iloc[0]  # Identify the first invalid price
            raise FinancialAssetError.InvalidPriceError(price=invalid_price)
        
        # Case 1: if 'return' column is not provided, calculate it
        if 'return' not in df.columns:
            df = FinancialAssetUtilities.calculate_return(df)
            
        # Ensure 'return' column exists after calculation
        if 'return' in df.columns:
            cumulative_return = (1 + df['return']).cumprod() - 1
            df['cumulative_return'] = cumulative_return
        
        return df


class FactorAnalysis:
    """
    A class to create customizable functions for factor analysis.
    """
    
    @staticmethod
    def calculate_factor_decomposition(df: pd.DataFrame) -> pd.DataFrame:
        
        
        
        return

class CreateVisuals:
    """
    A class to create customizable visualizations, including line charts.
    This class can be extended with additional methods for different types of plots.
    """

    @staticmethod
    def plot_line_chart(
        df: pd.DataFrame,  # The data to plot
        x_col: str,  # The name of the column for the X-axis (should be 'date')
        color: Optional[str] = None,  # A single color for the lines (default is None)
        palette: str = 'pastel',  # Color palette to use (default is 'pastel')
        grid_bool: bool = False,  # Whether to display the grid (default is False)
        show_ci: bool = False,  # Whether to display the confidence interval (default is True)
        style: str = 'whitegrid',  # Style of the plot (default is 'whitegrid')
        context: str = 'talk',  # Context of the plot (default is 'talk')
        figsize: Tuple[int, int] = (12, 6),  # Size of the figure (default is (12, 6))
        linestyle: str = '-',  # Line style (default is '-')
        marker: str = 'o',  # Marker style (default is 'o')
        linewidth: float = 2.5,  # Line width (default is 2.5)
        grid_color: str = 'grey',  # Color of the grid (default is 'grey')
        grid_linestyle: str = '--',  # Line style of the grid (default is '--')
        grid_linewidth: float = 0.5,  # Line width of the grid (default is 0.5)
        title: str = 'Customized Line Chart',  # Title of the plot (default is 'Customized Line Chart')
        title_fontsize: int = 14,  # Font size of the title (default is 14)
        show_marker: bool = False,  # Whether to display markers (default is False)
        legend_loc: str = 'upper right'  # Location of the legend (default is 'upper right')
    ) -> None:  # No return value as it shows the plot
        """
        Creates a customizable line chart with multiple series based on columns (excluding 'date').
        
        Parameters:
            df (pd.DataFrame): The data to plot.
            x_col (str): The name of the column for the X-axis (should be 'date').
            color (Optional[str], optional): A single color for the lines (default is None).
            palette (str, optional): Color palette to use (default is 'pastel').
            grid_bool (bool, optional): Whether to display the grid (default is False).
            show_ci (bool, optional): Whether to display the confidence interval (default is True).
            style (str, optional): Style of the plot (default is 'whitegrid').
            context (str, optional): Context of the plot (default is 'talk').
            figsize (Tuple[int, int], optional): Size of the figure (default is (12, 6)).
            linestyle (str, optional): Line style (default is '-.').
            marker (str, optional): Marker style (default is 'o').
            linewidth (float, optional): Line width (default is 2.5).
            grid_color (str, optional): Color of the grid (default is 'grey').
            grid_linestyle (str, optional): Line style of the grid (default is '--').
            grid_linewidth (float, optional): Line width of the grid (default is 0.5).
            title (str, optional): Title of the plot (default is 'Customized Line Chart').
            title_fontsize (int, optional): Font size of the title (default is 14).
            show_marker (bool, optional): Whether to display markers (default is False).
            legend_loc (str, optional): Location of the legend (default is 'upper right').
        """
        # Check if the 'x_col' is in datetime format
        if df[x_col].dtype != 'datetime64[ns]':
            raise ValueError(f"The column '{x_col}' must be in datetime format.")
        
        # Check if 'date' column is in the first position
        if df.columns[0] != x_col:
            raise ValueError("The 'date' column must be in the first position.")

        # Set style, context, and palette
        sns.set_style(style)  # Options: 'white', 'dark', 'whitegrid', 'darkgrid', 'ticks'
        sns.set_context(context)  # Options: 'talk', 'paper', 'notebook', 'poster'
        sns.set_palette(palette)  # Uses the specified palette
    
        # Create the plot
        plt.figure(figsize=figsize)  # Figure size
    
        # If `show_ci` is False, we pass `errorbar=None` to avoid the confidence interval
        errorbar_value = None if not show_ci else ('ci', 95)
    
        # Set marker based on user input
        plot_marker = marker if show_marker else None
    
        # Loop through columns (except 'date') and plot them
        for col in df.columns[1:]:
            sns.lineplot(x=x_col, y=col, data=df, color=color, 
                         linestyle=linestyle, marker=plot_marker, 
                         linewidth=linewidth, errorbar=errorbar_value, label=col)
    
        # Display grid if specified
        if grid_bool:
            plt.grid(color=grid_color, linestyle=grid_linestyle, 
                     linewidth=grid_linewidth)
        else:
            plt.grid(False)
    
        # Add title and labels
        plt.title(title, fontsize=title_fontsize)
        plt.xlabel(x_col)
        plt.ylabel('Values')
    
        # Add legend
        plt.legend(loc=legend_loc)
    
        # Show the plot
        plt.show()


