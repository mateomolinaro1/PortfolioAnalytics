# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 20:33:47 2024

@author: mateo
"""
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from typing import Union, Optional, Tuple

from my_package.exceptions import FinancialAssetError

class DataChecks:
    """
    A class containing methods for performing validation checks on dataframes used in financial analysis.

    The purpose of this class is to ensure that the 'x' and 'y' dataframes passed to it are correctly formatted
    for further analysis. The checks include validating column structure, data types, row counts, and index consistency.
    """
    
    @staticmethod
    def check_dataframe(x: pd.DataFrame, y: pd.DataFrame) -> None:
        """
        Validates the structure and content of the 'x' and 'y' DataFrames.

        This method checks that the 'y' DataFrame has 2 columns, with the first column named 'date' and 
        containing datetime values, and the second column containing float values. It also ensures that the
        index of 'y' starts at 0. Additionally, it validates the 'x' DataFrame to ensure it has at least one column,
        two rows, and that all columns contain numeric values. The method also checks that the index of 'x' starts
        at 0 and that both DataFrames have the same number of rows.

        Parameters:
        x (pd.DataFrame): The 'x' DataFrame to validate, containing numerical data.
        y (pd.DataFrame): The 'y' DataFrame to validate, with 'date' as the first column and numerical values in the second column.

        Raises:
        AssertionError: If 'y' does not meet the specified structural or data type requirements.
        ValueError: If 'x' does not meet the specified structural or data type requirements or if the number of rows in 'x' and 'y' are mismatched.
        """
        
        # Data validation checks for 'y' DataFrame
        if y.shape[1] != 2:
            raise AssertionError("The 'y' DataFrame must have 2 columns.")
        # Check if y has at least two rows
        if y.shape[0] < 2:
            raise ValueError("DataFrame 'x' must have at least two rows.")
        if y.columns[0] != 'date':
            raise AssertionError("The name of the first column must be 'date'.")
        if not pd.api.types.is_datetime64_any_dtype(y.iloc[:, 0]):
            raise AssertionError("The first column must be of datetime type.")
        if not pd.api.types.is_float_dtype(y.iloc[:, 1]):
            raise AssertionError("The second column must be of float type.")
        if y.index[0] != 0:
            raise AssertionError("The index of 'y' must start at 0.")
        
        # Data validation checks for 'x' DataFrame
        # Check if x has at least one column
        if x.shape[1] < 1:
            raise ValueError("DataFrame 'x' must have at least one column.")
        # Check if x has at least two rows
        if x.shape[0] < 2:
            raise ValueError("DataFrame 'x' must have at least two rows.")
        # Check if all columns in x are numeric
        if not all(pd.api.types.is_numeric_dtype(x[col]) for col in x.columns):
            raise ValueError("All columns in 'x' must be numeric.")
        # Check if the index starts at 0
        if x.index[0] != 0:
            raise ValueError("Index of DataFrame 'x' must start at 0.")
        # Check if x and y have the same number of rows
        if x.shape[0] != y.shape[0]:
            raise ValueError("DataFrame 'x' and 'y' must have the same number of rows.")
            
        return
    
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
    # Need to handle the case if we have log returns
    # Need to handle the case where we want to compute cum perf for several columns in the df
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
        
        # # Input check: freq must be a string
        # if not isinstance(log_ret, bool):
        #     raise FinancialAssetError.IncorrectInputType(input_type=type(log_ret), required_type=bool)
            
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
    
    # Need to handle the case where we want to compute annualized perf on several columns of a df
    def calculate_annualized_perf(series: pd.Series, freq: str = 'd', prc_or_ret: str = 'prc', geo_or_arith: str = 'geo',
                                  ret_log: bool = False) -> float:
        """
        Calculate annualized performance based on a series of prices or returns.
    
        Parameters
        ----------
        series : pd.Series
            Series of prices or returns.
        freq : str
            Frequency of data ('d' for daily, 'w' for weekly, 'm' for monthly, 'y' for yearly).
        prc_or_ret : str
            Indicates if the input is prices ('prc') or returns ('ret').
        geo_or_arith : str
            For prices, calculate geometric ('geo') or arithmetic ('arith') returns.
        ret_log : bool
            If True, assumes the input returns are log returns.
    
        Returns
        -------
        float
            Annualized return.
        """
    
        # Input check: NaN values
        if series.isnull().any():
            raise ValueError("Series contains NaN values. Please clean the data before passing it.")
        
        # Input check: type
        if not isinstance(series, pd.Series):
            raise FinancialAssetError.IncorrectInputType(input_type=type(series), required_type=pd.core.series.Series)

        # Input check: dtype
        if not series.map(type).eq(float).all():
            raise FinancialAssetError.IncorrectInputType(input_dtype=series.dtype, required_dtype=float, d_type=True)
        
        # Input check: length of the serie must be >= 2
        if not len(series)>=2:
            raise FinancialAssetError.NotEnoughData(required_nb_data_point=2)
        
        # Input check: series.iloc[0] must be different from 0
        if series.iloc[0]==0:
            raise ZeroDivisionError
        
        # Input check: freq must be a string
        if not isinstance(freq, str):
            raise FinancialAssetError.IncorrectInputType(input_type=type(freq), required_type=str)
            
        # Input check: freq must be either 'd', 'w', 'm', or 'y'
        if freq not in ['d', 'w', 'm', 'y']:
            raise ValueError("freq must be either 'd', 'w', 'm', or 'y'")
        
        # Input check: prc_or_ret type
        if not isinstance(prc_or_ret, str):
            raise FinancialAssetError.IncorrectInputType(input_type=type(prc_or_ret), required_type=str)
        
        # Input check: prc_or_ret
        if prc_or_ret not in ['prc', 'ret']:
            raise ValueError("prc_or_ret must be either 'prc' or 'ret'")
        
        freq_map = {'d': 252, 'w': 52, 'm': 12, 'y': 1}
        freq_int = freq_map.get(freq)
        
        # Prc or ret
        if prc_or_ret == 'prc':
            size = len(series) - 1
            if geo_or_arith == 'geo':
                annualized_ret = ( series.iloc[size]/series.iloc[0] )**(freq_int/size) - 1
            elif geo_or_arith == 'arith':
                annualized_ret = ( series.iloc[size]/series.iloc[0] - 1 )*(freq_int/size)

        elif prc_or_ret == 'ret':
            size = len(series)
            if ret_log == False:
                annualized_ret = ( np.cumprod(1+series)[size-1] )**(freq_int/size) - 1
            elif ret_log == True:
                mean_log_ret = np.mean(series)
                annualized_ret = np.exp(mean_log_ret * freq_int) - 1
        
        
        return annualized_ret
    
@staticmethod
def calculate_volatility():
    
    return

class FactorAnalysis(DataChecks):
    """
    A class to create customizable functions for factor analysis.
    """
    
    @staticmethod
    def rolling_regression(y: pd.DataFrame, x:pd.DataFrame, window:int, add_const:bool):
        """
        Perform a rolling window linear regression on the provided 'x' and 'y' dataframes.
    
        This method applies a rolling regression with a specified window size and optionally adds a constant to 
        the 'x' data for each regression. The function checks the structure of the input dataframes and validates 
        the parameters. It performs an Ordinary Least Squares (OLS) regression over each window and stores the 
        regression coefficients in a new dataframe, returning the results after removing any NaN values.
    
        Parameters:
        y (pd.DataFrame): A dataframe containing the dependent variable. The first column should be 'date' (datetime), 
                          and the second column should be numeric (float).
        x (pd.DataFrame): A dataframe containing the independent variables. All columns must be numeric.
        window (int): The size of the rolling window for the regression.
        add_const (bool): Whether or not to add a constant (intercept) to the regression model.
    
        Returns:
        pd.DataFrame: A dataframe containing the regression coefficients (including 'date' and possibly a constant) 
                      for each window, with NaN values dropped.
    
        Raises:
        ValueError: If 'window' is not an integer or 'add_const' is not a boolean.
        AssertionError: If the 'x' or 'y' dataframes do not meet the required structure (e.g., correct column names, 
                        data types, and index starting at 0).
        """
        # x sans constante, col1 = dates, col2 = facteur 1. y avec date
        
        # Dataframes check
        DataChecks.check_dataframe(x, y)
        
        # Check if window is an integer
        if not isinstance(window, int):
            raise ValueError("The variable 'window' must be an integer.")
        
        # Check if add_const is a boolean
        if not isinstance(add_const, bool):
            raise ValueError("The variable 'add_const' must be a boolean.")
            
        # Storing
        names = x.columns.tolist()  # Column name extraction
        df_rolling_regression = pd.DataFrame(np.nan, index=range(x.shape[0]), columns=names)  # DataFrame creation
        df_rolling_regression.insert(0, 'date', pd.NaT)  # Insert 'date' column at the start
        
        # Constant for regression
        if add_const == True:
            x = sm.add_constant(x)
            df_rolling_regression.insert(loc=1, column='constant', value=[np.nan] * len(df_rolling_regression))

        for i in range(window, x.shape[0]):
            
            x_window = x.iloc[i - window: i, :]
            y_window = y.iloc[i - window: i, 1].to_frame()
            model = sm.OLS(y_window, x_window, missing='drop').fit()
            date = y.iloc[i, 0]
            b = pd.DataFrame(model.params).T
            b.insert(loc=0, column='date', value=date)
            b['date'] = pd.to_datetime(b['date'])
            df_rolling_regression.iloc[i] = b.iloc[0, :]
        
        return df_rolling_regression.dropna()
    
    @staticmethod
    def calculate_factor_decomposition(y: pd.DataFrame, x: pd.DataFrame, window:int, add_const:bool) -> pd.DataFrame:
        # y avec dates, x sans constante
        
        # Dataframes check
        DataChecks.check_dataframe(x, y)
        
        # Check if window is an integer
        if not isinstance(window, int):
            raise ValueError("The variable 'window' must be an integer.")
        
        # Check if add_const is a boolean
        if not isinstance(add_const, bool):
            raise ValueError("The variable 'add_const' must be a boolean.")
            
        df_rolling_regression = FactorAnalysis.rolling_regression(y, x, window, add_const)
        df_betas = df_rolling_regression.iloc[:,1:]
        X = x.iloc[window:,:]
        Y = y.iloc[window:,1:]
        Y_np = Y.values
        Y_np = Y_np.flatten() 
        dates = y.iloc[window:,0]
        dates = dates.reset_index(drop=True)
        
        # Using numpy for vectorization
        betas = df_betas.values
        X_np = X.values
        betas_times_factors = X_np*betas
        y_pred = np.sum(betas_times_factors, axis=1)
        resid = y_pred - Y_np
        betas_times_factors = np.column_stack((betas_times_factors, -1*resid))
        
        Y_np_reshaped = np.repeat(Y_np[:, np.newaxis], betas_times_factors.shape[1], axis=1)
        # factor_repartition_percentage = np.where(Y_np_reshaped != 0, betas_times_factors / Y_np_reshaped * 100, np.nan)
        
        factor_repartition_percentage = np.where((Y_np_reshaped != 0) & np.isfinite(Y_np_reshaped), 
                                         betas_times_factors / Y_np_reshaped * 100, 
                                         np.nan)
        
        # check = np.sum(factor_repartition_percentage, axis=1)
        
        # check_y = y_pred - resid
        
        names = x.columns.to_list()
        names.append('residual')
        df_factor_decomposition = pd.DataFrame(factor_repartition_percentage, columns=names)
        df_factor_decomposition.insert(0, 'date', dates)
        return df_factor_decomposition

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

