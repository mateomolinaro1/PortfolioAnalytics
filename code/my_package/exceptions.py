# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 21:24:48 2024

@author: mateo
"""

class FinancialAssetError(ValueError):
    """Base class for all exceptions related to financial asset calculations.""" 
    pass


# Définir chaque exception normalement
class InvalidPriceError(FinancialAssetError):
    """
    Exception raised when the price is zero or negative.
    """
    def __init__(self, price=None, message="Price cannot be zero or negative."):
        self.price = price
        super().__init__(message)


class InvalidMethodError(FinancialAssetError):
    """
    Exception raised for invalid return calculation method.
    """
    def __init__(self, method=None):
        message = f"Invalid method '{method}'. Use 'simple' or 'log'."
        super().__init__(message)


class MissingPriceColumnError(FinancialAssetError):
    """
    Exception raised when the 'price' column is missing in the DataFrame.
    """
    def __init__(self):
        super().__init__("DataFrame must contain a 'price' column.")

class IncorretInputType(FinancialAssetError):
    """
    Exception raised when the type or dtype of an input is incorrect.
    """
    def __init__(self, input_type=None, input_dtype=None, required_type=None, required_dtype=None, d_type=False):
        self.input_type = input_type
        self.input_dtype = input_dtype
        self.required_type=required_type
        self.required_dtype = required_dtype
        self.d_type = d_type
        if not d_type:
            super().__init__(f"The type of the input: '{input_type}' is incorrect. Must be of type: '{required_type}'")
        else:
            super().__init__(f"The dtype of the input: '{input_dtype}' is incorrect. Must be of dtype: '{required_dtype}'")
            
class NotEnoughData(FinancialAssetError):
    """
    Exception raised when there is not enough data point provided.
    """
    def __init__(self, required_nb_data_point=None):
        self.required_nb_data_point = required_nb_data_point
        super().__init__(f"Not enough data point(s) provided. Minimum number of data point(s) required: {required_nb_data_point}")
        

# Assignation des exceptions comme attributs de FinancialAssetError
FinancialAssetError.InvalidPriceError = InvalidPriceError
FinancialAssetError.InvalidMethodError = InvalidMethodError
FinancialAssetError.MissingPriceColumnError = MissingPriceColumnError
FinancialAssetError.IncorrectInputType = IncorretInputType
FinancialAssetError.NotEnoughData = NotEnoughData
