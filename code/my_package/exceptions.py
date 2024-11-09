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


# Assignation des exceptions comme attributs de FinancialAssetError
FinancialAssetError.InvalidPriceError = InvalidPriceError
FinancialAssetError.InvalidMethodError = InvalidMethodError
FinancialAssetError.MissingPriceColumnError = MissingPriceColumnError
