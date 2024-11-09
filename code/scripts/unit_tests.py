# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 21:24:48 2024

@author: mateo
"""

import os
os.chdir(r'C:\Users\mateo\Code\Python POO\Projet\code')
import unittest

if __name__ == '__main__':
    # Chargement des tests depuis le fichier test_Analysis_FinancialAssetUtilities
    suite = unittest.defaultTestLoader.discover('.', pattern='test_Analysis_FinancialAssetUtilities.py')
    
    # Exécution des tests
    runner = unittest.TextTestRunner()
    runner.run(suite)