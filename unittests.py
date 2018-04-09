# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 12:17:07 2018

@author: 19591676
"""
import unittest
import numpy as np
from utils import remove_outliers, load_test_data, load_training_data

class UnitTests(unittest.TestCase):

    def test_loadData(self):
        test_data = load_test_data()
        training_data = load_training_data()
        
        self.assertEqual(len(training_data), 7, msg='Column number incorrect')
        self.assertEqual(len(test_data), 7, msg='Column number incorrect')
        
        self.assertEqual(len(training_data['t']), 956, msg='Row number incorrect')
        self.assertEqual(len(test_data['p']), 506, msg='Row number incorrect')
    
    def test_removeOutliers(self):
        arr = np.random.random(100)
        arr[10]=3
        arr[50]=4
        arr[80]=5
        
        removed = remove_outliers(arr)
        self.assertEqual(len(removed), 97, msg='Outliers were not removed properly')


if __name__ == '__main__':
    unittest.main()
