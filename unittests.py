# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 12:17:07 2018

@author: 19591676
"""
import unittest
import numpy as np
from utils import load_test_data, load_training_data, remove_outliers, remove_outliers_from_dataset

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
        
        data, positions = remove_outliers(arr)
        self.assertEqual(len(data), 97, msg='Outliers were not removed properly')
        outliers = np.take(data, positions)
        self.assertEqual(len(outliers), 3, msg='Outliers were not properly returned')

    def test_removeOutliersFromDataset(self):
        data = load_training_data()
        data2 = remove_outliers_from_dataset(data)
        
        self.assertNotEqual(data, data2)
        self.assertEqual(len(data['t']), 956)
        self.assertEqual(len(data2['t']), 901)
        

if __name__ == '__main__':
    unittest.main()
