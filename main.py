# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 12:09:26 2018

@author: 19591676
"""

import utils

training_data = utils.load_training_data()
training_data = utils.remove_outliers(training_data)

test_data = utils.load_test_data()
test_data = utils.remove_outliers(test_data)


