# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 12:09:26 2018

@author: 19591676
"""

import utils
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

""" Data loading and preprocessing """

training_data = utils.load_training_data()
# plt.plot(training_data['p'], 'o')
training_data = utils.remove_outliers_from_dataset(training_data)
plt.plot(training_data['p'], 'o')
plt.title('Training set after outlier removal')

test_data = utils.load_test_data()
test_data = utils.remove_outliers_from_dataset(test_data)

""" Fuzzy membership function definitions """

N_FUZZY_SUBSETS = 7

fuzzy_memberships = {}
for key in training_data:
    if key == 'p':  # Because this is a Consequent rather than an Antecedent
        fuzzy_memberships['p'] = ctrl.Consequent(np.arange(0, max(training_data['p']), 1), 'p')
    else:
        fuzzy_memberships[key] = ctrl.Antecedent(np.arange(0, max(training_data[key]), 1), key)

    fuzzy_memberships[key].automf(N_FUZZY_SUBSETS)
    fuzzy_memberships[key].view()
    
""" Fuzzy rules definitions """