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
fuzzy_memberships = {}
fuzzy_memberships['t_2'] = ctrl.Antecedent(np.arange(0, max(test_data['t_2']), 1), 't_2')
fuzzy_memberships['t_1'] = ctrl.Antecedent(np.arange(0, max(test_data['t_1']), 1), 't_1')
fuzzy_memberships['t'] = ctrl.Antecedent(np.arange(0, max(test_data['t']), 1), 't')

fuzzy_memberships['d_2'] = ctrl.Antecedent(np.arange(0, max(test_data['d_2']), 1), 'd_2')
fuzzy_memberships['d_1'] = ctrl.Antecedent(np.arange(0, max(test_data['d_1']), 1), 'd_1')
fuzzy_memberships['d'] = ctrl.Antecedent(np.arange(0, max(test_data['d']), 1), 'd')

fuzzy_memberships['p'] = ctrl.Consequent(np.arange(0, max(test_data['p']), 1), 'p')

n_fuzzy_subsets = 7
for key in fuzzy_memberships:
    fuzzy_memberships[key].automf(n_fuzzy_subsets)
    fuzzy_memberships[key].view()
    
""" Fuzzy rules definitions """