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
training_data = utils.remove_outliers_from_dataset(training_data)

plt.subplot(121)
plt.plot(training_data['p'], 'o')
plt.suptitle('Training & Test sets after outlier removal')

test_data = utils.load_test_data()
test_data = utils.remove_outliers_from_dataset(test_data)

plt.subplot(122)
plt.plot(test_data['p'], 'o')
plt.show()

""" Correlation matrix """
# TODO: This could be automated
trainingA = np.row_stack((training_data['t_2'], training_data['t_1'], training_data['t'],
                          training_data['d_2'], training_data['d_1'], training_data['d'],
                          training_data['p']))

correlationCoefficients = np.corrcoef(trainingA)
fig = plt.figure()
ax = fig.add_subplot(111)
print('Correlation Coefficients Matrix Values: \n%s' % correlationCoefficients)
ax.imshow(correlationCoefficients, interpolation='nearest')
plt.show()

# Choose t-2, d and p for data set
trainingA = np.row_stack((training_data['t_2'], training_data['d'], training_data['p']))


""" Fuzzy membership function definitions """
N_FUZZY_SUBSETS = 5

temp = ctrl.Antecedent(np.arange(0, max(training_data['t_2']), 1), 'Temperature')
demand = ctrl.Antecedent(np.arange(0, max(training_data['d']), 1), 'Demand')
price = ctrl.Consequent(np.arange(0, max(training_data['p']), 1), 'Price')

universes = [temp, demand, price]

names = ['very low', 'low', 'medium', 'high', 'very high']
print('Membership functions:')
for universe in universes:
    universe.automf(N_FUZZY_SUBSETS, names=names)
    universe.view()
    
""" Fuzzy rules definitions """
# create all possible and rules
rules = []
for in_name1 in names:
    for in_name2 in names:
        for out_name in names:
            rules.append(ctrl.Rule(temp[in_name1] | demand[in_name2], price[out_name]))

# Todo: Feature selection
#   Add 2, remove 1, keep best performing combination

""" while stop condition: """
applied_rules = []
available_rules = list(range(len(rules)))
while stop_condition:
    # pick two
    
    # drop one
    


# Structure: rule = ctrl.Rule(temp[0], price[4])
tipping_ctrl = ctrl.ControlSystem(rules)
simulation = ctrl.ControlSystemSimulation(tipping_ctrl)

def build_simulation():
    # TODO: implement
    pass

def evaluate_simulation():
    # TODO: implement
    pass
