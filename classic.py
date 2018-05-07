# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 12:09:26 2018

@author: 19591676
"""

import sys
import utils
import numpy as np
from skfuzzy import control as ctrl, trimf, interp_membership
import matplotlib.pyplot as plt

# Print helper method
def print_inplace(text, percentage, time_taken=None, comment=""):
        percentage = int(percentage)
        length_factor = 5
        progress_bar = int(round(percentage/length_factor)) * "*" + (round((100-percentage)/length_factor)) * "."
        progress_bar = progress_bar[:round(len(progress_bar)/2)] + "|" + str(int(percentage)) + "%|" + progress_bar[round(len(progress_bar)/2):]
        sys.stdout.write("\r%s |%s|" % (text, progress_bar) + (" Time: %s" % str(time_taken).split(".")[0] if time_taken else "") + comment)
        sys.stdout.flush()

        if percentage == 100:
            print()

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

fig = plt.figure()
fig.set_figwidth(15)
ax = fig.add_subplot(131)
ax.hist(training_data['t_1'], 20)
ax.set_title('t-1 distribution')

ax = fig.add_subplot(132)
ax.hist(training_data['d'], 20)
ax.set_title('d distribution')

ax = fig.add_subplot(133)
ax.hist(training_data['p'], 20)
ax.set_title('p distribution')

temp = ctrl.Antecedent(np.arange(0, max(training_data['t_2'])*1.2, 1), 'temp')
temp_low = trimf(temp.universe, [21, 26, 29])
temp['low'] = temp_low
temp_high = trimf(temp.universe, [27, 30, 32])
temp['high'] = temp_high
temp.view()

demand = ctrl.Antecedent(np.arange(0, max(training_data['d'])*1.2, 1), 'demand')
demand_low = trimf(demand.universe, [3900, 4400, 4900])
demand['low'] = demand_low
demand_mid = trimf(demand.universe, [4400, 5300, 6000])
demand['medium'] = demand_mid
demand_high = trimf(demand.universe, [5500, 6300, 6700])
demand['high'] = demand_high
demand.view()

price = ctrl.Consequent(np.arange(0, max(training_data['p'])*1.2, 1), 'price')
price_low = trimf(price.universe, [10, 18, 28])
price['low'] = price_low
price_medium = trimf(price.universe, [15, 24, 28])
price['medium'] = price_medium
price_high = trimf(price.universe, [20, 30, 52])
price['high'] = price_high
price.view()

""" Statistical Analysis """
grid = [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]]

for t, d, p in zip(training_data['t_2'], training_data['d'], training_data['p']):
    t = max((interp_membership(temp.universe, temp_low, t), 0), (interp_membership(temp.universe, temp_high, t), 1))[1]
    d = max((interp_membership(demand.universe, demand_low, d), 0), (interp_membership(demand.universe, demand_mid, d), 1), (interp_membership(demand.universe, demand_high, d), 2))[1]
    p = max((interp_membership(price.universe, price_low, p), 0), (interp_membership(price.universe, price_medium, p), 1), (interp_membership(price.universe, price_high, p), 2))[1]
    grid[p][t][d] += 1

print(grid)
fig = plt.figure()
fig.set_figwidth(15)
ax = fig.add_subplot(131)
ax.imshow(grid[0], interpolation='nearest')
ax = fig.add_subplot(132)
ax.imshow(grid[1], interpolation='nearest')
ax = fig.add_subplot(133)
ax.imshow(grid[2], interpolation='nearest')
plt.show()

""" Fuzzy rules definitions """
# High support rules
rules = []
rules.append(ctrl.Rule(temp['low'] | demand['low'], price['low']))
rules.append(ctrl.Rule(temp['low'] | demand['medium'], price['medium']))
rules.append(ctrl.Rule(temp['high'] | demand['high'], price['high']))

# Medium support rules
rules.append(ctrl.Rule(temp['low'] | demand['high'], price['medium']))
rules.append(ctrl.Rule(temp['low'] | demand['medium'], price['high']))

# Low support rules


""" Build and evaluate simulation """
control_system = ctrl.ControlSystem(rules)
simulation = ctrl.ControlSystemSimulation(control_system)

error = 0
for t, d, p in zip(test_data['t_2'], test_data['d'], test_data['p']):
    simulation.input["temp"] = t
    simulation.input["demand"] = d
    simulation.compute()
    error += abs(simulation.output['price'] + p)/p

error /= len(test_data['p'])

print('Average error: %s' % error)








