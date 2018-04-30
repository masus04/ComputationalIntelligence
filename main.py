# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 12:09:26 2018

@author: 19591676
"""

import utils
from random import randint
from copy import deepcopy
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
TEMP_SUBSETS = 3
DEMAND_SUBSETS = 5
PRICE_SUBSETS = 5

def names(num_names):
    names = ('ultra low', 'very low', 'low', 'medium', 'high', 'very high', 'ultra high')
    
    if num_names == 7:
        return list(names)
    if num_names == 5:
        return list(names[1:-1])
    if num_names == 3:
        return list(names[2:-2])
    raise Exception('Illegal number of names')

temp = ctrl.Antecedent(np.arange(0, max(training_data['t_2']), 1), 'Temperature')
temp.automf(TEMP_SUBSETS, names=names(TEMP_SUBSETS))
demand = ctrl.Antecedent(np.arange(0, max(training_data['d']), 1), 'Demand')
demand.automf(DEMAND_SUBSETS, names=names(DEMAND_SUBSETS))
price = ctrl.Consequent(np.arange(0, max(training_data['p']), 1), 'Price')
price.automf(PRICE_SUBSETS, names=names(PRICE_SUBSETS))

universes = [temp, demand, price]

print('Membership functions:')
for universe in universes:
    universe
    universe.view()
    
""" Fuzzy rules definitions """
# create all possible and rules
rules = []
for tmp_name in names(TEMP_SUBSETS):
    for dem_name in names(DEMAND_SUBSETS):
        for out_name in names(PRICE_SUBSETS):
            rules.append(ctrl.Rule(temp[tmp_name] | demand[dem_name], price[out_name]))

# Todo: Feature selection by crossValidation
""" Feature selection: Evolutionary approach """
# Parameters
TOTAL_FEATURES = TEMP_SUBSETS * DEMAND_SUBSETS * PRICE_SUBSETS
ITERATIONS = TOTAL_FEATURES * 2 // 3


def evaluate_simulation(pop):
    return randint(0, 100)
    # TODO: implement
    pass


def build_simulation():
    # TODO: implement
    pass


def mutate(pop, index, direction):
    """ Mutates(flips) each gene and evaluates it. Returns a tuple (accuracy, population) """
    new_pop = deepcopy(pop)
    
    if new_pop[index] != direction:             # Mutate if not already active
        new_pop[index] = direction
        return evaluate_simulation(new_pop), new_pop
    
    return -1, None                             # Return -1 accuracy if mutation was not performed


# Generate initial population
population = [False for i in range(TOTAL_FEATURES)]
best_population = evaluate_simulation(population), population

for i in range(ITERATIONS):
    # Pick two
    for takes in range(2):
        candidate = max([mutate(population, index, True) for index in range(len(population))])
        # Save best performing
        if candidate[0] > best_population[0]:
            best_population = deepcopy(candidate)
   
    # Drop one
    candidate = max([mutate(population, index, False) for index in range(len(population))])
    # Save best performing
    if candidate[0] > best_population[0]:
        best_population = deepcopy(candidate)


# Structure: rule = ctrl.Rule(temp[0], price[4])
# tipping_ctrl = ctrl.ControlSystem(rules)
# simulation = ctrl.ControlSystemSimulation(tipping_ctrl)

print("Best Population:")
print(best_population)

