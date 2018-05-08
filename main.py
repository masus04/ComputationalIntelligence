# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 12:09:26 2018

@author: 19591676
"""

import sys
import utils
from datetime import datetime
from random import choice
from copy import deepcopy
import numpy as np
from skfuzzy import control as ctrl
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

temp = ctrl.Antecedent(np.arange(0, max(training_data['t_2']), 1), 'temp')
temp.automf(TEMP_SUBSETS, names=names(TEMP_SUBSETS))
demand = ctrl.Antecedent(np.arange(0, max(training_data['d']), 1), 'demand')
demand.automf(DEMAND_SUBSETS, names=names(DEMAND_SUBSETS))
price = ctrl.Consequent(np.arange(0, max(training_data['p']), 1), 'price')
price.automf(PRICE_SUBSETS, names=names(PRICE_SUBSETS))

universes = [temp, demand, price]

print('Membership functions:')
for universe in universes:
    universe
    universe.view()
    
""" Fuzzy rules definitions """
# create all possible and rules
allRules = []
for tmp_name in names(TEMP_SUBSETS):
    for dem_name in names(DEMAND_SUBSETS):
        for out_name in names(PRICE_SUBSETS):
            allRules.append(ctrl.Rule(temp[tmp_name] | demand[dem_name], price[out_name]))

# Todo: Feature selection by crossValidation
""" Feature selection: Evolutionary approach """
# Parameters
start = datetime.now()

TOTAL_FEATURES = TEMP_SUBSETS * DEMAND_SUBSETS * PRICE_SUBSETS
ITERATIONS = TOTAL_FEATURES * 2 // 3


def build_simulation(rule_indices):
    rules = [rule for rule, selected in zip(allRules, rule_indices) if selected]
    control_system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(control_system)

    
def evaluate_simulation(simulation, evaluation_dataset):
    error = 0
    for t2, d, p in zip(evaluation_dataset["t_2"], evaluation_dataset["d"], evaluation_dataset["p"]):
        simulation.input["temp"] = t2
        simulation.input["demand"] = d
        simulation.compute()
        error += abs(simulation.output['price'] + p)/p
        
    return -error/len(test_data['p'])


def evaluate_population(pop, evaluation_dataset):
    simulation = build_simulation(pop)
    return evaluate_simulation(simulation, evaluation_dataset)
    

def mutate(pop, index, direction):
    """ Mutates(flips) each gene and evaluates it. Returns a tuple (accuracy, population) """
    # print_inplace('Mutation %s/%s' % (index, len(pop)), index/len(pop)*100, time_taken=datetime.now()-start)
    new_pop = list(deepcopy(pop))
    
    if new_pop[index] != direction:             # Mutate if not already active
        new_pop[index] = direction
        return evaluate_population(new_pop, training_data), new_pop
    
    return -100000, new_pop                     # Return -1 accuracy if mutation was not performed


# Generate initial population
population = [False for i in range(TOTAL_FEATURES)]
population[0], population[-1] = True, True
best_population = evaluate_population(population, training_data), deepcopy(population)

for i in range(ITERATIONS):
    print_inplace('Feature selection iteration %s/%s' % (i+1, ITERATIONS), i/ITERATIONS*100, time_taken=datetime.now()-start, comment=' | Best Population: Average Error: %s, population: %s' % (abs(best_population[0]), best_population[1]))
    
    # Add Rules
    for takes in range(2):
        score, population = max([mutate(population, index, True) for index in range(len(population))])
        # Save best performing
        if score > best_population[0]:
            best_population = score, deepcopy(population)
   
    # Remove Rules
    for takes in range(1):
        score, population = max([mutate(population, index, False) for index in range(len(population))])
        # Save best performing
        if score > best_population[0]:
            best_population = score, deepcopy(population)

print("Best Population:")
print(best_population[1])
print('Average error on training set: %s' % best_population[0])

# Evaluate best population on testset
print('Average error on training set: %s' % evaluate_population(best_population[1], test_data))