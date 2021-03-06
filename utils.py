# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 12:14:08 2018

@author: 19591676
"""
import csv
import numpy as np
from copy import deepcopy

def load_data(path):
    """
    Returns a dictionary containing the following values: t_2, t_1, t, d_2, d_1, d, p
    """
    t_2, t_1, t = [], [], []
    d_2, d_1, d = [], [], []
    p = []
    
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            t_2.append(float(row['T(t-2)']))
            t_1.append(float(row['T(t-1)']))
            t.append(float(row['T(t)']))
            d_2.append(float(row['D(t-2)']))
            d_1.append(float(row['D(t-1)']))
            d.append(float(row['D(t)']))
            p.append(float(row['P(t+1)']))
            
    return {'t_2': t_2, 't_1': t_1, 't': t, 'd_2': d_2, 'd_1': d_1, 'd': d, 'p': p}        

def load_training_data():
    """
    Returns a dictionary containing the following values: t_2, t_1, t, d_2, d_1, d, p
    """
    return load_data('./data/2018_CI_Assignment_Training_Data.csv')

def load_test_data():
    """
    Returns a dictionary containing the following values: t_2, t_1, t, d_2, d_1, d, p
    """
    return load_data('./data/2018_CI_Assignment_Testing_Data.csv')

def remove_outliers(x):
    x = deepcopy(x)
    Q1 = np.percentile(x, 25)
    Q3 = np.percentile(x, 75)
    range = [Q1-1.5*(Q3-Q1),Q3+1.5*(Q3-Q1)]
    positions = np.concatenate((np.where(x>range[1]),np.where(x<range[0])),axis=1)[0]
    outliers = np.take(x, positions)
    if len(outliers) > 0:
        print('Removed outliers: \n%s' % outliers)
    return np.delete(x, positions), positions

def remove_outliers_from_dataset(data):
    data = deepcopy(data)
    
    positions = set()
    for key in data:
        d, pos = remove_outliers(data[key])
        positions.update(pos)
    
    for key in data:
        data[key] = np.delete(data[key], list(positions))
    
    return data




