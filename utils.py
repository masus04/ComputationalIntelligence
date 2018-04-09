# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 12:14:08 2018

@author: 19591676
"""
import csv
import numpy as np

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
            t_2.append(row['T(t-2)'])
            t_1.append(row['T(t-1)'])
            t.append(row['T(t)'])
            d_2.append(row['D(t-2)'])
            d_1.append(row['D(t-1)'])
            d.append(row['D(t)'])
            p.append(row['P(t+1)'])
            
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
    Q1=np.percentile(x, 25)
    Q3=np.percentile(x, 75)
    range=[Q1-1.5*(Q3-Q1),Q3+1.5*(Q3-Q1)]
    print('Q1: %s, Q3: %s, range: %s' %(Q1, Q3, range))
    position=np.concatenate((np.where(x>range[1]),np.where(x<range[0])),axis=1)
    return np.delete(x, position)