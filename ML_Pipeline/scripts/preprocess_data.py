#Functions to find/replace missing data in various ways


import pandas as pd
from numpy.random import choice

COLS_FOR_SUMMARY = ['GPA', 'Age', 'Days_missed']

def save_dataframe(filename, data):
    '''
    saves dataframe to csv 
    '''
    data.to_csv('output/' + filename + '.csv')

def update_with_mean(data, summary_dict, columns= COLS_FOR_SUMMARY):
    '''
    Updates the dataframe with the overall mean as instructed in 3A
    '''
    for col in columns:
        filler = summary_dict[col]['mean']
        data[col] = data[col].fillna(filler)
    return data

def update_with_cc_means(data, columns = COLS_FOR_SUMMARY, groupvar = 'Graduated'):
    '''
    Updates the dataframe with the class-conditional mean as instructed in 3B
    '''
    means = data.groupby(groupvar).mean() # syntax like a dictionary: means[att][y/n]
    for col in columns:
        for i, b in enumerate(data[col].isnull()):
            if b:
                att = data[groupvar][i]
                val = means[col][att]
                data.set_value(i, col, val)
    return data

def update_with_variance(data, columns = COLS_FOR_SUMMARY):
    '''
    Because all the data are whole numbers, I find the probability of a randomly chosen person to be
    a given value, and assign missing values based on that.

    Implemented as instructed in 3C.

    '''
    vc = {}
    for col in columns:
        vc[col] = data[col].value_counts()
    for col in columns:
        for i, b in enumerate(data[col].isnull()): 
            if b:
                vals = vc[col].axes[0]
                probs = list(map(lambda x: x/sum(vc[col]), vc[col]))
                data.set_value(i, col, int(choice(vals, 1, probs)))  #choice is from numpy.random
    return data

def save_dataframe(filename, data):
    '''
    saves dataframe to csv 
    '''
    data.to_csv('output/' + filename + '.csv')