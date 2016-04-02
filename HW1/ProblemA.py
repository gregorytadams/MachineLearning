# This file contains the python code for Problem A for Homework 1


import csv
# import numpy as np
# import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt 

COLS_FOR_SUMMARY = ['GPA', 'Age', 'Days_missed']
COLS_FOR_HIST = COLS_FOR_SUMMARY

def read(filename):
    '''
    Reads in csv data as a pandas DataFrame
    '''
    return pd.read_csv(filename)

def get_summaries(data, columns = COLS_FOR_SUMMARY):
    '''
    Gets all the summaries 
    '''
    d = {}
    for col in columns:
        d[col] = {'mean': data[col].mean(), 'std': data[col].std(), 'mode': data[col].mode()[0], \
        'median': data[col].median(), 'missing': len([i for i in data[col].isnull() if i == True])}
    return d

def save_summaries(filename, d):
    with open('output/' + filename + 'txt', 'w') as f:
        for col in d:
            f.write(col + '\n')
            for i in d[col]:
                f.write(i + ': ' + str(d[col][i]) + '\n')
            f.write('\n')

def generate_histogram(data, column, filename):
    '''
    Generates a histogram of all the non-missing values
    '''
    l = []
    for i, value in enumerate(data[column].isnull()):
        if value == False:
            l.append(data[column][i])
    plt.hist(l)
    plt.title(column + ' Histogram')
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.savefig('output/' + filename + '.pdf')
    plt.close()

if __name__ == "__main__":
    data = read('mock_student_data.csv')
    for i in COLS_FOR_HIST:
        generate_histogram(data, i, str(i) + '_initial')
    save_summaries('summaries_initial', get_summaries(data))