# This file contains the python code for Problem A for Homework 1

# import csv
# import numpy as np
# import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request as ur 
import json

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

    input: pandas dataframe 

    output: dictionary
    '''
    d = {}
    for col in columns:
        d[col] = {'mean': data[col].mean(), 'std': data[col].std(), 'mode': data[col].mode()[0], \
        'median': data[col].median(), 'missing': len([i for i in data[col].isnull() if i == True])}
    return d

def save_summaries(filename, d):
    '''
    Saves summaries to new file
    '''
    with open('output/' + filename + '.txt', 'w') as f:
        json.dump(d, f)

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

def genderize(name):
    '''
    Gives gender of a name.

    input: 
    a first name as a string

    output:
    'male' or 'female'
    '''
    with ur.urlopen('https://api.genderize.io/?name=' + name.lower()) as q:
        rv = json.loads(q.read().decode('utf-8'))['gender']
    return rv

def add_genders(data):
    '''
    Adds in all the genders.  (Kinda slow though.)
    Brings up warning.
    '''
    for i, value in enumerate(data['Gender'].isnull()):
        if value == True:
            data['Gender'].loc[i] = genderize(data['First_name'].loc[i])
            # print("filled in {}".format(data['First_name'].loc[i]))
    return data 

def save_dataframe(data, filename):
    '''
    saves dataframe to csv 
    '''
    with open('output/' + filename + '.txt', 'w') as f:
        data.to_csv(f)

def update_with_mean(data, summary_dict, columns= COLS_FOR_SUMMARY):
    for col in columns:
        filler = summary_dict[col]['mean']
        data[col] = data[col].fillna(filler)
    return data

# def get_cc_means(data, columns=COLS_FOR_SUMMARY):
#     cc_means = {}
#     yes_means = data[data.Graduated == 'Yes'].mean() # returns class conditional means for all relevant columns
#     no_means =  data[data.Graduated == 'No'].mean()
#     for col in columns: 
#         cc_means[col] = {'Yes': yes_means[col], 'No': no_means[col]}
#         # cc_means[col][y/n] gives class-conditional mean
#     return cc_means

def update_with_cc_means(data, columns = COLS_FOR_SUMMARY):
    means = data.groupby('Graduated').mean() # syntax like a dictionary 
    for col in columns:
        data.apply(lambda x: means[col.Graduated] if pd.isnull(x[col]) else x[col])
    return data


    #     for i in data[data[col].isnull()].index:



    # for i in df[df.Vals.isnull()].index:
    #     df.loc[i, 'Vals'] = means[df.loc[i].Cat]



    # for i in range(len(data.index)):
    #     for col in cc_means:
    #         if data.ix[i][col].isnull():
    #             data.ix[i][col] = ccmeans[col][data.ix[i]['Graduated']]




    # data[col].groupby(data['Graduated']).mean()['Yes']
    # for col in cc_means:

    # pass
    


# if __name__ == "__main__":
#     data = read('mock_student_data.csv')
#     for i in COLS_FOR_HIST:
#         generate_histogram(data, i, str(i) + '_initial')
#     save_summaries('summaries_initial', get_summaries(data))
#     data = add_genders(data)
#     save_dataframe('data_with_genders', data)
#     save_dataframe('data_with_mean', update_with_mean(data))
#     save_dataframe('data_with_cc_mean', update_with_cc_mean(data))