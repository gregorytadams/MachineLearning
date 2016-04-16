# Functions to get summaries and histograms


import pandas as pd
import matplotlib.pyplot as plt
import urllib.request as ur 
import json

COLS_FOR_SUMMARY = ['GPA', 'Age', 'Days_missed']
COLS_FOR_HIST = COLS_FOR_SUMMARY

def save_dataframe(filename, data):
    '''
    saves dataframe to csv 
    '''
    data.to_csv('output/' + filename + '.csv')

def get_summaries(data):
    '''
    Gets all the summaries 

    input: pandas dataframe 

    output: dictionary
    '''
    columns = data.columns.values
    d = {}
    for col in columns:
        d[col] = {'mean': data[col].mean(), \
        'std': data[col].std(), \
        'mode': data[col].mode(), \
        'median': data[col].median(), \
        'missing': len([i for i in data[col].isnull() if i == True])}
    return d

def descriptive_stats(data):
    return data.describe()

def save_summaries(filename, d):
    '''
    Saves summaries to new file in output folder
    '''
    with open('output/' + filename + '.txt', 'w') as f:
        json.dump(d, f)

def show_hist(data, column='', log_scale=False):
    if column == '':
        data.hist(log=log_scale)
        plt.show()
    else:
        for i in list(column):
            data[i].hist(log=log_scale)
            plt.show()

def save_hist(data, filename, column='', log_scale=False):
    if column == '': #do the graph matrix
        data.hist(log = log_scale)
        plt.savefig('output/' + filename)
        plt.close()
    else:
        if type(column) == list: #make multiple
            for i in [column]:
                data[i].hist(log=log_scale)
                plt.savefig('output/' + str(i) + '_' + filename)
                plt.close()
        else: #just one graph
            data[column].hist(log=log_scale)
            plt.savefig('output/' + column + '_' + filename)
            plt.close()

LOG_SCALE_LIST = [ 'RevolvingUtilizationOfUnsecuredLines','NumberOfTime30-59DaysPastDueNotWorse','DebtRatio', \
'MonthlyIncome','NumberOfTimes90DaysLate','NumberRealEstateLoansOrLines', \
 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents']

def make_all_indiv_hist(data, log_scale_list=LOG_SCALE_LIST, dont_plot_index = 0):
    for i in data.columns.values[dont_plot_index:]:
        if str(i) in log_scale_list:
            print("Saving {}".format(str(i)))
            save_hist(data, str(i) + 'hist.pdf', i, log_scale=True)
        else:
            print("Saving {}".format(str(i)))
            save_hist(data, str(i) + 'hist.pdf', i)

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

def add_genders(data, gender_col_name, FN_col):
    '''
    Adds in all the genders.  I ran the same loop twice because it's much, much quicker
    to gather all the names and run one query of the API rather than running multiple queries.
    '''

    for i, value in enumerate(data[gender_col_name].isnull()):
        if value == True:
            data.set_value(i, gender_col_name, genderize(data[FN_col].loc[i])) #d[data.First_name.loc[i]]['gender'])
    return data 





