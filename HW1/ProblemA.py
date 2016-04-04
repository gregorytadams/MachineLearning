# This file contains the python code for Problem A for Homework 1

import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import choice
import urllib.request as ur 
import json

COLS_FOR_SUMMARY = ['GPA', 'Age', 'Days_missed']
COLS_FOR_HIST = COLS_FOR_SUMMARY

def go():
    '''
    Master function, called in __main__ block.  Calls everything.  
    '''
    data = read('mock_student_data.csv')
    for i in COLS_FOR_HIST:
        generate_histogram(data, i, str(i) + '_initial')
    summaries = get_summaries(data)
    save_summaries('summaries_initial', summaries)
    data = add_genders(data)
    save_dataframe('data_with_genders', data) 
    save_dataframe('data_with_mean', update_with_mean(data.copy(), summaries)) # modifies in place if it's not a copy
    save_dataframe('data_with_cc_mean', update_with_cc_means(data.copy()))
    save_dataframe('data_with_variance', update_with_variance(data.copy())) 

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
            data.set_value(i, 'Gender', genderize(data['First_name'].loc[i]))
            print("filled in {}".format(data['First_name'].loc[i])) # watch it work!
    return data 

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
                # print('replaced col {} index {} with {}'.format(col, i, data[col][i])) # watch it work!
    return data

if __name__ == "__main__":
    go()


