# Functions to get summaries and histograms


import pandas as pd
import matplotlib.pyplot as plt
import urllib.request as ur 
import json

COLS_FOR_SUMMARY = ['GPA', 'Age', 'Days_missed']
COLS_FOR_HIST = COLS_FOR_SUMMARY

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

def descriptive_stats(data):
    return data.describe()

def save_summaries(filename, d):
    '''
    Saves summaries to new file in output folder
    '''
    with open('output/' + filename + '.txt', 'w') as f:
        json.dump(d, f)

# def generate_histogram(data, column, filename):
#     '''
#     Generates a histogram of all the non-missing values
#     '''
#     l = []
#     for i, value in enumerate(data[column].isnull()):
#         if value == False:
#             l.append(data[column][i])
#     plt.hist(l)
#     plt.title(column + ' Histogram')
#     plt.xlabel(column)
#     plt.ylabel("Frequency")
#     # plt.savefig('output/' + filename + '.pdf')
#     # plt.close()

def make_hist(data):
    data.hist()
    plt.show()

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
    # list_of_names = []
    # for i, value in enumerate(data['Gender'].isnull()):
    #     if value == True:
    #         list_of_names.append(data.First_name.loc[i])
    # print(list_of_names)
    # d = get_genders(list_of_names)
    # print(d)
    for i, value in enumerate(data[gender_col_name].isnull()):
        if value == True:
            data.set_value(i, gender_col_name, genderize(data[FN_col].loc[i])) #d[data.First_name.loc[i]]['gender'])
    return data 

# def get_genders(list_of_names):
    '''
    Grabs the names all at once instead of one at a time.  Turns out that the api limits 
    you to 10 at a time, though, so it doesn't matter that much.  Fuck me, right?
    '''
#     url_ending = ''
#     for i, name in enumerate(list_of_names):
#         url_ending += 'name[{}]={}&'.format(i, name)
#     print("HERE IT IS: " + url_ending)
#     with ur.urlopen('https://api.genderize.io/?' + url_ending[:len(url_ending)-1]) as q:
#         l = json.loads(q.read().decode('utf-8'))
#     print("LEN L" + str(len(l)))
#     d = {}
#     for sub_dict in l:
#         d[sub_dict['name']] = sub_dict
#     return d



