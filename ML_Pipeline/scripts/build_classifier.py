
# from itertools import combinations
from __future__ import division
import pandas as pd
import numpy as np
from preprocess_data import update_with_cc_means
from generate_features import cat_to_binary
from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition, svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, recall_score, auc, f1_score
import random
import pylab as pl
import matplotlib.pyplot as plt
from scipy import optimize
import time
import csv
print("imports done")

def format_for_models(data, response, predictor):
    y = data[response].values
    if type(predictor) == str:
        x = data[predictor].values
        x = x.reshape(len(x), 1)
        return x, y
    else:
        my_array = data[predictor.pop(0)].values
        my_array = my_array.reshape(len(my_array), 1)
        for i in predictor:
            a = data[i].values
            a = a.reshape(len(a), 1)
            my_array = np.concatenate((my_array, a), axis=1)
        return my_array, y




def define_clfs_params():

    clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        'NB': GaussianNB(),
        'DT': DecisionTreeClassifier(),
        'SGD': SGDClassifier(loss="hinge", penalty="l2"),
        'KNN': KNeighborsClassifier(n_neighbors=3) 
            }

    grid = { 
    'RF':{'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
    'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [1,10,100,1000,10000], 'criterion' : ['gini', 'entropy'] ,'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'GB': {'n_estimators': [1,10,100,1000,10000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
           }

    return clfs, grid

def magic_loop(models_to_run, clfs, params, X, y, k):
    ''' 
    X and y need to be formatted
    '''
    model_list = [['Models', 'Parameters', 'Split', 'Accuracy', 'Recall', 'AUC', 'F1', 'precision at' + str(k)]]
    for n in range(1, 2):
        print("split: {}".format(n))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        for index,clf in enumerate([clfs[x] for x in models_to_run]):
            print(models_to_run[index])
            parameter_values = params[models_to_run[index]]
            for p in ParameterGrid(parameter_values):
                try:
                    d = {}
                    print("parameters {}".format(p))
                    clf.set_params(**p)
                    clf.fit(X_train, y_train)
                    y_pred_probs = clf.predict_proba(X_test)[:,1]
                    y_pred = clf.predict(X_test)
                    d['accuracy'] = clf.score(X_test, y_test)
                    d['recall'] = recall_score(y_test, y_pred)
                    d['AUC'] = roc_auc_score(y_test, y_pred_probs)
                    d['F1'] = f1_score(y_test, y_pred)
                    d['precision at' + str(k)] = precision_at_k(y_test,y_pred_probs,k)
                    print(d)
                    # plot_precision_recall_n(y_test, y_pred_probs, clf)
                    model_list.append([models_to_run[index], p, n, d['accuracy'], d['recall'], d['AUC'], d['F1'], d['precision at' + str(k)]])
                except IndexError as e:
                    print('Error:',e)
                    continue
    return model_list


def precision_at_k(y_true, y_scores, k):
    threshold = np.sort(y_scores)[::-1][int(k*len(y_scores))]
    y_pred = np.asarray([1 if i >= threshold else 0 for i in y_scores])
    return metrics.precision_score(y_true, y_pred)

def plot_precision_recall_n(y_true, y_prob, model_name):
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    name = model_name
    plt.title(name)
    #plt.savefig(name)
    plt.show()




def main(data_filename, response, output_filename): 
    clfs, grid = define_clfs_params()
    models_to_run=['KNN','LR', 'RF', 'ET','AB','GB','NB','DT'] 
    data = pd.read_csv(data_filename)
    data = update_with_cc_means(data, response)
    # data = cat_to_binary(data, response)
    X, y = format_for_models(data, response, list(data.columns.values))
    model_list = magic_loop(models_to_run,clfs,grid,X,y, 0.05)
    # print(model_list)
    with open(output_filename, 'w') as f:
        w = csv.writer(f)
        for line in model_list:
            w.writerow(line)

if __name__ == "__main__":
    main('data/cs-training.csv', 'SeriousDlqin2yrs', 'output/whatever.csv')
