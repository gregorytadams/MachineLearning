
import pandas as pd 
import numpy as np 
import sklearn.linear_model
from sklearn import svm
from sklearn import tree 
from sklearn import neighbors
from sklearn import naive_bayes 
from sklearn.cross_validation import train_test_split
from preprocess_data import split_train_test
from itertools import combinations

def format_for_logit(data, response, predictor):
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


def do_logit(x, y):
	'''
	returns 
	'''
	REGR = sklearn.linear_model.LogisticRegression()
	REGR.fit(x, y)
	return REGR

def do_SVM(x, y, C_val=1.0):
    SVM = sklearn.svm.LinearSVC(C=C_val)
    SVM.fit(x, y)
    return SVM

def do_decision_tree(x, y):
    TREE = sklearn.tree.DecisionTreeClassifier()
    TREE.fit(x, y)
    return TREE 

def do_bayesian(x, y):
    BAYES = sklearn.naive_bayes.GaussianNB()
    BAYES.fit(x ,y)
    return BAYES

def do_knn(x, y):
    KNN = sklearn.neighbors.KNeighborsClassifier()
    KNN.fit(x,y)
    return KNN

def find_best_model(data, response, features):
    '''
    Do every model for every combination and find the best model for the data
    '''
    model_d = {}
    x, y = format_for_logit(data, response, features)
    x, x_test, y, y_test  = split_train_test(x, y, 0.2) #from preprocess
    model_d['logit'] = do_logit(x, y)
    # print("Logit constructed")
    model_d['SVM'] = do_SVM(x,y)
    # print("SVM constructed")
    model_d['decision_tree'] = do_decision_tree(x,y)
    # print("Decision Tree constructed")
    model_d['bayesian'] = do_bayesian(x,y)
    # print("Bayesian constructed")
    model_d['knn'] = do_knn(x,y)
    # print("K-Nearest-Neighbor constructed")
    best_model, best_score, score_d = score_models(model_d, x_test, y_test)
    return model_d, score_d, best_model, best_score

def score_models(d, x_test, y_test):
    # print("evaluating...")
    score_d = {}
    for i in d:
        score_d[i] = d[i].score(x_test, y_test)
    best_model = max(score_d, key=score_d.get)
    return best_model, score_d[best_model], score_d

def run_all_combos(data, response, features):
    '''
    This will take a long time.  Do not run until necessary.
    '''
    l = []
    for i in range(3, len(features)):
        for j in combinations(features, i):
            l.append(list(j))
    tracking_d = {}
    best_features = (0,0,0)
    for i, combo in enumerate(l):
        model_d, score_d, best_model, best_score = find_best_model(data, response, combo)
        if best_score > best_features[2]:
            best_features = (combo, best_model, best_score)
        print("Features {}/{}".format(i, len(l)))
        # tracking_d[i] = (best_model, best_score)
    # best_features = max(tracking_d, key=tracking_d.get)
    # return best_features, tracking_d[best_features]
    return best_features

### Ensemble models

def bagging():
    pass

def boosting():
    pass

def random_forest():
    pass

