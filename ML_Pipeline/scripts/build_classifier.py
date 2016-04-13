
import pandas as pd 
import numpy as np 
from sklearn import linear_model
# from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

def format_for_logit(data, response, predictor):
    y = data[response].values
    if type(predictor) == str:
        x = data[predictor].values
        # y = y.reshape(len(y), 1)
        x = x.reshape(len(x), 1)
        return x, y
    else:
        my_array = data[predictor.pop(0)].values
        my_array = my_array.reshape(len(my_array), 1)
        for i in predictor:
            a = data[i].values
            a = a.reshape(len(a), 1)
            # a = a.reshape(len(a), 1)
            my_array = np.concatenate((my_array, a), axis=1)
        return my_array, y

def logit(x, y):
	'''
	returns 
	'''
	regr = linear_model.LogisticRegression()
	regr.fit(x, y)
	return regr
