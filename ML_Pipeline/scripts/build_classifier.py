
import pandas as pd 
import numpy as np 
from sklearn import linear_model
# from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

# def logistic_regr(training_data, response, predictors):
# 	'''
# 	'''
# 	cls = LogisticRegression()
# 	features = list(predictors)
# 	target = list(response)
# 	features_train, features_test, target_train, target_test = train_test_split(features, target)
# 	cls.fit(features_train, target_train)
# 	predictions = cls.predict(features_test)
# 	return predictions

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

def replace_missing(x,y,replace_val=0):
	'''
	awk AF
	'''
	if x.shape[1] > 1:
		for i, val in enumerate(x):
		    for j, k in enumerate(val):
		        if np.isnan(k):
		            x[i][j] = replace_val
    else:
	    for i, val in enumerate(x):
	        if np.isnan(val):
			    x[i] = replace_val
	for i, val in enumerate(y):
		if np.isnan(val):
			y[i] = replace_val
	return x, y



def logit(x, y):
	'''
	returns 
	'''
	print(x.shape)
	print(y.shape)
	regr = linear_model.LogisticRegression()
	regr.fit(x, y)
	return regr

# def multiple_logit():

# def plot_logit()
