# This is a file with broken code, code I decided not to use, oddly specific debugging code, etc. \
# that probably won't ever be used again, but may help to have.  I break every rule of good programming here \
# (globals everywhere, magic numbers, hardcoded solutions), so please do not look for logic here.  None exists.

    # list_of_names = []
    # for i, value in enumerate(data['Gender'].isnull()):
    #     if value == True:
    #         list_of_names.append(data.First_name.loc[i])
    # print(list_of_names)
    # d = get_genders(list_of_names)
    # print(d)


# def get_genders(list_of_names):
    # '''
    # Grabs the names all at once instead of one at a time.  Turns out that the api limits 
    # you to 10 at a time, though, so it doesn't matter that much.  Fuck me, right?
    # '''
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



    #     for i, b in enumerate(data[col].isnull()): 
    #         if b:
    #             vals = vc[col].axes[0]
    #             probs = list(map(lambda x: x/sum(vc[col]), vc[col]))
    #             data.set_value(i, col, int(choice(vals, 1, probs)))  #choice is from numpy.random
    # return data


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




# def replace_missing(x,y,replace_val=0):
# 	'''
# 	awk AF
# 	'''
# 	if x.shape[1] > 1:
# 		for i, val in enumerate(x):
# 		    for j, k in enumerate(val):
# 		        if np.isnan(k):
# 		            x[i][j] = replace_val
#     else:
# 	    for i, val in enumerate(x):
# 	        if np.isnan(val):
# 			    x[i] = replace_val
# 	for i, val in enumerate(y):
# 		if np.isnan(val):
# 			y[i] = replace_val
# 	return x, y

#ipython stuff for the quickness
# def dont_run_this_it_isnt_python():
# 	'''
# 	I use this for when I hstart ipython to save me time
# 	'''
	# run scripts/read_data
	# run scripts/explore_data
	# run scripts/preprocess_data
	# run scripts/generate_features
	# run scripts/build_classifier
	# run scripts/evaluate_classifier
	# import pandas
	# import matplotlib.pyplot as plt
	# import sklearn as skl 
	# import numpy as np 

LOG_SCALE_LIST = [ 'RevolvingUtilizationOfUnsecuredLines','NumberOfTime30-59DaysPastDueNotWorse','DebtRatio', \
'MonthlyIncome','NumberOfTimes90DaysLate','NumberRealEstateLoansOrLines', \
 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents']

from explore_data import save_hist

def make_all_indiv_hist(data, log_scale_list=LOG_SCALE_LIST):
	for i in data.columns.values[1:]:
		# print(type(i))
		# try:
		if str(i) in log_scale_list:
			print("IF Saving {}".format(str(i)))
			save_hist(data, 'hist.pdf', i, log_scale=True)
		else:
			print("ELSE Saving {}".format(str(i)))
			save_hist(data, 'hist.pdf', i)
		# except:
		# 	print("{} didn't save".format(i))
	# print("Finished")
