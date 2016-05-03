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

# LOG_SCALE_LIST = [ 'RevolvingUtilizationOfUnsecuredLines','NumberOfTime30-59DaysPastDueNotWorse','DebtRatio', \
# 'MonthlyIncome','NumberOfTimes90DaysLate','NumberRealEstateLoansOrLines', \
#  'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents']

# from explore_data import save_hist

# def make_all_indiv_hist(data, log_scale_list=LOG_SCALE_LIST):
# 	for i in data.columns.values[1:]:
# 		# print(type(i))
# 		# try:
# 		if str(i) in log_scale_list:
# 			print("IF Saving {}".format(str(i)))
# 			save_hist(data, 'hist.pdf', i, log_scale=True)
# 		else:
# 			print("ELSE Saving {}".format(str(i)))
# 			save_hist(data, 'hist.pdf', i)
# 		# except:
# 		# 	print("{} didn't save".format(i))
# 	# print("Finished")


#         # tracking_d[i] = (best_model, best_score)
    # best_features = max(tracking_d, key=tracking_d.get)
    # return best_features, tracking_d[best_features]


# def do_logit(x, y):
#   '''
#   returns 
#   '''
#   REGR = sklearn.linear_model.LogisticRegression()
#   REGR.fit(x, y)
#   return REGR

# def do_SVM(x, y, C_val=1.0):
#     SVM = sklearn.svm.LinearSVC(C=C_val)
#     SVM.fit(x, y)
#     return SVM

# def do_decision_tree(x, y):
#     TREE = sklearn.tree.DecisionTreeClassifier()
#     TREE.fit(x, y)
#     return TREE 

# def do_bayesian(x, y):
#     BAYES = sklearn.naive_bayes.GaussianNB()
#     BAYES.fit(x ,y)
#     return BAYES

# def do_knn(x, y):
#     KNN = sklearn.neighbors.KNeighborsClassifier()
#     KNN.fit(x,y)
#     return KNN

# def find_best_model(data, response, features):
#     '''
#     Do every model for every combination and find the best model for the data
#     '''
#     model_d = {}
#     x, y = format_for_logit(data, response, features)
#     x, x_test, y, y_test  = split_train_test(x, y, 0.2) #from preprocess
#     model_d['logit'] = do_logit(x, y)
#     # print("Logit constructed")
#     # model_d['SVM'] = do_SVM(x,y)
#     # print("SVM constructed")
#     model_d['decision_tree'] = do_decision_tree(x,y)
#     # print("Decision Tree constructed")
#     model_d['bayesian'] = do_bayesian(x,y)
#     # print("Bayesian constructed")
#     model_d['knn'] = do_knn(x,y)
#     # print("K-Nearest-Neighbor constructed")
#     best_model, best_score, score_d = score_models(model_d, x_test, y_test)
#     return model_d, score_d, best_model, best_score

# def score_models(d, x_test, y_test):
#     # print("evaluating...")
#     score_d = {}
#     for i in d:
#         score_d[i] = d[i].score(x_test, y_test)
#     best_model = max(score_d, key=score_d.get)
#     return best_model, score_d[best_model], score_d

# def run_all_combos(data, response, features, min_features):
#     '''
#     This will take a long time.  Do not run until necessary.
#     '''
#     l = []
#     for i in range(min_features, len(features)):
#         for j in combinations(features, i):
#             l.append(list(j))
#     tracking_d = {}
#     best_features = (0,0,0)
#     for i, combo in enumerate(l):
#         model_d, score_d, best_model, best_score = find_best_model(data, response, combo)
#         if best_score > best_features[2]:
#             best_features = (combo, best_model, best_score)
#         print("Features {}/{}".format(i, len(l)))
#     return  best_features

# def split_train_test(data_x, data_y, prop_in_test):
#     '''
#     splits and formats for logit function
#     '''
#     X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=float(prop_in_test))
#     return X_train, X_test, y_train, y_test