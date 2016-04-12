
import read_data
import explore_data
import preprocess_data
import generate_features
import build_classifier
import evaluate_classifier
import pandas
import matplotlib.pyplot as plt
from sklearn import linear_model

def HW2(training_file, testing_file):
	df = read_data.make_dataframe(training_file)
	explore_data.save_dataframe('summary', df.describe())
	explore_data.save_hist(df, 'output/histograms.pdf')
	df = preprocess_data.update_with_mean(df)
	x, y = preprocess_data.format_for_logit(df.copy(), 'SeriousDlqin2yrs', list(df.columns.values[2:9]))
	X_train, X_test, y_train, y_test = preprocess_data.split_train_test(x, y, 0.2)
	regr = build_classifier.logit(X_train, y_train)
	score = regr.score(X_test, y_test)
	df_pred = read_data.make_dataframe(testing_file)
	df_pred = preprocess_data.update_with_mean(df_pred)
	x2, y2 = preprocess_data.format_for_logit(df_pred.copy(), 'SeriousDlqin2yrs', list(df_pred.columns.values[2:9]))
	return regr.predict(x2), score