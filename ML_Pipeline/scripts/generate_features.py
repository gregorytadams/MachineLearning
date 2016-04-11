
import pandas as pd 

def round_col(data, col_name, num_digits=0):
	'''
	Rounds all the variables in a given row by num_digits

	data: Dataframe
	'''
	data[col_name] = data[col_name].apply(lambda x: round(x, num_digits))
	return data

# example of what the dict might look like
# keys are names of columns
CATEGORICAL_DICT = {'col_one': [("low", 0, 2), ("medium", 2, 6), ("high", 6, float('inf'))], \
'col_two': [("low", 0.1, 95), ("medium", 95, 141), ("high", 141, float('inf'))]} 

def bucket_values(data, col_name, categorical_list=CATEGORICAL_DICT):
	data[col_name] = data[col_name].apply(lambda x: convert_value(x, col_name))
	return data

def convert_value(value, col_name, categorical_dict=CATEGORICAL_DICT):
	for triplet in categorical_dict[col_name]: 
	    if value >= triplet[1] and value < triplet[2]:
	        return triplet[0]
	        
def cat_to_binary(data, col_name):
	'''
	Converts a categorical variable to integer values.  Will make variable binary if there
	are only two values.
	'''
	vals = data[col_name].unique()
	for i, val in enumerate(vals):
		print("Replacing {} with {}".format(val, i))
		data[col_name].replace(val, i, inplace=True)
	return data
	

