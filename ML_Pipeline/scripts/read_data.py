#Reads in data

from pandas import read_csv

def make_dataframe(filename):
	return read_csv(filename)