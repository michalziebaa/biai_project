from csv import reader
import numpy as np
import sys

def load_csv(filename):
	file = open(filename, "r")
	lines = reader(file)
	dataset = list(lines)
	return dataset 

def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

def load_csv_numpy(filename):
	data = np.genfromtxt(filename, delimiter=',', names=True)
	return data


def convert_to_wanted_format(filename):
	array = load_csv_numpy(filename)

	dataset_x = np.zeros(shape=(  len(array), int(len(array[0])-1) ),dtype="float32")
	dataset_y = np.zeros(len(array))
	shape = dataset_x.shape

	dataset_y = [np.int64(array[i][0]) for i in xrange(len(array))]
	dataset_y = np.asarray(dataset_y)

	for row in range(1, shape[0]):
		for col in range(1, shape[1]):
			dataset_x[row][col] = np.float32(array[row][col+1])

	dataset_final = (dataset_x, dataset_y)
	return dataset_final



