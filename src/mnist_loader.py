#!/usr/bin/env python
import cPickle
import gzip

import numpy as np
import load_csv

def load_data():

    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)

    #training_data = load_csv.convert_to_wanted_format('emnist-letters-train.csv')
    #test_data = load_csv.convert_to_wanted_format('emnist-letters-test.csv')

    print(type(training_data))
    print("---")
    print(type(training_data[0]))
    print(type(training_data[0][0]))
    print(type(training_data[0][0][0]))

    print("---")
    print(type(training_data[1]))
    print(type(training_data[1][0]))

    print("======================")
    #f.close()
    return (training_data, test_data)
    #return (training_data, validation_data, test_data)


def load_data_wrapper():

    tr_d, te_d = load_data()
    #zmieniamy kazdy ndarray na wymiar 784x1
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    #zmieniamy druga wartosc tak aby byla ndarrayem 
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)

    #validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    #validation_data = zip(validation_inputs, va_d[1])

    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    #test_results = [vectorized_result(y) for y in te_d[1]]

    test_data = zip(test_inputs, te_d[1])
    


    return (training_data, test_data)

def vectorized_result(j):
    #remember to change this to 26 when working with letters
    e = np.zeros((26, 1))
    e[j] = 1.0
    return e
