#!/usr/bin/env python

from network1 import Network1
import mnist_loader
import pickle

pixels_inputs = 784
hidden_neurons = 100
outputs = 26

#wczytanie danych

#training_data, test_data = mnist_loader.load_data_wrapper()

#f = open('ready_data', 'wb')
#pickle.dump((training_data, test_data), f)
#f.close()
#print("file ready")




f = open('ready_data', 'rb')
training_data, test_data = pickle.load(f)
f.close()


#read_from_file = 'learning_40_40_40h_1'
#pixels-inputs   hidden_neurons   outputs
net = Network1([pixels_inputs, 50, 35, 32, outputs])

# loading weights and biases
#net.load_biases_and_weights(read_from_file)

epochs = 1200
mini_batch_size = 32
learning_rate = 0.2


save_to_file = 'learning_50_35_32h_1'
net.SGD(training_data, epochs, mini_batch_size, learning_rate, save_to_file, test_data=test_data)
