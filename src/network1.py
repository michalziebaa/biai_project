#!/usr/bin/env python
import random
import numpy as np
import pickle

class Network1(object):

    def __init__(self, sizes):

        #For init we load up list with [pixels-inputs  , hidden_neurons  , outputs]
        #We set up random biases and weights for layers
        self.num_layers = len(sizes)
        self.sizes = sizes

        #create array of random biases and weights
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def load_biases_and_weights(self, filename):
        f = open(filename, 'rb')
        self.biases, self.weights = pickle.load(f)
        f.close()
        print "File read: {0}".format(filename)

    def feedforward(self, a):
        #return the output of the network if ``a`` is input.
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, learning_sessions, mini_batch_size, learning_rate, filename, test_data=None):
        if test_data: 
            n_test = len(test_data)
        n = len(training_data)

        for j in xrange(learning_sessions):
            random.shuffle(training_data)
            #create mini_batches from training_data
            #size is set up in the main.py
            mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]

            #for each mini_batch perform the gradient descent in update_mini_batch
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)

            if test_data:
                result = self.evaluate(test_data)

                percentage = float(float(result)/float(n_test))
                percentage = percentage * 100
                print "Learning session {0}: {1} / {2}   {3}%".format(j, result, n_test, percentage)
            else:
                
                print "Learning session {0} complete".format(j)

        f = open(filename, 'wb')
        pickle.dump((self.biases, self.weights), f)
        f.close()
        print("file saved")

    def update_mini_batch(self, mini_batch, learning_rate):

        #create new ndarrays 
        sum_bias = [np.zeros(b.shape) for b in self.biases]
        sum_weight = [np.zeros(w.shape) for w in self.weights]

        #for each mini_batch perform backpropagation
        for x, y in mini_batch:
            delta_bias, delta_weight = self.backprop(x, y)

            #sum up the results of the backpropagation
            sum_bias = [sum_b+delta_b for sum_b, delta_b in zip(sum_bias, delta_bias)]
            sum_weight = [sum_w+delta_w for sum_w, delta_w in zip(sum_weight, delta_weight)]

        #update the biases and weights 
        self.weights = [w-(learning_rate/len(mini_batch))*sum_w for w, sum_w in zip(self.weights, sum_weight)]
        self.biases = [b-(learning_rate/len(mini_batch))*sum_b for b, sum_b in zip(self.biases, sum_bias)]

    def backprop(self, x, y):
        """
        Algorithm:
            1. Input x: Set the corresponding acrivation a1 for the input layer.
            2. Feedforward: For each l = 2,3,...,L compute z(l) = w(l)*a(l-1) + b(l) and a(l)=sigmoid(z(l))
            3. Output error: Compute the vector delta(L) = DELTA(a)C ** sigmoid_prime(z(L))
            4. Backpropagate the error: For each l = L -1, L - 2,... compute 
            5. Output: The gradient of the cost function is given by 
        """

        #create new ndarrays
        sum_bias = [np.zeros(b.shape) for b in self.biases]
        sum_weight = [np.zeros(w.shape) for w in self.weights]

        # feedforward

        # Step 1.
        activation = x
        activations = [x] #list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer

        # Step 2.
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b     #w0*a0 + w1*a1 + ... wn*an + b
            zs.append(z)                    #add z to the list
            activation = sigmoid(z)         #calculate the activation aj(L)
            activations.append(activation)  #add activation to list

        # backward pass
        # Step 3.
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])


        sum_bias[-1] = delta
        sum_weight[-1] = np.dot(delta, activations[-2].transpose())

        # Step 4.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp

            #Step 5.
            sum_bias[-l] = delta
            sum_weight[-l] = np.dot(delta, activations[-l-1].transpose())
        return (sum_bias, sum_weight)

    #check the tests
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
