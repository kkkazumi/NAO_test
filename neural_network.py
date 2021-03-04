#https://qiita.com/ta-ka/items/bcdfd2d9903146c51dcb
import numpy as np
import math
import random

class Neural:
    # constructor
    def __init__(self, n_input, n_hidden, n_output):
        self.hidden_weight = np.random.random_sample((n_hidden, n_input + 1))
        self.output_weight = np.random.random_sample((n_output, n_hidden + 1))
        self.hidden_momentum = np.zeros((n_hidden, n_input + 1))
        self.output_momentum = np.zeros((n_output, n_hidden + 1))
        self.output_num = n_output

        self.input_size = n_input
        self.output_size = n_output

    def train(self, X, T, epsilon, mu, epoch):
        self.error = np.zeros(epoch)
        N = X.shape[0]
        for epo in range(epoch):
            for i in range(N):
                x = X[i, :]
                t = T[i, :]

                self.__update_weight(x, t, epsilon, mu)

            self.error[epo] = self.__calc_error(X, T)

    def error_graph(self):
        pyplot.ylim(0.0, 2.0)
        pyplot.plot(np.arange(0, self.error.shape[0]), self.error)
        pyplot.show()

    def predict(self, X):
        N = X.shape[0]
        C = np.zeros(N).astype('int')
        Y = np.zeros((N, self.output_num))
        for i in range(N):
            x = X[i, :]
            z, y = self.__forward(x)

            Y[i] = y
            C[i] = y.argmax()

        return (C, Y)

    # private method
    def __sigmoid(self, arr):
        return np.vectorize(lambda x: 1.0 / (1.0 + math.exp(-x)))(arr)

    def __forward(self, x):
        # z: output in hidden layer, y: output in output layer
        z = self.__sigmoid(self.hidden_weight.dot(np.r_[np.array([1]), x]))
        y = self.__sigmoid(self.output_weight.dot(np.r_[np.array([1]), z]))

        return (z, y)

    def __update_weight(self, x, t, epsilon, mu):
        z, y = self.__forward(x)

        # update output_weight
        output_delta = (y - t) * y * (1.0 - y)
        _output_weight = self.output_weight
        self.output_weight -= epsilon * output_delta.reshape((-1, 1)) * np.r_[np.array([1]), z] - mu * self.output_momentum
        self.output_momentum = self.output_weight - _output_weight

        # update hidden_weight
        hidden_delta = (self.output_weight[:, 1:].T.dot(output_delta)) * z * (1.0 - z)
        _hidden_weight = self.hidden_weight
        self.hidden_weight -= epsilon * hidden_delta.reshape((-1, 1)) * np.r_[np.array([1]), x]
        self.hidden_momentum = self.hidden_weight - _hidden_weight

    def __calc_error(self, X, T):
        N = X.shape[0]
        err = 0.0
        for i in range(N):
            x = X[i, :]
            t = T[i, :]

            z, y = self.__forward(x)
            err += (y - t).dot((y - t).reshape((-1, 1))) / 2.0

        return err


if __name__ == '__main__':

    X = np.array([[2, 0.5,2], [0, 0.1,3], [1, 0,0.2], [1, 1,0.4]])
    T = np.array([[1, 0.2], [0.3, 1], [0.5, 1], [1, 0]])
    N = X.shape[0] # number of data

    input_size = X.shape[1]
    hidden_size = X.shape[1]
    output_size = T.shape[1]

    epsilon = 0.1
    mu = 0.9
    epoch = 10000

    nn = Neural(input_size, hidden_size, output_size)
    nn.train(X, T, epsilon, mu, epoch)

    C, Y = nn.predict(X)

    for i in range(N):
        x = X[i, :]
        y = Y[i, :]
        t = T[i, :]
        c = C[i]

        #print x
        print y
        print t
