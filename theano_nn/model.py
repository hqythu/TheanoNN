import csv
import theano
from theano import tensor as T
from theano import function
import scipy.io as sio
import numpy as np


class Model():
    def __init__(self, learning_rate, momentum, regularization, batch_size, epoch_time, disp_freq=1):
        self.layers = []
        self.params = []
        self.regularization_param = []
        self.input = T.fmatrix()
        self.output = self.input
        self.test_output = self.input
        self.label = T.fmatrix()

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.regularization = regularization
        self.batch_size = batch_size
        self.epoch_time = epoch_time
        self.disp_freq = disp_freq

    def add_layer(self, layer):
        self.layers.append(layer)
        self.params += layer.params
        self.regularization_param += layer.regularization
        self.output = layer.get_output(self.output)
        self.test_output = layer.get_test_output(self.test_output)

    def set_loss_function(self, loss_function):
        self.loss_function = loss_function

    def build(self):
        self.cost = self.loss_function(self.output, self.label)
        reg_cost = self.cost
        for reg in self.regularization_param:
            reg_cost = reg_cost + self.regularization * reg

        self.label_predict = T.argmax(self.test_output, axis=1)

        self.params_update = [theano.shared(param.get_value() * 0) for param in self.params]
        updates = [(param, param - self.learning_rate * param_update)
            for param, param_update in zip(self.params, self.params_update)]
        updates += [(param_update, param_update * self.momentum +
            (1.0 - self.momentum) * T.grad(reg_cost, param))
            for param, param_update in zip(self.params, self.params_update)]

        self.train = function([self.input, self.label], self.cost,
            updates=updates, allow_input_downcast=True)
        self.cost_fun = function([self.input, self.label], self.cost,
            allow_input_downcast=True)
        self.predict = function([self.input], self.label_predict,
            allow_input_downcast=True)

    def train_model(self, train_x, train_y, valid_x, valid_y):
        for i in range(self.epoch_time):
            if ((i+1) % self.disp_freq == 0):
                print 'epoch:', i+1, ',',
            cost = []
            for start, end in zip(range(0, len(train_x), self.batch_size),
                range(self.batch_size, len(train_x), self.batch_size)):
                cost += [self.train(train_x[start:end], train_y[start:end])]
            accuracy = np.mean(np.argmax(valid_y, axis=1) == self.predict(valid_x))
            valid_cost = self.cost_fun(valid_x, valid_y)
            if ((i+1) % self.disp_freq == 0):
                print 'training cost:', np.mean(cost), ',', 'validation cost:', valid_cost, \
                    ',', 'accuracy:', accuracy
