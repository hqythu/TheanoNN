import csv
import theano
from theano import tensor as T
from theano import function
import scipy.io as sio
import numpy as np

from load import mnist


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

        self.label_predict = self.test_output #T.argmax(self.output, axis=1)

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
            tmp = self.predict(valid_x) - valid_y
            accuracy = np.sqrt(np.mean(tmp * tmp)) * 48
            valid_cost = self.cost_fun(valid_x, valid_y)
            if ((i+1) % self.disp_freq == 0):
                print 'training cost:', np.mean(cost), ',', 'validation cost:', valid_cost, \
                    ',', 'accuracy:', accuracy

    def save_test_result(self, test_x):
        dic = {
                'left_eye_center_x'         :  0,
                'left_eye_center_y'         :  1,
                'right_eye_center_x'        :  2,
                'right_eye_center_y'        :  3,
                'left_eye_inner_corner_x'   :  4,
                'left_eye_inner_corner_y'   :  5,
                'left_eye_outer_corner_x'   :  6,
                'left_eye_outer_corner_y'   :  7,
                'right_eye_inner_corner_x'  :  8,
                'right_eye_inner_corner_y'  :  9,
                'right_eye_outer_corner_x'  : 10,
                'right_eye_outer_corner_y'  : 11,
                'left_eyebrow_inner_end_x'  : 12,
                'left_eyebrow_inner_end_y'  : 13,
                'left_eyebrow_outer_end_x'  : 14,
                'left_eyebrow_outer_end_y'  : 15,
                'right_eyebrow_inner_end_x' : 16,
                'right_eyebrow_inner_end_y' : 17,
                'right_eyebrow_outer_end_x' : 18,
                'right_eyebrow_outer_end_y' : 19,
                'nose_tip_x'                : 20,
                'nose_tip_y'                : 21,
                'mouth_left_corner_x'       : 22,
                'mouth_left_corner_y'       : 23,
                'mouth_right_corner_x'      : 24,
                'mouth_right_corner_y'      : 25,
                'mouth_center_top_lip_x'    : 26,
                'mouth_center_top_lip_y'    : 27,
                'mouth_center_bottom_lip_x' : 28,
                'mouth_center_bottom_lip_y' : 29
              }

        answer = self.predict(test_x)
        answer = answer * 48.0 + 48
        sio.savemat('result.mat', {'test_x': test_x,
                                   'test_y': answer} )
        csvfile = file('result.csv', 'wb')
        writer = csv.writer(csvfile)
        writer.writerow(['RowId', 'Location'])

        csvmodel = file('IdLookupTable.csv', 'rb')
        csvmodel.readline()
        text = np.loadtxt(csvmodel, dtype=np.str, delimiter=",")
        csvmodel.close()
        for i in range(len(text)):
            writer.writerow([ int(text[i][0]), answer[ int(text[i][1])-1 ][ dic[text[i][2]] ] ])

        csvfile.close()
