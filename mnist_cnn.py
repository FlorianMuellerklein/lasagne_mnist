import gzip
import pickle
import numpy as np
import pandas as pd

import theano
from theano import tensor as T

import lasagne
from lasagne.nonlinearities import rectify, softmax, very_leaky_rectify
from lasagne.updates import nesterov_momentum
from lasagne.layers import InputLayer, MaxPool2DLayer, Conv2DLayer, DenseLayer, DropoutLayer, helper

from helpers import batch_iterator, load_data_cv, load_test_data

import seaborn as sns
from matplotlib import pyplot

BATCHSIZE = 32
PIXELS = 28
imageSize = PIXELS * PIXELS
num_features = imageSize

# set up functions needed to train the network
def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def lasagne_model():
    l_in = InputLayer(shape=(None, 1, 28, 28))

    l_conv1 = Conv2DLayer(l_in, num_filters = 128, filter_size=(3,3), nonlinearity=rectify)
    l_conv1b = Conv2DLayer(l_conv1, num_filters = 128, filter_size=(3,3), nonlinearity=rectify)
    l_pool1 = MaxPool2DLayer(l_conv1b, pool_size=(2,2))
    l_dropout1 = DropoutLayer(l_pool1, p=0.2)

    l_conv2 = Conv2DLayer(l_pool1, num_filters = 256, filter_size=(3,3), nonlinearity=rectify)
    l_conv2b = Conv2DLayer(l_conv2, num_filters = 256, filter_size=(3,3), nonlinearity=rectify)
    l_pool2 = MaxPool2DLayer(l_conv2b, pool_size=(2,2))
    l_dropout2 = DropoutLayer(l_pool2, p=0.2)

    l_hidden3 = DenseLayer(l_dropout2, num_units = 1024, nonlinearity=rectify)
    l_dropout3 = DropoutLayer(l_hidden3, p=0.5)

    l_hidden4 = DenseLayer(l_dropout3, num_units = 1024, nonlinearity=rectify)
    l_dropout4 = DropoutLayer(l_hidden4, p=0.5)

    l_out = DenseLayer(l_dropout4, num_units=10, nonlinearity=softmax)

    return l_out

def main():
    # load the training and validation data sets
    train_X, test_X, train_y, test_y = load_data_cv('data/train.csv')

    X = T.ftensor4()
    Y = T.fmatrix()

    # set up theano functions to generate output by feeding data through network
    output_layer = lasagne_model()
    output_train = lasagne.layers.get_output(output_layer, X)
    output_valid = lasagne.layers.get_output(output_layer, X, deterministic=True)

    # set up the loss that we aim to minimize
    loss_train = T.mean(T.nnet.categorical_crossentropy(output_train, Y))
    loss_valid = T.mean(T.nnet.categorical_crossentropy(output_valid, Y))

    # prediction functions for classifications
    pred = T.argmax(output_train, axis=1)
    pred_valid = T.argmax(output_valid, axis=1)

    # get parameters from network and set up sgd with nesterov momentum to update parameters
    params = lasagne.layers.get_all_params(output_layer)
    updates = nesterov_momentum(loss_train, params, learning_rate=0.003, momentum=0.9)

    # set up training and prediction functions
    train = theano.function(inputs=[X, Y], outputs=loss_train, updates=updates, allow_input_downcast=True)
    valid = theano.function(inputs=[X, Y], outputs=loss_valid, allow_input_downcast=True)
    predict_valid = theano.function(inputs=[X], outputs=pred_valid, allow_input_downcast=True)

    # loop over training functions for however many iterations, print information while training
    train_eval = []
    valid_eval = []
    valid_acc = []
    try:
        for i in range(45):
            train_loss = batch_iterator(train_X, train_y, BATCHSIZE, train)
            train_eval.append(train_loss)
            valid_loss = valid(test_X, test_y)
            valid_eval.append(valid_loss)
            acc = np.mean(np.argmax(test_y, axis=1) == predict_valid(test_X))
            valid_acc.append(acc)
            print 'iter:', i, '| Tloss:', train_loss, '| Vloss:', valid_loss, '| valid acc:', acc

    except KeyboardInterrupt:
        pass

    # save weights
    all_params = helper.get_all_param_values(output_layer)
    f = gzip.open('data/weights.pklz', 'wb')
    pickle.dump(all_params, f)
    f.close()

    # plot loss and accuracy
    train_eval = np.array(train_eval)
    valid_eval = np.array(valid_eval)
    valid_acc = np.array(valid_acc)
    sns.set_style("whitegrid")
    pyplot.plot(train_eval, linewidth = 3, label = 'train loss')
    pyplot.plot(valid_eval, linewidth = 3, label = 'valid loss')
    pyplot.legend(loc = 2)
    pyplot.twinx()
    pyplot.plot(valid_acc, linewidth = 3, label = 'valid accuracy', color = 'r')
    pyplot.grid()
    pyplot.ylim([.9,1])
    pyplot.legend(loc = 1)
    pyplot.savefig('data/training_plot.png')
    #pyplot.show()

if __name__ == '__main__':
    main()
