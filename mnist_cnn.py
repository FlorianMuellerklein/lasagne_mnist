import numpy as np
import pandas as pd

import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d

import lasagne
from lasagne.nonlinearities import rectify, softmax, very_leaky_rectify
from lasagne.updates import nesterov_momentum
from lasagne.layers import InputLayer, MaxPool2DLayer, Conv2DLayer, DenseLayer, DropoutLayer, helper

from random import randint, uniform
from helpers import batch_iterator, load_data_cv, load_test_data

import seaborn as sns
from matplotlib import pyplot

BATCHSIZE = 32
PIXELS = 28
imageSize = PIXELS * PIXELS
num_features = imageSize

srng = RandomStreams()

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
    # load the training and testing data sets
    train_X, test_X, train_y, test_y = load_data_cv('data/train.csv')

    X = T.ftensor4()
    Y = T.fmatrix()

    # set up theano functions to generate output by feeding data through network
    output_layer = lasagne_model()
    output = helper.get_output(output_layer, X)

    # set up the loss that we aim to minimize
    loss_train = T.mean(lasagne.objectives.categorical_crossentropy(output, Y))

    # set up theano functions to generate outputs for the validation dataset
    output_valid = helper.get_output(output_layer, X, deterministic=True)
    loss_valid = T.mean(lasagne.objectives.categorical_crossentropy(output_valid, Y))

    # prediction functions for classifications
    pred = T.argmax(output, axis=1)
    pred_valid = T.argmax(output_valid, axis=1)

    # get parameters from network and set up sgd with nesterov momentum to update parameters
    params = helper.get_all_params(output_layer)
    updates = nesterov_momentum(loss_train, params, learning_rate=0.005, momentum=0.9)

    # set up training and prediction functions
    train = theano.function(inputs=[X, Y], outputs=loss_train, updates=updates, allow_input_downcast=True)
    predict_valid = theano.function(inputs=[X], outputs=pred_valid, allow_input_downcast=True)

    # loop over training functions for however many iterations, print information while training
    for i in range(60):
        loss_train = batch_iterator(train_X, train_y, BATCHSIZE, train)
        print 'iter:', i, '| Tloss:', loss_train, '| valid acc:', np.mean(np.argmax(test_y, axis=1) == predict_valid(test_X))

    # after training create output for kaggle
    testing_inputs = load_test_data('data/test.csv')
    predictions = []
    for j in range((testing_inputs.shape[0] + BATCHSIZE -1) // BATCHSIZE):
        sl = slice(i * BATCHSIZE, (j + 1) * BATCHSIZE)
        X_batch = testing_inputs[sl]
        predictions.extend(predict_valid(X_batch))
    print len(predictions)
    out = pd.read_csv('data/convnet_preds.csv')
    out['Label'] = predictions
    out.to_csv('preds/convnet_preds.csv', index = False)

if __name__ == '__main__':
    main()
