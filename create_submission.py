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

from mnist_cnn import lasagne_model
from helpers import batch_iterator, batch_iterator_no_aug, load_data_cv, load_test_data

BATCHSIZE = 32
PIXELS = 28
imageSize = PIXELS * PIXELS
num_features = imageSize

def main():
    # load model and parameters
    output_layer = lasagne_model()
    f = gzip.open('data/weights.pklz', 'rb')
    all_params = pickle.load(f)
    f.close()

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
    helper.set_all_param_values(output_layer, all_params)
    params = lasagne.layers.get_all_params(output_layer)
    updates = nesterov_momentum(loss_train, params, learning_rate=0.0001, momentum=0.9)

    # set up training and prediction functions
    train = theano.function(inputs=[X, Y], outputs=loss_train, updates=updates, allow_input_downcast=True)
    valid = theano.function(inputs=[X, Y], outputs=loss_valid, allow_input_downcast=True)
    predict_valid = theano.function(inputs=[X], outputs=pred_valid, allow_input_downcast=True)

    # fine tune network
    train_X, test_X, train_y, test_y = load_data_cv('data/train.csv')
    train_eval = []
    valid_eval = []
    valid_acc = []
    try:
        for i in range(5):
            train_loss = batch_iterator_no_aug(train_X, train_y, BATCHSIZE, train)
            train_eval.append(train_loss)
            valid_loss = valid(test_X, test_y)
            valid_eval.append(valid_loss)
            acc = np.mean(np.argmax(test_y, axis=1) == predict_valid(test_X))
            valid_acc.append(acc)
            print 'iter:', i, '| Tloss:', train_loss, '| Vloss:', valid_loss, '| valid acc:', acc

    except KeyboardInterrupt:
        pass

    # after training create output for kaggle
    testing_inputs = load_test_data('data/test.csv')
    predictions = []
    for j in range((testing_inputs.shape[0] + BATCHSIZE -1) // BATCHSIZE):
        sl = slice(j * BATCHSIZE, (j + 1) * BATCHSIZE)
        X_batch = testing_inputs[sl]
        predictions.extend(predict_valid(X_batch))
    out = pd.read_csv('data/convnet_preds.csv')
    out['Label'] = predictions
    out.to_csv('preds/convnet_preds.csv', index = False)

if __name__ == '__main__':
    main()
