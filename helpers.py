import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder

from random import randint, uniform

import seaborn as sns
from matplotlib import pyplot
from skimage.io import imshow
from skimage.util import crop
from skimage import transform, filters, exposure

PIXELS = 28
imageSize = PIXELS * PIXELS
num_features = imageSize

def fast_warp(img, tf, output_shape, mode='nearest'):
    return transform._warps_cy._warp_fast(img, tf.params, output_shape=output_shape, mode=mode)

def batch_iterator(data, y, batchsize, train):
    '''
    Data augmentation batch iterator for feeding images into CNN.
    This example will randomly rotate all images in a given batch between -10 and 10 degrees
    and to random translations between -5 and 5 pixels in all directions.
    Random zooms between 1 and 1.3.
    Random shearing between -20 and 20 degrees.
    Randomly applies sobel edge detector to 1/4th of the images in each batch.
    Randomly inverts 1/2 of the images in each batch.
    '''

    n_samples = data.shape[0]
    loss = []
    for i in range((n_samples + batchsize -1) // batchsize):
        sl = slice(i * batchsize, (i + 1) * batchsize)
        X_batch = data[sl]
        y_batch = y[sl]

        # set empty copy to hold augmented images so that we don't overwrite
        X_batch_aug = np.empty(shape = (X_batch.shape[0], 1, PIXELS, PIXELS), dtype = 'float32')

        # random rotations betweein -8 and 8 degrees
        dorotate = randint(-5,5)

        # random translations
        trans_1 = randint(-3,3)
        trans_2 = randint(-3,3)

        # random zooms
        zoom = uniform(0.8, 1.2)

        # shearing
        shear_deg = uniform(-10, 10)

        # set the transform parameters for skimage.transform.warp
        # have to shift to center and then shift back after transformation otherwise
        # rotations will make image go out of frame
        center_shift   = np.array((PIXELS, PIXELS)) / 2. - 0.5
        tform_center   = transform.SimilarityTransform(translation=-center_shift)
        tform_uncenter = transform.SimilarityTransform(translation=center_shift)

        tform_aug = transform.AffineTransform(rotation = np.deg2rad(dorotate),
                                              scale =(1/zoom, 1/zoom),
                                              shear = np.deg2rad(shear_deg),
                                              translation = (trans_1, trans_2))

        tform = tform_center + tform_aug + tform_uncenter

        # images in the batch do the augmentation
        for j in range(X_batch.shape[0]):

            X_batch_aug[j][0] = fast_warp(X_batch[j][0], tform,
                                          output_shape = (PIXELS, PIXELS))

        # use sobel edge detector filter on one quarter of the images
        indices_sobel = np.random.choice(X_batch_aug.shape[0], X_batch_aug.shape[0] / 4, replace = False)
        for k in indices_sobel:
            img = X_batch_aug[k][0]
            X_batch_aug[k][0] = filters.sobel(img)

        # fit model on each batch
        loss.append(train(X_batch_aug, y_batch))

    return np.mean(loss)

def one_hot(x,n):
	if type(x) == list:
		x = np.array(x)
	x = x.flatten()
	o_h = np.zeros((len(x),n))
	o_h[np.arange(len(x)),x] = 1
	return o_h

def load_data_cv(train_path):

    print('read data')
    # reading training data
    training = pd.read_csv(train_path)
    training = training.values

    # split training labels and pre-process them
    training_targets = training[:,num_features]
    training_targets = training_targets.astype('int32')
    print training_targets.shape
    training_targets = one_hot(training_targets, 10)
    print training_targets.shape

    # split training inputs and scale data 0 to 1
    training_inputs = training[:,0:num_features].astype('float32')
    training_inputs = training_inputs / 255.
    print training_inputs.shape

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(training_inputs, training_targets, test_size = 0.1)

    print 'train size:', x_train.shape[0], 'eval size:', x_test.shape[0]

    # reshaping training and testing data so it can be feed to convolutional layers
    x_train = x_train.reshape(x_train.shape[0], 1, PIXELS, PIXELS)
    x_test = x_test.reshape(x_test.shape[0], 1, PIXELS, PIXELS)

    return x_train, x_test, y_train, y_test

def load_test_data(test_path):
    print('read data')
    # reading training data
    testing = pd.read_csv(test_path)
    testing = testing.values

    # split training inputs and scale data 0 to 1
    testing_inputs = testing[:,0:num_features].astype('float32')
    testing_inputs = testing_inputs / 255.
    print testing_inputs.shape

    # reshaping training and testing data so it can be feed to convolutional layers
    testing_inputs = testing_inputs.reshape(testing_inputs.shape[0], 1, PIXELS, PIXELS)

    return testing_inputs
