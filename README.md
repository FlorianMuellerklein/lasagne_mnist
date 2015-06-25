# MNIST_CNN

VGG style convolution neural network for the classic MNIST dataset on Kaggle. Currently gets 99.6% on the Kaggle leaderboards.  

### Architecture

The input are 28 x 28 greyscale images
4 convolution layers with filter size 3x3 and ReLU activations. Max pooling layers after every other convolution layer.
2 hidden layers with dropout. Softmax output.

| Layer Type | Parameters |
| -----------|----------- |
| Input      | size: 28x28, channel: 1 |
| convolution| kernel: 3x3, channel: 128 |
| ReLU |  |
| convolution| kernel: 3x3, channel: 128 |
| ReLU | |
| max pool | kernel: 2x2 |
| dropout | 0.2 |
| convolution| kernel: 3x3, channel: 256 |
| ReLU |  |
| convolution| kernel: 3x3, channel: 256 |
| ReLU |  |
| max pool | kernel: 2x2 |
| dropout | 0.2 |
| fully connected | units: 1024 |
| ReLU |  |
| dropout | 0.5 |
| fully connected | units: 1024 |
| ReLU |  |
| dropout | 0.5 |
| softmax | units: 10 |

### Data augmentation

Images are randomly transformed 'on the fly' while they are being prepared in each batch. The CPU will prepare each batch while the GPU will run the previous batch through the network. 

* Random rotations between -5 and 5 degrees.
* Random translation between -3 and 3 pixels in any direction. 
* Random zoom between factors of 0.8 and 1.2. 
* Random shearing between -10 and 10 degrees.
* Sobel edge detector applied to 1/4 of images.

### To-do

Stream data from SSD instead of holding all images in memory (need to install SSD first).
Try different network archetectures and data pre-processing.

### References

* Karen Simonyan, Andrew Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition", [link](http://arxiv.org/abs/1409.1556)
* Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton, "ImageNet Classification with Deep Convolutional Neural Networks", [link](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
* Sander Dieleman, "Classifying plankton with deep neural networks", [link](http://benanne.github.io/2015/03/17/plankton.html)