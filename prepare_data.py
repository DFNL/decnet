#  !/usr/bin/env python
#  -*- coding:utf-8 -*-


# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# https://www.tensorflow.org/get_started/mnist/beginners
# https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist
#


import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
tf.set_random_seed(0)


# data
mnist = read_data_sets("../MNIST_data/", one_hot=True)
print("MNIST data ready for analysis!\n")  # get data ready
batch_size = 100  # how many imgs in each batch?

batch_xs, batch_ys = mnist.train.next_batch(batch_size)
mnist.test.images, mnist.test.labels


'''
In [15]: shape(batch_xs)
Out[15]: (100, 784)

In [16]: shape(batch_ys)
Out[16]: (100, 10)

In [17]: shape(mnist.test.images)
Out[17]: (10000, 784)

In [18]: shape(mnist.test.labels)
Out[18]: (10000, 10)
'''
