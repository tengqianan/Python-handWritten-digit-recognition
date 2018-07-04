# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 19:31:54 2017

@author: pmmxh
"""

from PIL import Image
import numpy

#import input_data
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

W = tf.Variable(tf.zeros([784,10]), name="W")
b = tf.Variable(tf.zeros([10]), name="b")

#init_op = tf.global_variables_initializer()

saver = tf.train.Saver({"W": W, "b": b})

with tf.Session() as sess:
    y = tf.matmul(x,W) + b
    saver.restore(sess, "./tmp/model.ckpt")
    
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #print(accuracy.eval(session=sess, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    #print(sess.run(accuracy, feed_dict={x: numpy.reshape(numpy.array(Image.open('2.png').convert("L")),(1,784))/255, y_: numpy.reshape([0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],(1,10))}))
    print(numpy.argmax(sess.run(y, feed_dict={x: 1-numpy.reshape(numpy.array(Image.open('2.png').convert("L")),(1,784))/255})))