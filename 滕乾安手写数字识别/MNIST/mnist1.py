# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 16:39:11 2017

@author: pmmxh
"""
#from PIL import Image
#import numpy

import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

W = tf.Variable(tf.zeros([784,10]), name="W")
b = tf.Variable(tf.zeros([10]), name="b")

init_op = tf.global_variables_initializer()

saver = tf.train.Saver()

init_op.run()

y = tf.nn.softmax(tf.matmul(x,W) + b)

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

for i in range(1000):
  batch = mnist.train.next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})
  
save_path = saver.save(sess, "./tmp/model.ckpt")
  
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

#print(sess.run(accuracy, feed_dict={x: numpy.reshape(numpy.array(Image.open('1.png').convert("L")),(1,784))/255, y_: numpy.reshape([0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],(1,10))}))

#pprint(dump(str(mnist.test.images[0])))