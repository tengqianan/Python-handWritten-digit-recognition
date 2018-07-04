# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 20:15:39 2017

@author: pmmxh
"""

import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

W = tf.Variable(tf.zeros([784,10]), name="W")
b = tf.Variable(tf.zeros([10]), name="b")

init_op = tf.global_variables_initializer()

saver = tf.train.Saver({"W": W, "b": b})

with tf.Session() as sess:
    sess.run(init_op)
    y = tf.matmul(x,W) + b
    cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    for i in range(1000):
        batch = mnist.train.next_batch(600000)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
        print("%.1fprecent" % (i/10.0))
    
    save_path = saver.save(sess, "./tmp/model.ckpt")
    
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(accuracy.eval(session=sess, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    