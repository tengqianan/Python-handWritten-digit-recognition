import sys
import os
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QLineEdit, QMessageBox
from PyQt5.QtCore import pyqtSlot
from PIL import Image
import numpy

import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

W = tf.Variable(tf.zeros([784,10]), name="W")
b = tf.Variable(tf.zeros([10]), name="b")

init_op = tf.global_variables_initializer()

saver = tf.train.Saver({"W": W, "b": b})
 
class App(QMainWindow):
 
    def __init__(self):
        super().__init__()
        self.title = '数据挖掘课程设计-手写数字识别'
        self.left = 10
        self.top = 10
        self.width = 400
        self.height = 140
        self.initUI()
 
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
 
        self.textbox = QLineEdit(self)
        self.textbox.move(20, 20)
        self.textbox.resize(240,40)
        self.textbox.setText("100")
 
        self.button = QPushButton('验证', self)
        self.button.move(20,80)
        
        self.button1 = QPushButton('训练', self)
        self.button1.move(120,80)
        
        self.button2 = QPushButton('检测', self)
        self.button2.move(220,80)
        
        self.button3 = QPushButton('编辑', self)
        self.button3.move(290,20)
 
        self.button.clicked.connect(self.on_click)
        self.button1.clicked.connect(self.on_click1)
        self.button2.clicked.connect(self.on_click2)
        self.button3.clicked.connect(self.on_click3)
        self.show()
 
    @pyqtSlot()
    def on_click(self):
        with tf.Session() as sess:
            y = tf.matmul(x,W) + b
            saver.restore(sess, "./tmp/model.ckpt")
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            acr = str(accuracy.eval(session=sess, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
            print(acr)
            QMessageBox.question(self, '验证集', "正确率: " + acr, QMessageBox.Ok, QMessageBox.Ok)
            self.textbox.setText("100")
            
    def on_click1(self):
        textboxValue = self.textbox.text()
        with tf.Session() as sess:
            sess.run(init_op)
            y = tf.matmul(x,W) + b
            cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
            train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
            for i in range(int(textboxValue)):
                batch = mnist.train.next_batch(600000)
                train_step.run(feed_dict={x: batch[0], y_: batch[1]})
                print("%.1fprecent" % (i*100.0/int(textboxValue)))
            saver.save(sess, "./tmp/model.ckpt")
        QMessageBox.question(self, '完成', "训练完成 ", QMessageBox.Ok, QMessageBox.Ok)
    
    def on_click2(self):
        with tf.Session() as sess:
            y = tf.matmul(x,W) + b
            saver.restore(sess, "./tmp/model.ckpt")
            QMessageBox.question(self, '检测', "识别结果: " + str(numpy.argmax(sess.run(y, feed_dict={x: 1-numpy.reshape(numpy.array(Image.open('2.png').convert("L")),(1,784))/255}))), QMessageBox.Ok, QMessageBox.Ok)
       
    def on_click3(self):
        os.popen('copy 1.png 2.png')
        os.popen('mspaint.exe 2.png')
 
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())