import sys
import os
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QLineEdit, QMessageBox
from PyQt5.QtCore import pyqtSlot
from PIL import Image
import numpy

import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#导入TensorFlow
import tensorflow as tf
#通过操作符号变量来描述这些可交互的操作单元
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])
#赋予tf.Variable不同的初值来创建不同的Variable
W = tf.Variable(tf.zeros([784,10]), name="W")
b = tf.Variable(tf.zeros([10]), name="b")
#初始化变量
init_op = tf.global_variables_initializer()
#保存和恢复模型的方法
saver = tf.train.Saver({"W": W, "b": b})
 
class App(QMainWindow):
 
    def __init__(self):
        super().__init__()
        self.title = '手写数字识别'
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
		#启动图
        with tf.Session() as sess:
			#tf.matmul(​​X，W)表示x乘以W，对应之前等式里面的
            y = tf.matmul(x,W) + b
			#保存变量到这个目录“./tmp/model.ckpt”
            saver.restore(sess, "./tmp/model.ckpt")
			#tf.argmax(y_,1) 代表正确的标签，tf.argmax()返回最大数值的下标 通常和tf.equal()一起使用，计算模型准确度
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
            #tf.float32数值类型，tf.cast将correct_prediction的数据格式转化成tf.float32，tf.reduce_mean求平均值tf.reduce_mean
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            #str() 函数将对象转化为适于人阅读的形式。eval语句用来计算存储在代码对象或字符串中的有效的Python表达式
            acr = str(accuracy.eval(session=sess, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
            print(acr)
            QMessageBox.question(self, '验证集', "正确率: " + acr, QMessageBox.Ok, QMessageBox.Ok)
            self.textbox.setText("100")
            
    def on_click1(self):
        textboxValue = self.textbox.text()
        with tf.Session() as sess:
            sess.run(init_op)
            y = tf.matmul(x,W) + b
			#tf.nn.softmax_cross_entropy_with_logits这个函数内部自动计算softmax，然后再计算交叉熵代价函数
			#对向量求均值，logits：就是神经网络最后一层的输出，labels：实际的标签
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
            #tf.train.GradientDescentOptimizer创建优化器并将梯度下降应用于可训练变量。TensorFlow会用你选择的优化算法来不断地修改变量以降低成本
            train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
            #开始训练模型，让模型循环训练，该循环的每个步骤中，我们都会随机抓取训练数据中的60000个批处理数据点，然后我们用这些数据点作为参数替换之前的占位符来运行train_step
            for i in range(int(textboxValue)):
                batch = mnist.train.next_batch(600000)
				#每一步迭代，我们都会加载60000个训练样本，训练100次得到模型，然后执行一次train_step，并通过feed_dict将x 和 y_张量占位符用训练训练数据替代。
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