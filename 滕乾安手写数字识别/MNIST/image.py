# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 18:21:17 2017

@author: pmmxh
"""

from PIL import Image
import numpy

import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

img=numpy.array(Image.open('1.png').convert("L"))
print(numpy.reshape(numpy.array(Image.open('1.png').convert("L")),784)/255)
print(mnist.test.images[0])