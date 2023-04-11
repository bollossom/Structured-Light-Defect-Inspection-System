import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series,DataFrame
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
import matplotlib.image as mpimg
import os
from PIL import Image
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import os
from skimage import io
import random
import math
import cv2
import scipy
import cv2



def load_data(data_path, size):
    image_path = []
    label_path = []
    signal = os.listdir(data_path)  # 所有文件夹的名称（集合）

    count_x = 70 # 数据集的图片总数 160
    step = 1  # 从12张图的第几张开始读取
    ad = 10  # 每个小文件夹里读取多少张图

    data_x = np.empty((count_x, size[0], size[1], 1), dtype="float32")
    data_y = np.empty((count_x, size[0], size[1], 1), dtype="float32")
    data_b = np.empty((count_x, size[0], size[1], 1), dtype="float32")

    q = 0  # 用来标记文件夹数量
    for fsingal in signal:
        filepath = data_path + "\\" + fsingal  # 图片所在文件夹的路径（单）
        for j in range(1):  # 每12张图读取几张
            x = np.zeros((size[0], size[1]), dtype="float32")
            y = np.zeros((size[0], size[1]), dtype="float32")
            m = np.zeros((size[0], size[1]), dtype="float32")
            n = np.zeros((size[0], size[1]), dtype="float32")
            b = np.zeros((size[0], size[1]), dtype="float32")
            for i in range(10):
                # 在我的笔记本里下面这个要用从cv2.imread(),但是在实验室要用plt.imread()
                img = cv2.imread(filepath + "\\" + fsingal + "_" + "%d" % (i + 1) + ".tif", 0)  # 以灰度图方式读取图片
                # print(filepath + "\\"  + "%d" % (i + 1) + ".tif")
                # img = cv2.imread(filepath + "\\"  + "%d" % (i + 1) + ".tif", 0)
                arr = np.array(img, dtype="float32")
                y = y + arr
                m = m + arr * math.sin(2 * math.pi * (i + 1) / 10)
                n = n + arr * math.cos(2 * math.pi * (i + 1) / 10)
            y = y / 2550.0
            y[y < 0.14] = 0.0
            m[y < 0.14] = 0.0
            n[y < 0.14] = 0.0
            m = m / 255.0
            n = n / 255.0
            b = 5 * np.sqrt(m ** 2 + n ** 2)
            for i in range(10):
                img = cv2.imread(filepath + "\\" + fsingal + "_" + "%d" % (i + 1) + ".tif", 0)  # 以灰度图方式读取图片
                # img = cv2.imread(filepath + "\\" + "%d" % (i + 1) + ".tif", 0)
                arr = np.array(img, dtype="float32")
                arr[y < 0.14] = 0.0
                data_x[i + j * 10 + q * ad, :, :, 0] = arr / 255.0
            for k in range(10):
                data_y[k + j * 10 + q * ad, :, :, 0] = y
                data_b[k + j * 10 + q * ad, :, :, 0] = b
        q = q + 1
    data_x=data_x.transpose(0,3,1,2)
    data_b = data_b.transpose(0, 3, 1, 2)
    return data_x, data_b
def test():
    train_path = r'E:\phase_data\use'
    size=[2056,2452]
    indx=1
    data_x, data_b = load_data(train_path, [size[0], size[1]])

    # data_x=np.load('data_x_test.npy')
    # data_b=np.load('data_b_test.npy')
    # from skimage import io
    a = np.zeros((2056,2452,3),dtype='float32')
    for i in range(3):
        a[:,:,i]=data_b[indx,0,:,:]
    #
    # # a = np.resize(a,[300,400])
    #
    a = cv2.resize(a, (514, 613), )
    io.imsave('test_modulation.png',a)
    # print(a.shape)
    # plt.imshow(a)
    # plt.show()
    # b = data_b[indx, 0, :, :]
    # cv2.resize(b,(514,613),)
    # plt.imshow(data_x[indx,0,:,:],cmap='gray')
    # plt.show()
    # plt.imshow(data_b[indx, 0, :, :], cmap='gray')
    # plt.show()
test()
# a = plt.imread('test_2.png')
# print(a.shape)
# def save():
#     train_path = 'D:/fanluyao/tiaozhidu_yanzheng2_200_480x640'
#     size = [480, 640]
#
#     data_x, data_b = load_data(train_path, [size[0], size[1]])
#     np.save('data_x_test.npy',data_x)
#     np.save('data_b_test.npy',data_b)
# save()
# test()
    # print(data_x.shape)
    # print(data_b.shape)
