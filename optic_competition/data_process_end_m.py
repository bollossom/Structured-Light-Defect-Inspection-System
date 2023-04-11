import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series,DataFrame

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

    count_x = 2020# 数据集的图片总数 160
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
def test1():
    train_path = 'D:/fanluyao/optics_compeletition'
    size=[480,640]
    data_x, data_b = load_data(train_path, [size[0], size[1]])
    # use_b=np.zeros((202,1,480,640),dtype=np.uint8)
    # for i in range(202):
    #     use_b[i,:,:,:]=data_b[10*i,:,:,:]
    # print(use_b.shape)
    # plt.imshow(use_b[200, 0, :,:],cmap='gray')
    # plt.show()
    # indx=random.randint(0,110)
    for indx in range(202):
        # indx=2019
        # a=os.listdir('D:/fanluyao/optics_compeletition')
        # print(len(a))
        # data_x,data_b=load_data(train_path,[size[0],size[1]])
        # np.save('data_x.npy', data_x)
        # np.save('data_b.npy', data_b)
        # data_x=np.load('data_x.npy')
        # data_b=np.load('data_b.npy')
        # print(data_x.shape)
        # data_b[10 * i, :, :, :]
        b=data_b[10 * indx, 0, :, :]
        # x=data_x[indx, 0, :,:]

        io.imsave('D:/optics_competition/data_end/00' + "%d" % (indx+1) + '.png', b)
        # plt.subplot(1,2,1)
        # plt.imshow(x, cmap='gray')
        # plt.subplot(1,2,2)
        # plt.imshow(b,cmap='gray')
        # plt.show()
# test1()
def test():
    train_path = 'D:/fanluyao/optics_compeletition'
    size = [480, 640]
    indx=1200
    data_x, data_b = load_data(train_path, [size[0], size[1]])
    b=data_b[indx,0,:,:]
    plt.imshow(b,cmap='gray')
    plt.show()
    io.imsave('D:/optics_competition/00' + "%d" % (indx + 1) + '.tif', b)
    print(b)
test()

# def save():
#     train_path = 'D:/fanluyao/new'
#     size = [480, 640]
#
#     data_x, data_b = load_data(train_path, [size[0], size[1]])
#     np.save('data_x.npy',data_x)
#     np.save('data_b.npy',data_b)
# save()
# test()
    # print(data_x.shape)
    # print(data_b.shape)
