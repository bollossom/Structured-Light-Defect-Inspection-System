# import numpy as np
# from skimage import io
# import matplotlib.pyplot as plt
# input=np.load('data_x_test_new.npy')
# label=np.load('data_b_test_new.npy')
# print(input.shape,label.shape)
# # print()
# # output=io.imread('output.tif')
# # label=io.imread('label.tif')
# indx=3
# x=input[indx,0,:,:]
# y=label[indx,0,:,:]
# plt.subplot(1,2,1)
# plt.imshow(x, cmap='gray')
# plt.subplot(1,2,2)
# plt.imshow(y)
# plt.show()
# print(x)
# io.imsave('output.tif',output)
# io.imsave('label.tif',label)

from skimage import data,filters
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
# img = plt.imread('D:/optics_competition/data_end/0010.png')
import cv2
import numpy as np
import matplotlib.pyplot as plt


def fix_threshold(img, thresh, maxval=255):
    return np.where(((img > thresh) & (img < maxval)), 255, 0)


# img =cv2.imread('D:/optics_competition/data_end/00116.png')
# import cv2
#
# path = 'D:/optics_competition/data_end/001.png'
# src = cv2.imread(path,cv2.IMREAD_UNCHANGED) # 读取要进行闭运算的图像
# close_img = cv2.morphologyEx(src,cv2.MORPH_CLOSE,kernel=np.ones((5,5),np.uint8))
# cv2.imshow('original_close',src)
# cv2.imshow('close_img',close_img)
# cv2.moveWindow("original_close", 1000, 400)                  # 移动窗口位置
# cv2.imshow('img',src-close_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
import os
def load_data2(data_path, size):
    image_path = []
    label_path = []
    signal = os.listdir(data_path)  # 每个文件夹的名称（集合）
    count_x = 840#输入的图片个数
    data_x = np.empty((count_x, 1,size[0], size[1]), dtype="float32")
    data_y = np.empty((count_x, 1,size[0], size[1]), dtype="float32")
    data_m = np.empty((count_x, 1,size[0], size[1]), dtype="float32")
    data_n = np.empty((count_x, 1,size[0], size[1]), dtype="float32")
    q=0
    for fsingal in signal:
        filepath = data_path + "\\" + fsingal  # 图片所在文件夹的路径（单）
        # print(filepath)
        # print(signal)
        # print(fsingal)
        for j in range(12):
            x = np.zeros((size[0], size[1]), dtype="float32")
            y = np.zeros((size[0], size[1]), dtype="float32")
            m = np.zeros((size[0], size[1]), dtype="float32")
            n = np.zeros((size[0], size[1]), dtype="float32")
            for i in range(10):
                img = plt.imread(filepath + "\\" + fsingal + "_" + "%d" % (i + 1) + "_" + "%d" % (
                            j + 1) + ".tif")  # 以灰度图方式读取图片
                arr = np.array(img, dtype="float32")
                y = y + arr
                # print(filepath + "\\" + fsingal + "_" + "%d" % (i + 1) + "_" + "%d" % (
                #             j + 1) + ".tif")
                m = m + arr * math.sin(2 * math.pi * (i + 1) / 10)
                n = n + arr * math.cos(2 * math.pi * (i + 1) / 10)

            for i in range(10):
                img = plt.imread(filepath + "\\" + fsingal + "_" + "%d" % (i + 1) + "_" + "%d" % (
                            j + 1) + ".tif")  # 以灰度图方式读取图片
                arr = np.array(img, dtype="float32")
                arr[y < 0.08] = 0.0
                data_x[i + j * 10 + q * 120, 0, :, :] = arr/255
            y[y < 0.1] = 0.0
            # m[y < 0.2] = 1.0
            m[y < 0.2] = 0.0
            # m[y < 1] = 0.0
            n[y < 0.2] = 1.0
            for k in range(10):
                data_y[k + j * 10 + q * 120, 0, :, :] = y/255
                # print(k + j * 10+120*(q))
                data_m[k + j * 10 + q * 120, 0, :, :] = m/255
                data_n[k + j * 10 + q * 120, 0, :, :] = n/255
                # print(data_m)
        q=q+1
    return data_x,data_y,data_m, data_n

path='D:\\data new\\sumsung_04\\x'
data_x,data_y,data_m, data_n=load_data2(path,[224,224])
print(data_y.shape)
plt.imshow(data_y[0,0,:,:],cmap='gray')
plt.show()


