import os
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage import io
def load_data2(data_path):
    image_path = []
    label_path = []
    size = [224, 224]
    signal = os.listdir(data_path)  # 每个文件夹的名称（集合）
    count_x = 2160#输入的图片个数14*120
    data_x = np.empty((count_x, 1,size[0], size[1]), dtype="float32")
    data_y = np.empty((count_x, 1,size[0], size[1]), dtype="float32")
    data_m = np.empty((count_x, 1,size[0], size[1]), dtype="float32")
    data_n = np.empty((count_x, 1,size[0], size[1]), dtype="float32")
    data_b = np.empty((count_x,1, size[0], size[1]), dtype="float32")
    q=0
    # random.shuffle(data_path)
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
            b = np.zeros((size[0], size[1]), dtype="float32")
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
            y = y / 2550.0
            y[y < 0.14] = 0.0
            m[y < 0.14] = 0.0
            n[y < 0.14] = 0.0
            m = m / 255.0
            n = n / 255.0
            b = 5 * np.sqrt(m ** 2 + n ** 2)
            for k in range(10):
                data_y[k + j * 10 + q * 120, 0, :, :] = y/255
                # print(k + j * 10+120*(q))
                data_m[k + j * 10 + q * 120, 0, :, :] = m/255
                data_n[k + j * 10 + q * 120, 0, :, :] = n/255
                data_b[k + j * 10 + q * 120, 0, :, :] = b/255
                # print(data_m)
        q=q+1
    return data_x,data_y,data_m, data_n,data_b
path='D:/data_2_end'
size=[224,224]
data_x,data_y,data_m, data_n,data_b=load_data2(data_path=path)
# print(data_b.shape)
data_b_end=np.zeros((216,1,224,224),dtype='float32')
data_x_end=np.zeros((216,1,224,224),dtype='float32')
for i in range(216):
    data_b_end[i,0,:,:]=data_b[10 * i, 0, :, :]
    b=data_b_end[i,0,:,:]
    # plt.imsave('D:/data_b/00' + "%d" % (i + 1) + '.png',b,cmap='gray')
    data_x_end[i, 0, :, :] = data_x[10 * i, 0, :, :]
    x = data_x_end[i, 0, :, :]

    # plt.imsave('D:/data_x/00' + "%d" % (i + 1) + '.png', x, cmap='gray')
np.save('data_x.npy',data_x_end)

# for i in range(230,240):
#     plt.imshow(data_b_end[i,0,:,:],cmap='gray')
#     plt.show()

# for indx in range(2390,2400):
#     # print(indx)
# # indx=2770
#     plt.imshow(data_b[indx,0,:,:],cmap='gray')
#     plt.show()