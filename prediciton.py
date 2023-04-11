import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import math
import cv2
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.utils.data as Data
from Mix_Unet import Mix_Unet
class MyDataset(Dataset):
    def __init__(self,a,b):
        # a=np.load('D:\code_data\input_train.npy')
        # b=np.load('D:\code_data\label_train.npy')
        a=np.array(a,dtype='float32')
        b = np.array(b, dtype='float32')
        txt_data=np.concatenate((a,b),axis=1)
        # txt_data = np.loadtxt('1.txt', delimiter=',')
        # a=np.array(txt_data[:, :2],dtype='float32')
        # b = np.array(txt_data[:, 2], dtype='float32')
        self._x = torch.tensor(a)
        self._y = torch.tensor(b)
        self._len = len(txt_data)
    def __getitem__(self, item):
        return self._x[item], self._y[item]

    def __len__(self):
        return self._len


def load():
    input=np.load('data_x_test.npy')
    label=np.load('data_b_test.npy')
    return input,label
# save()
input,label=load()
test_set = MyDataset(input,label)
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=Mix_Unet().to(DEVICE)
test_loader=Data.DataLoader(test_set, batch_size=1, shuffle=False)

def prediction():
    path='best_MIX_unet.pt'
    model_CKPT = torch.load(path)
    model.load_state_dict(model_CKPT)
    model.eval()
    i=0
    for input, label in test_loader:
        # print(input.shape)
        input, label = input.to(DEVICE), label.to(DEVICE)
        output = model(input)
        # print(1)
        loss = F.l1_loss(output, label)
        i = i + 1
        # plot(output,label,loss.item(),i)
        np.save('20.npy',output[20, 0, :,:].detach().cpu().numpy())
        np.save('30.npy',output[30, 0, :, :].detach().cpu().numpy())
        np.save('40.npy',output[40, 0, :, :].detach().cpu().numpy())

        print('loss',loss.item())

def plot(output,label,loss,i):
    indx=0
    GT=label[indx, 0, :,:].detach().cpu().numpy()
    output = output[indx, 0, :,:].detach().cpu().numpy()

    plt.suptitle('loss.{}'.format(loss))
    plt.subplot(1, 2, 1)
    plt.title('label')
    plt.imshow(GT, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title('output')
    plt.imshow(output, cmap='gray')
    plt.show()
prediction()
# def test():
#     test_path = 'D:/fanluyao/tiaozhidu_xunlian2_200_480x640'