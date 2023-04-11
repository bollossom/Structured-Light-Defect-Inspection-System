import torch
import numpy as np
from torchvision import transforms as trans
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.utils.data as Data
from Mix_Unet import Mix_Unet
from depthwise_Unet import IUnet
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
    input_train= np.load('data_x.npy')
    label_train = np.load('data_b.npy')
    input_test = np.load('data_x_test.npy')
    label_test = np.load('data_b_test.npy')
    return input_train,label_train,input_test,label_test
input_train,label_train,input_test,label_test=load()
print(input_train.shape)
print(input_test.shape)
train_set = MyDataset(input_train,label_train)
test_set=MyDataset(input_test,label_test)
writer = SummaryWriter(comment='test_your_comment', filename_suffix="_test_your_filename_suffix")
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE='cpu'
model=IUnet().to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(),lr=0.0001,
                             weight_decay=1e-05)
BATCH_SIZE_train=2
train_loader=Data.DataLoader(train_set, batch_size=BATCH_SIZE_train, shuffle=True)
test_loader=Data.DataLoader(test_set, batch_size=BATCH_SIZE_train, shuffle=True)
n_epochs=350

def train():

    model.train()
    for input , label in train_loader:
        # print(input.shape)
        input, label = input.to(DEVICE), label.to(DEVICE)

        # print(my.shape)
        # print(sy.shape)
        output=model(input)


        optimizer.zero_grad()

        loss=F.mse_loss(output,label)

        loss.backward()
        optimizer.step()

        mae_loss = F.l1_loss(output, label)
        writer.add_scalars("train_loss", {"Train": mae_loss.item()}, epoch)
def test():
    model.eval()
    test_loss=0

    with torch.no_grad():
        for input, label in test_loader:
            # print(input.shape)
            input, label = input.to(DEVICE), label.to(DEVICE)
            output = model(input)

            mae_loss=F.l1_loss(output, label)
            writer.add_scalars("test_loss", {"Test": mae_loss.item()}, epoch)
# print(str(1) + '_hrnet.pt')
for epoch in range(1, n_epochs + 1):
    train()
    test()
    print('epoch',epoch)
    if epoch%10 == 0:
        model_out_path =  str(epoch) + 'UNET.pt'
        torch.save(model.state_dict(), model_out_path)
# tensorboard --logdir=D:\opencv-python\optic_competition\runs\Aug13_14-47-47_LAPTOP-85DL9FARtest_your_comment
