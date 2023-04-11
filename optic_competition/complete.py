import matplotlib.pyplot as plt
import numpy as np
from skimage import io
pred=np.load(r'C:\Users\86153\Documents\Tencent Files\2223453260\FileRecv\end3.npy')
print(pred.shape)
pred = pred[0,0,:,:]
plt.imshow(pred,cmap='gray')
plt.show()
gt = plt.imread(r'D:\opencv-python\TV_Conv\deep-image-prior-master\data\sr\test.png')
plt.imshow(gt,cmap='gray')
plt.show()
plt.imshow(gt-pred,cmap='gray')
plt.show()
print(pred)
print(sum(sum(abs(gt-pred)))/(640*480))
# io.imsave('pre.png',prediction)
# plt.imshow(prediction,cmap='gray')
# plt.show()
