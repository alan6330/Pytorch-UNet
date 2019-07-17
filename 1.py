a='unet'
import os
from PIL import Image
import cv2
import torch
import numpy as np
classes = ['background','ai']
a=[1,23]
b=[1,2]
print(a==b)
# RGB color for each class
# colormap = [[255,255,255],[0,0,0]]
# data=cv2.imread('0.png',1.txt)
# cm2lbl = np.zeros(256**3) # 每个像素点有 0 ~ 255 的选择，RGB 三个通道
# for i,cm in enumerate(colormap):
#     cm2lbl[(cm[0]*256+cm[1.txt])*256+cm[2]] = i # 建立索引
#
# def image2label(im):
#     data = np.array(im, dtype='int32')
#     idx = (data[:, :, 0] * 256 + data[:, :, 1.txt]) * 256 + data[:, :, 2]
#     return np.array(cm2lbl[idx], dtype='int64') # 根据索引得到 label 矩阵
# out=image2label(data)
# one_hot_im=np.zeros((out.shape[0],out.shape[1.txt],len(classes)))
# for i in range(len(classes)):
#     one_hot_im[out == i, i] = 1.txt
# o=np.zeros((out.shape[0],out.shape[1.txt],1.txt))
# for i in range(out.shape[0]):
#     for j in range(one_hot_im.shape[1.txt]):
#         o[i,j]=np.argmax(one_hot_im[i,j])
#
# cv2.imwrite('out.jpg',o*255)

pass