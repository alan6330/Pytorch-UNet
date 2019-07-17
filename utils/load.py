#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os

import numpy as np
from PIL import Image
import cv2
from .utils import resize_and_crop, get_square, normalize, hwc_to_chw

classes = ['background','ai']

# RGB color for each class
colormap = [[255,255,255],[0,0,0]]
def get_ids(dir):
    """Returns a list of the ids in the directory"""
    return (f[:-4] for f in os.listdir(dir))


def split_ids(ids, n=2):
    """Split each id in n, creating n tuples (id, k) for each id"""
    return ((id, i)  for id in ids for i in range(n))


def to_cropped_imgs(ids, dir, suffix):
    """From a list of tuples, returns the correct cropped img"""
    allimg=[]
    for id in ids:
        # im = resize_and_crop(Image.open(dir + id + suffix), scale=scale)
        im =cv2.imread(dir + id + suffix,1)[...,::-1]
        im=hwc_to_chw(im)
        im=normalize(im)
        allimg.append(im)
    return np.array(allimg)

def to_cropped_masks(ids, dir, suffix):
    """From a list of tuples, returns the correct cropped img"""
    allimg=[]
    for id in ids:
        im = cv2.imread(dir + id + suffix,1)[...,::-1]
        cm2lbl = np.zeros(256 ** 3)  # 每个像素点有 0 ~ 255 的选择，RGB 三个通道
        for i, cm in enumerate(colormap):
            cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i  # 建立索引

        def image2label(im):
            data = np.array(im, dtype='int32')
            idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
            return np.array(cm2lbl[idx], dtype='int64')  # 根据索引得到 label 矩阵

        out = image2label(im)
        ##one-hot编码
        one_hot_im=np.zeros((im.shape[0],im.shape[1],len(classes)))
        for i in range(len(classes)):
            one_hot_im[out==i,i]=1
        one_hot_im = hwc_to_chw(one_hot_im)
        allimg.append(one_hot_im)
    return np.array(allimg)

def get_imgs_and_masks(ids, dir_img, dir_mask):
    """Return all the couples (img, mask)"""
    imgs = to_cropped_imgs(ids, dir_img, '.png')
    masks = to_cropped_masks(ids, dir_mask, '.png')

    return zip(imgs, masks)


def get_full_img_and_mask(id, dir_img, dir_mask):
    im = Image.open(dir_img + id + '.jpg')
    mask = Image.open(dir_mask + id + '_mask.gif')
    return np.array(im), np.array(mask)
