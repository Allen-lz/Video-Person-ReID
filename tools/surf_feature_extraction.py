#!/usr/bin/env python3
# -*- coding:utf-8 -*-
u'''
Created on 2020年1月3日
@author: luzeng
'''
from pylab import *
import os
import cv2
import numpy as np
from multiprocessing import Pool
from sklearn.decomposition import PCA

# 执行序号3
# print('cv version: ', cv2.__version__)
# 面向西安杨森会议数据集
def bgr_rgb(img):
    (r, g, b) = cv2.split(img)
    return cv2.merge([b, g, r])


def sift_detect(img, detector='surf'):
    if detector.startswith('si'):
        # print("sift detector......")
        sift = cv2.xfeatures2d.SURF_create()
    else:
        # print("surf detector......")
        sift = cv2.xfeatures2d.SURF_create()
    # find the keypoints and descriptors with SIFT
    kps, des = sift.detectAndCompute(img, None)
    # kps = np.float32([kp.pt for kp in kps])
    # return np.hstack((np.array(kps), np.array(des)))
    return np.array(des)

# def write_f_2_npy(feature_name, sift_f):
#     # 排除没有任何特征的图片
#     if len(sift_f.shape) > 1:
#         np.save(feature_name, sift_f)
#     else:
#         np.save(feature_name, np.array([0]))


def write_f_2_txt(filename, sift_f):
    # 排除没有任何特征的图片
    if len(sift_f.shape) > 1:
        savetxt(filename, sift_f)
    else:
        savetxt(filename, np.array([0]))

def sift2txt(image, feature_name):
    # 获得sift特征
    sift_f = sift_detect(image)

    # =====通过pca将sift_f中的特征融合成一个64维度的特征向量=====
    if len(sift_f.shape) > 1:
        pca = PCA(n_components=1)
        pca.fit(sift_f.T)
        newX = pca.fit_transform(sift_f.T)
        sift_f = newX.T
    # =====================================================
    write_f_2_txt(feature_name, sift_f)
    return sift_f

def get_img_list(_dir, img_list):
    items = os.listdir(_dir)
    for item in items:
        path = os.path.join(_dir, item)
        if os.path.isdir(path):
            get_img_list(path, img_list)
        else:
            if (os.path.basename(path).lower().endswith(('.png', '.jpg', '.jpeg'))):
                img_list.append(path)

if __name__ == "__main__":
    # load image
    # 绝对路径
    pool = Pool(32)
    results = list()
    merge_f = list()

    root = '/home/dm/datasets/Reid/DukeMTMC-VideoReID/'
    sift_root = '/home/dm/datasets/Reid/duke_surf_features'

    if not os.path.exists(sift_root):
        os.makedirs(sift_root)
    img_list = []
    get_img_list(root, img_list)

    for index, img_path in enumerate(img_list):

        image = cv2.imread(img_path)
        image = bgr_rgb(image)

        name = img_path.replace(root, "")

        feature_dir = os.path.join(sift_root, name.replace(os.path.basename(name), ""))
        if not os.path.exists(feature_dir):
            os.makedirs(feature_dir)

        feature_path = os.path.join(feature_dir, os.path.basename(name).split(".")[0] + ".txt")


        results.append(index)
        res = pool.apply_async(
                sift2txt,
                args=(image, feature_path))
        # sleep(0.02)
        sift_f = res.get()

        if index == 0:
            pass
        else:
            if len(sift_f.shape) > 1:
                pass
            # 排除没有任何特征的图片
            elif len(sift_f.shape) == 1:
                print('no sift')
                pass

    pool.close()
    pool.join()




