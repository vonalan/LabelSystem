# -*- coding: utf-8 -*-
"""
Created on Wed Aug 02 22:21:40 2017

@author: dapengguai
"""


import cv2, os, copy
import pdb
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque


def resize(frame=None, shape=None, scale=1.0):
    '''
    to substitute
    cv2.namedWindow('image', flags=cv2.WINDOW_NORMAL)
    for user experience
    '''
    # (544,960) or
    # (960,544)
    orishape = (frame.shape[0], frame.shape[1])
    objshape = (544,960) if orishape[0] < orishape[1] else (960,544)

    r0 = objshape[0] / float(orishape[0])
    r1 = objshape[1] / float(orishape[1])
    scale = max(r0, r1)

    resizeFrame = frame
    if scale < 1.0:
        objshape = map(lambda x : int(x * scale), orishape)
        resizeFrame = cv2.resize(frame, tuple(reversed(objshape)), interpolation=cv2.INTER_CUBIC)

    return scale, resizeFrame

if __name__ == '__main__':
    imgdir = r'F:\Users\kingdom\Documents\GIT\LabelSystem\VideoLabel-DF\images'
    imglist = os.listdir(imgdir)

    for img in imglist:
        imgpath = os.path.join(imgdir, img)
        origin = cv2.imread(imgpath)
        print 'scale: %4f, shape: %s' % (1.0, origin.shape)
        scale, frame = resize(origin)
        print 'scale: %4f, shape: %s'%(scale, frame.shape)
        cv2.imwrite(imgpath[:-4]+'_resize.png', frame)
        while True:
            cv2.imshow('origin', origin)
            cv2.imshow('resize', frame)

            key = cv2.waitKey()
            if key == 27: break


