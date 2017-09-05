# -*- coding: utf-8 -*-

import numpy as np
import cv2

import os
import sys

import videoReader as VR
import videoLabel as VL


videoDir = r'F:\Users\Kingdom\Desktop\LabelSystem\VideoLabel-DF\videos' # 视频文件夹地址
imageDir = r'F:\Users\Kingdom\Desktop\LabelSystem\VideoLabel-DF\images' # 不用设置
outputDir = r'F:\Users\Kingdom\Desktop\LabelSystem\VideoLabel-DF\outputs' # images和xmls输出地址
labelName = r'.\labels.txt'

'''settings'''
sample_factor = 6  # 每6帧抽取一帧
mini_batch_size = 12  # 每次载入的帧数
'''settings'''

vr = VR.VideoReader(sample_factor=sample_factor, mini_batch_size=mini_batch_size)

videoList = os.listdir(videoDir)
for video in videoList:
    videoPath = os.path.join(videoDir, video)
    for batch, frameList in enumerate(vr.read(videoPath=videoPath)):
        vl = VL.VideoLabel(videoDir, imageDir, labelName, outputDir)
        vl.length = 5
        vl.mini_batch_size = mini_batch_size
        vl.video = video  ###
        vl.frameList = frameList  ###

        vl.update_outputDir(video)
        vl.labelling(batch)