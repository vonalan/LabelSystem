# -*- coding: utf-8 -*-

"""
https://github.com/suriyasingh/pose-annotation-tool
"""
import os
import copy

import cv2

NUM_JOINTS = 15
JOINT_TEMPLATE = ['', 0, -1, -1]
JOINTS_TEMPLATE = NUM_JOINTS * JOINT_TEMPLATE

JOINT_NAMES = [
    "Right ankle",
    "Right knee",
    "Right hip",
    "Left hip",
    "Left knee",
    "Left ankle",
    "Right wrist",
    "Right elbow",
    "Right shoulder",
    "Left shoulder",
    "Left elbow",
    "Left wrist",
    "Neck",
    "Head top"]

class PoseAnnotation(object):
    def __init__(self, imageDir, outputDir):
        self.frame = None
        self.bufferFrame = None
        self.joints = []
        self.bufferJoints = []

        self.imageDir = imageDir
        self.outputDir = outputDir

        self.imageList = []
    def update(self, index):
        imagePath = os.path.join(self.imageDir, self.imageList[index])
        self.frame = cv2.imread(imagePath)
        self.bufferFrame = copy.copy(self.frame)
        self.joints = []
        self.bufferJoints = []
    def update_frame(self):
        pass
    def call_back_func(self, event, x, y, flags, param):
        pass
    def annotation(self):
        self.imageList = os.listdir(imageDir)
        cidx = 0
        self.update(cidx)

        cv2.namedWindow('image', flags=cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('image', self.call_back_func)
        while True:
            cv2.imshow('image', self.frame)
            key = cv2.waitKey(20)

            if key == 27:
                break



if __name__ == '__main__':
    imageDir = '../images/'
    outputDir = '../outputs'
    pa = PoseAnnotation(imageDir, outputDir)
    pa.annotation()
