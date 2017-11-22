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
    def call_back_func(self):
        pass
    def annotation(self):
        imageList = os.listdir(imageDir)

if __name__ == '__main__':
    imageDir = './images/'
    outputDir = './outputs'
    pa = PoseAnnotation(imageDir, outputDir)
