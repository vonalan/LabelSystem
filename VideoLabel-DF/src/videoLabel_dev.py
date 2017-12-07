# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import cv2

bbox = [-1, [[-1,-1], [-1, -1]]]
boxes = bbox * 3

class VideoLabel(object):
    def __init__(self):
        self.frame = None
        self.bufferFrame = None

        self.boxes = []
        self.bufferBoxes = []

        self.curBoxIdx = -1
    def label(self):
        pass

def extract_frame():
    pass

if __name__ == '__main__':
    video_dir = '.'
    for video in os.listdir(video_dir):
        extract_frame()
        vl = VideoLabel()