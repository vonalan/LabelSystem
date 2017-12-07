# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import cv2

bbox = [-1, [[-1,-1], [-1, -1]]]
boxes = bbox * 3

class ImageLabel(object):
    def __init__(self):
        self.frame = None
        self.bufferFrame = None

        self.boxes = []
        self.bufferBoxes = []

        self.curBoxIdx = -1
        self.curVertexIdx = -1
        self.curEdgeIdx = -1

    def update(self, image_list, idx):
        pass

    def update_alpha(self):
        pass

    def update_frame(self):
        pass

    def call_back_func(self):
        pass

    def label(self, image_dir):
        image_list = os.listdir(image_dir)
        self.update(image_list, 0)

class VideoLabel(object):
    def __init__(self):
        pass
    def extract(self):
        pass
    def label(self):
        pass

def main():
    pass

if __name__ == '__main__':
    pass