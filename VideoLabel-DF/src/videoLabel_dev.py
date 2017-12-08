# -*- coding: utf-8 -*-

import os
import sys
import copy

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

        # TODO: i =  ax + by + cz, b = [0:4], c = [0:4], then i = 12cx + 3cy + cz
        self.alpha = None
        self.curBoxIdx = -1
        self.curVertexIdx = -1
        self.curEdgeIdx = -1

    def update(self, image_dir, image_list, idx):
        image_path = os.path.join(image_dir, image_list[idx])
        self.frame = cv2.imread(image_path)
        # self.bufferFrame = copy.deepcopy(self.frame)
        print self.frame.shape

    def update_alpha(self):
        pass

    def update_frame(self):
        pass

    def call_back_func(self):
        pass

    def label(self, image_dir, xml_dir):
        image_list = os.listdir(image_dir)
        xml_list = os.listdir(xml_dir)
        image_list = [img for img in image_list if img not in xml_list]
        if not len(image_list): return 0
        print image_list

        self.update(image_dir, image_list, 0)
        self.update_alpha()
        self.update_frame()

        cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
        while True:
            # cv2.setMouseCallback('image', self.call_back_func())

            cv2.imshow('image', self.frame)
            # cv2.imshow('alpha', self.alpha)

            key = cv2.waitKey(0)

            if key == 27:
                break
        cv2.destroyAllWindows()

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
    ImageLabel().label(r'../../PoseAnnotation/inputs/images/', r'../../PoseAnnotation/inputs/')