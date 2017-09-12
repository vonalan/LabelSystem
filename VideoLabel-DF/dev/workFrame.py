# -*- coding: utf-8 -*-

import numpy as np 
import cv2 


def coordinatesWrapper(cls):
    return [(cls.xmin, cls.ymin), (cls.xmax, cls.ymax)]

@coordinatesWrapper
class Rectagle(object):
    '''
    Rectagle
    [(xmin, ymin), (xmax, ymax)]
    '''
    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = 0
        self.ymin = 0
        self.xmax = 0
        self.ymax = 0

        self.coordinates = None
        self.points = None

    def get_coordinates(self):
        return [[self.xmin,self.ymin], [self.xmax,self.ymax]]

    def get_points(self):
        return [[self.xmin,self.ymin], [self.xmax,self.ymax],
                [self.xmax,self.ymin], [self.xmin,self.ymax]]

    def update(self):
        pass


class BBox(object):
    '''
    Bounding box
    '''
    def __int__(self):
        self.label = '0'
        self.mainRect = Rectagle() # [(xmin, ymin), (xmax, ymax)]
        self.subRects = self._get_sub_rects_()

    def _get_sub_rects_(self):
        pass

class WorkFrame(object):
    def __init__(self):
        self.name = None
        self.frame = None
        self.boxes = None
        self.boxImages = None

    def update_frame(self):
        pass

    def flush_frame(self):
        pass

    def draw_rect(self, event, x, y, flags, param):
        '''mouse'''
        pass

    def label(self):
        '''keyboard'''
        pass

class FrameFlow(object): 
    def __init__(self): 
        self.wkFrames = None
        self.curFrame = None

        self.rightClick = 0


if __name__ == "__main__":
    rect = Rectagle()
    print(rect)