# -*- coding: utf-8 -*-

import numpy as np 
import cv2 

class Rect(object):
    '''
    Rectagle
    '''
    def __init__(self):
        self.xmin = None
        self.ymin = None
        self.xmax = None
        self.ymax = None

class BBox(object):
    '''
    Bounding box
    '''
    def __int__(self):
        self.label = None 
        self.xmin = None 
        self.ymin = None
        self.xmax = None
        self.ymax = None

class WorkFrame(object):
    def __init__(self):
        self.name = None
        self.frame = None
        self.boxes = None

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
    
    def draw_rect(self, event, x, y, flags, params): 
        if self.rightClick == 0: 
            if event == cv2.EVENT_MOUSEMOVE:
                print(x,y)
    
    def label(self):
        cv2.namedWindow('image')
        cv2.setMouseCallback("image", self.draw_rect)
        cv2.destroyAllWindows()