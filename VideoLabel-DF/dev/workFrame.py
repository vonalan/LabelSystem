# -*- coding: utf-8 -*-

import numpy as np 
import cv2

import copy


class Coordinate(object): 
    '''2-D Coordinates'''
    def __init__(self, x=0, y=0): 
        self.x = x 
        self.y = y 


class Rectagle(object):
    '''
    Rectagle
    [[xmin, ymin], [xmax, ymax]]
    '''
    def __init__(self, xmin=0, ymin=0, xmax=0, ymax=0):
        self.min = Coordinate(x=xmin, y=ymin)
        self.max = Coordinate(x=xmax, y=ymax)

        # self.xmin = 0
        # self.ymin = 0
        # self.xmax = 0
        # self.ymax = 0

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
    def __int__(self, label='', xmin=0, ymin=0, xmax=0, ymax=0):
        self.label = label
        self.mainRect = Rectagle(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax) # [(xmin, ymin), (xmax, ymax)]
        self.subRects = self._get_sub_rects_() # 4*[(xmin, ymin), (xmax, ymax)]

    def _get_sub_rects_(self, thick=None):
        subRects = []
        for pt in self.mainRect.points: 
            xmin = pt.x - thick 
            ymin = pt.y - thick 
            xmax = pt.x + thick 
            ymax = pt.y + thick 
            rect = Rectagle(xmin, ymin, xmax, ymax)
            subRects.append(rect)
        return subRects

class WorkFrame(object):
    def __init__(self, name=''):
        self.name = name 
        self.frame = self._initialize_frame_()
        self.bufferframe = copy.deepcopy(self.frame)
        self.boxes = None
        self.boxImages = None
    
    def show_frame(self): 
        cv2.namedWindow(self.name,cv2.WINDOW_AUTOSIZE)
        while(True): 
            cv2.imshow(self.name, self.frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    def _initialize_frame_(self): 
        frame = cv2.imread(self.name)
        return frame 

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
    def __init__(self, nameList=[]): 
        self.nameList = nameList
        self.wkFrames = _initialize_frame_()
        self.curFrame = WorkFrame()
    
    def _initialize_flow_(self): 
        flow = [] 
        for idx, name in self.nameList: 
            frame = WorkFrame(name)
            flow.append(frame)
        return flow 
    
    def update(self): 
        '''interoplation'''
        pass 
    
    def flush(self): 
        '''flush buffer'''
        pass 

    

if __name__ == "__main__":
    wf = WorkFrame('zhangxuan.jpg')
    print(wf.frame.shape)
    wf.show_frame()