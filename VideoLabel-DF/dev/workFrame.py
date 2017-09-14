# -*- coding: utf-8 -*-

import numpy as np 
import cv2

import copy


class Coordinate(object): 
    '''2-D Coordinates'''
    def __init__(self, x=0, y=0): 
        self.x = x 
        self.y = y
    def get_coordinate(self): 
        return [self.x, self.y]


class Rectagle(object):
    '''
    Rectagle
    [[xmin, ymin], [xmax, ymax]]
    '''
    def __init__(self, xmin=0, ymin=0, xmax=0, ymax=0):
        self.min = Coordinate(x=xmin, y=ymin)
        self.max = Coordinate(x=xmax, y=ymax)

    def get_points(self):
        points = []
        points.append(self.min)
        points.append(Coordinate(x=self.min.x, y=self.max.y))
        points.append(Coordinate(x=self.max.x, y=self.min.y))
        points.append(self.max)
        return points


class BBox(object):
    '''
    Bounding box
    '''
    def __int__(self, label='', xmin=0, ymin=0, xmax=0, ymax=0):
        self.label = label
        self.mainRect = Rectagle(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax) # [(xmin, ymin), (xmax, ymax)]
        self.subRects = self.get_sub_rects() # 4*[(xmin, ymin), (xmax, ymax)]

    def get_sub_rects(self, thick=None):
        subRects = []
        for pt in self.mainRect.get_points():
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
        self.checked = False
        self.frame = self._initialize_frame_()
        self.bufferframe = copy.deepcopy(self.frame)
        self.shape = self.frame.shape
        self.boxes = [BBox()]
        self.boxImage = np.zeros((self.shape[0],self.shape[1])) - 1
    
    def show_frame(self): 
        cv2.namedWindow(self.name,cv2.WINDOW_AUTOSIZE)
        while(True): 
            cv2.imshow(self.name, self.frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    def _initialize_frame_(self, nameList=[]):
        frame = cv2.imread(self.name)
        return frame 

    def update_box_imgs(self, thick=None): 
        th = self.thick
        for i, box in enumerate(self.boxes):
            rect = box.mainRect
            self.boxImage[rect.min.y:rect.max.y, rect.min.x:rect.max.x] = i * 3 # [y1:y2,x1:x2]
            for p in box.get_sub_rects(): 
                self.boxImage[p[1] - th: p[1] + th, p[0] - th: p[0] + th] = i * 3 + 1

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
        self.wkFrames = self._initialize_flow_()
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