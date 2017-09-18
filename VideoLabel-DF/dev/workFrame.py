# -*- coding: utf-8 -*-

import numpy as np
import cv2

import copy


class Coordinate(object): 
    '''2-D Coordinates'''
    def __init__(self, x=-1, y=-1):
        self.x = x 
        self.y = y

    def to_tensor(self):
        '''1-D tensor'''
        tensor = [self.x, self.y]
        return tensor

    def from_tensor(self, tensor):
        self.x = tensor[0]
        self.y = tensor[1]

    def update(self,x=-1,y=-1):
        self.x = x
        self.y = y


class Rectagle(object):
    '''
    Rectagle
    [[xmin, ymin], [xmax, ymax]]
    '''
    def __init__(self, xmin=-1, ymin=-1, xmax=-1, ymax=-1, gap=25):
        self.min = Coordinate(x=xmin, y=ymin)
        self.max = Coordinate(x=xmax, y=ymax)
        self.gap = gap

    def to_tensor(self):
        '''2-D tensor'''
        tensor = [self.min.to_tensor(), self.max.to_tensor()]
        return tensor

    def from_tensor(self, tensor):
        self.min.from_tensor(tensor[0])
        self.max.from_tensor(tensor[1])

    def get_points(self):
        '''
        [[0,0],[0,1],[1,0],[1,1]]
        '''
        points = []
        points.append(self.min)
        points.append(Coordinate(x=self.min.x, y=self.max.y))
        points.append(Coordinate(x=self.max.x, y=self.min.y))
        points.append(self.max)
        return points

    def move_vertex(self):
        pass

    def move_edge(self):
        pass

    def move_all(self, deltaX=0, deltaY=0):
        self.min.x += deltaX
        self.min.y += deltaY
        self.max.x += deltaX
        self.max.y += deltaY

    def update(self, xmin=0, ymin=0, xmax=0, ymax=0):
        self.min.x = xmin
        self.min.y = ymin
        self.max.x = xmax
        self.max.y = ymax

    def check(self, shape=None):
        '''
        min.x >= 0 && min.x <= max.x
        min.y >= 0 && min.y <= max.y
        '''
        # swap min and max if neccessary
        if self.min.x > self.max.x:
            self.min.x, self.max.x = self.max.x, self.min.x
        if self.min.y > self.max.y:
            self.min.y, self.max.y = self.max.y, self.min.y

        # we know that self.max.x is 0 when self.mix.x lt 0
        if self.min.x < 0:
            self.min.x = 0
            self.max.x = self.gap
        # we know that self.max.x is 0 when self.mix.x lt 0
        if self.min.y < 0:
            self.min.y = 0
            self.max.y = self.gap
        # we know that self.min.x is shape[1] when self.max.x gt shape[1]
        if self.max.x > shape[1]:
            self.max.x = shape[1]
            self.min.x = shape[1] - self.gap
        # we know that self.min.y is shape[0] when self.max.y gt shape[0]
        if self.max.y > shape[0]:
            self.max.y = shape[0]
            self.min.y = shape[0] - self.gap


class BBox(object):
    '''
    Bounding box
    '''
    def __init__(self, xmin=-1, ymin=-1, xmax=-1, ymax=-1, label='', color=(0,0,0)):
        self.label = label
        self.color = color
        self.mainRect = Rectagle(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax) # [(xmin, ymin), (xmax, ymax)]
        # self.subRects = self.get_sub_rects() # 4*[(xmin, ymin), (xmax, ymax)]

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
        self.boxes = []
        self.curBox = BBox()
        self.boxImage = np.zeros((self.shape[0],self.shape[1])) - 1
        self.minBox = 25

    def to_tensor(self):
        '''3-d tensor'''
        tensor = []
        for i, box in enumerate(self.boxes):
            tensor.append(box.mainRect.to_tensor())
        return tensor

    def from_tensor(self, tensor):
        for i, (box, t)in enumerate(zip(self.boxes, tensor)):
            box.mainRect.from_tensor(t)

    def show(self):
        cv2.namedWindow(self.name,cv2.WINDOW_AUTOSIZE)
        while(True): 
            cv2.imshow(self.name, self.frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    def _initialize_frame_(self, nameList=None):
        frame = cv2.imread(self.name)
        return frame

class FrameBlock(object):
    def __init__(self, nameList=None):
        self.nameList = nameList
        self.wkFrames = self._initialize_frames_()
        self.length = len(self.wkFrames)

    def _initialize_frames_(self):
        wkFrames = []
        for idx, name in self.nameList: 
            frame = WorkFrame(name)
            wkFrames.append(frame)
        return wkFrames
    
    def update_old(self, sFrame=None, eFrame=None, steps=1):
        '''interoplation'''
        sFrame = self.wkFrames[0]
        eFrame = self.wkFrames[-1]
        steps = self.length
        assert len(sFrame.boxes) == len(eFrame.boxes)

        dFrame = copy.deepcopy(eFrame)
        for dbox, sbox, ebox in zip(dFrame.boxes, sFrame.boxes, eFrame.boxes):
            drect, srect, erect = dbox.mainRect, sbox.mainRect, ebox.mainRect
            drect.min.x = (erect.min.x - srect.min.x)/float(steps)
            drect.min.y = (erect.min.y - srect.min.y)/float(steps)
            drect.max.x = (erect.max.x - srect.max.x)/float(steps)
            drect.max.y = (erect.max.y - srect.max.y)/float(steps)

        for steps, cFrame in enumerate(self.wkFrames):
            if steps == 0 or steps == self.length - 1: continue
            for dbox, sbox, ebox in zip(dFrame.boxes, sFrame.boxes, eFrame.boxes):
                drect, srect, erect = dbox.mainRect, sbox.mainRect, ebox.mainRect
                erect.min.x = int(srect.min.x + drect.min.x * steps)
                erect.min.y = int(srect.min.y + drect.min.y * steps)
                erect.max.x = int(srect.max.x + drect.max.x * steps)
                erect.max.y = int(srect.max.y + drect.max.y * steps)

    def update(self):
        sArray = np.array(self.wkFrames[0].to_tensor())
        eArray = np.array(self.wkFrames[-1].to_tensor())
        dArray = (eArray - sArray)/float(self.length)
        
        for steps, cFrame in enumerate(self.wkFrames):
            if steps == 0 or steps == self.length - 1: continue
            cArray = sArray + dArray * steps
            cFrame.from_tensor(cArray.tolist())

    def flush(self): 
        '''flush buffer'''
        for idx, frame in enumerate(self.wkFrames):
            frame.draw_static()
        pass

    def check(self):
        flags = []
        for idx, frame in enumerate(self.wkFrames):
            if frame.checked == False:
                flags.append(idx)
        return flags

if __name__ == "__main__":
    wf = WorkFrame('../images/test.png')
    box = BBox(xmin=0,ymin=0,xmax=30,ymax=30)
    wf.boxes.append(box)
    wf.boxes.append(box)
    wf.boxes.append(box)
    t = wf.to_tensor()

    t = np.array(t)
    print t.shape
    print (t)

    t = t.tolist()
    for i in t:
        print i