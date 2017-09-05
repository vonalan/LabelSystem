# -*- coding: utf-8 -*-

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