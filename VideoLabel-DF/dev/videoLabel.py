# -*- coding: utf-8 -*-

import os 
import sys 

import cv2

import workFrame as WF

class VideoLabel(object): 
    def __init__(self, video): 
        '''global variables block'''
        self.labels = [str(i) for i in range(10)] 
        self.colors = [] 
        self.flowLength = 30

        '''path related block'''
        self.video = video
        self.outputImgDir = './'
        self.outputXMLDir = './'

        '''control block'''
        self.FC = 0 # control F
        self.AC = 1 # control A 
        self.DC = 1 # control D 
        self.UC = 1 # control update
        self.XC = 0 # control delete


    def _update_output_dirs_(self): 
        pass 
    
    def _extract_frames_(self): 
        pass 

    def draw_rect(self, event, x, y, flags, params):
        '''only one event in a momont'''
        if event == cv2.EVENT_MOUSEMOVE:
            print("cv2.EVENT_MOUSEMOVE", (x,y))
        elif event == cv2.EVENT_LBUTTONUP:
            print("cv2.EVENT_LBUTTONUP", (x,y))
        elif event == cv2.EVENT_LBUTTONDOWN:
            print("cv2.EVENT_LBUTTONDOWN", (x,y))
        elif event == cv2.EVENT_RBUTTONUP:
            print("cv2.EVENT_RBUTTONUP", (x,y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            print("cv2.EVENT_RBUTTONDOWN", (x,y))
        else:
            print("None", (x,y))
                
    def label(self): 
        self._extract_frames_()

        nameList = ['video_%s'%(str(i)) for i in range(100)]


        cv2.namedWindow('image')
        cv2.setMouseCallback("image", self.draw_rect)
        while (True): 
            key = cv2.waitKey(20)
            if key in map(ord, self.labels): 
                print(str(key-48)) 
            if key == ord('f'): 
                if self.AC == 1 and self.DC == 1:
                    self.FC = 1 
                    self.UC = 1 
                    self.AC = 0 
                    self.DC = 0 
                    print("F -- Duplication. ")
            if key == ord('a'): 
                if self.FC == 1:
                    if self.UC == 1: 
                        self.UC = 0 
                        print("A -- Interpolation. ")
                    else:
                        self.AC = 1
                        print('A -- Last. ')                       
            if key == ord('d'): 
                if self.FC == 1 and self.AC == 1: 
                    self.DC = 1 
                    print('D -- Next. ')
            if key == 27: 
                break 
        cv2.destroyAllWindows()

if __name__ == "__main__": 
    video = ''
    vl = VideoLabel(video)
    vl.label()