# -*- coding: utf-8 -*-

import os 
import sys 

import numpy as np 
import cv2 


class ImageReader(object):
    pass

class ImageWriter(object):
    pass

class VideoReader(object): 
    def __init__(self, sample_factor=6, mini_batch_size=20):
        self.sample_factor = sample_factor 
        self.mini_batch_size = mini_batch_size 
    
    def _read_frame_(self, videoPath=None): 
        cap = cv2.VideoCapture(videoPath)
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                yield frame 
                if cv2.waitKey(1) & 0xFF == ord('q'): 
                    break
            else: 
                break
        cap.release()
    
    def read(self, videoPath=None): 
        frameList = [] 
        
        factor = self.sample_factor 
        mbsize = self.mini_batch_size

        import random 
        random.seed(0)
        cnt, offset = 0, random.randint(0,factor)

        for idx, frame in enumerate(self._read_frame_(videoPath=videoPath)): 
            if not (offset+idx)%factor: 
                frameList.append(frame)
                cnt += 1
                if cnt > 0 and not cnt%mbsize: 
                    yield frameList
                    frameList = []
        if frameList: 
            yield frameList
            frameList = []

class VideoWriter(object):
    def __init__(self):
        pass

    def write(self, videoPath=None):
        pass
        # Define the codec and create VideoWriter object
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # out = cv2.VideoWriter('output.mp4', fourcc, 29.0, (1280, 720))
        #
        # cap = cv2.VideoCapture('VID_20170901_171941.mp4')
        # while (cap.isOpened()):
        #     ret, frame = cap.read()
        #     if ret == True:
        #         frame = cv2.flip(frame, 0)
        #
        #         # write the flipped frame
        #         out.write(frame)
        #
        #         cv2.imshow('frame', frame)
        #         if cv2.waitKey(1) & 0xFF == ord('q'):
        #             break
        #     else:
        #         break
        # cap.release()
        # out.release()
        # cv2.destroyAllWindows()



