# -*- coding: utf-8 -*-
"""
Created on Wed Aug 02 22:21:40 2017

@author: dapengguai
"""

import cv2, os, copy
import pdb
import numpy as np
import random


class videoLabel:
    def __init__(self, videoDir, imageDir, labelName):
        self.videoDir = videoDir
        self.imageDir = imageDir
        self.frame = None
        self.rectFlag = 0
        # 0，自由；1，画框，2，移动框, 3, 移动边缘
        self.curRect = []
        self.bufframe = None
        self.minBox = 10
        self.shape = None
        # box相关
        self.boxImg = None
        self.pixDict = {}
        self.rects = []
        self.thick = 10
        self.linethick = 3
        self.lineHighThick = 5
        self.chooseRect = -1
        self.chooseType = -1 # 0,选框， 1， 选角， 2，选边
        self.startPos = []
        self.chooseXY = [-1,-1]
        self.selectedX = -1
        self.selectedY = -1
        # label 相关
        self.labels = []
        self.labelRect = -1
        self.maxLabel = 0
        self.colorList = []
        self.labelHight = 20
        self.labelWidth = 0
        self.extractFrames()
        self.parseLabel(labelName)
        self.rightClick = 0
    
    def parseLabel(self, inname):
        infile = open(inname)
        self.maxLabel = 0
        for i, line in enumerate(infile):
            line = line.replace('\n', '')
            lenlab = len(line)
            if lenlab > self.maxLabel:
                self.maxLabel = lenlab
            random.seed(i)
            self.colorList.append((random.randint(0,255), random.randint(0,255), random.randint(0,255)))
            if len(line) > 0:
                self.labels.append(line)
        infile.close()
        self.labelWidth = self.maxLabel * 10
        # print self.colorList
            
    def extractFrames(self):
        imgNum = len(os.listdir(self.imageDir))
        if imgNum > 10:
            return
        for name in os.listdir(self.videoDir):
            videoName = os.path.join(videoDir, name)
            cmd = "ffmpeg -i " + videoName + " -q:v 2 -f image2 " + \
                    imageDir + '/' + name + "_%06d.png"
            # print cmd
            os.system(cmd)

    def draw_circle(self, event,x,y,flags,param):
        if event==cv2.EVENT_LBUTTONDOWN:
            # print "asdfag"
            cv2.circle(self.frame,(x,y),100,(255,0,0),-1)
            
    def update_boxImg(self):
        self.boxImg = np.zeros((self.shape[0], self.shape[1])) - 1
        th = self.thick
        for i, pts in enumerate(self.rects):
#            print i
#            print pts[0][0],pts[1][0], pts[0][1], pts[1][1]
            self.boxImg[ pts[0][1]:pts[1][1], pts[0][0]:pts[1][0] ] = i*3
            points = [pts[0], pts[1], [pts[0][0], pts[1][1]], [pts[1][0], pts[0][1]] ]
            for p in points:
                self.boxImg[ p[1]-th:p[1]+th, p[0]-th:p[0]+th ] = i*3+1
                
    def show_labels(self, x, y):
        for i, name in enumerate(self.labels):
            for j in range(self.labelHight):
                cv2.line(self.frame, tuple((x, y+j)), tuple((x+self.labelWidth, y+j)), self.colorList[i], thickness=1 )
        
    def update_frame(self, x=-1, y=-1):
        self.frame = copy.copy(self.bufframe)
        if self.rectFlag == 1:
            cv2.rectangle(self.frame, tuple(self.curRect[0]), (x,y), (0,255,0), thickness=self.linethick)
        if self.chooseRect >=0:
            points = self.rects[self.chooseRect]
            if self.chooseType == 0:
                # 移动框
                cv2.rectangle(self.frame, tuple(points[0]), tuple(points[1]), (255,255,255), thickness=self.linethick+2)
                if self.labelRect >=0:
                    self.show_labels(x, y)
#                cv2.rectangle(self.frame, tuple(points[0]), tuple(points[1]), (0,255,0), thickness=self.linethick)
            elif self.chooseType == 1:
                # 移动角
                pt = [0,0]
                if self.selectedX == -1:
                    if abs(points[0][0] - x) <= self.thick:
                        pt[0] = points[0][0]
                    else:
                        pt[0] = points[1][0]
                    if abs(points[0][1] - y) <= self.thick:
                        pt[1] = points[0][1]
                    else:
                        pt[1] = points[1][1]
                else:
                    pt = [points[self.selectedX][0], points[self.selectedY][1] ]
                th = self.thick
                cv2.rectangle(self.frame, (pt[0]-th, pt[1]-th), (pt[0]+th, pt[1]+th), (255,255,255),thickness = 1)
            elif self.chooseType == 2:
                # 移动边
                pass
            
        for i, pts in enumerate(self.rects):
            cv2.rectangle(self.frame, tuple(pts[0]), tuple(pts[1]), (0,255,0), thickness=self.linethick)
        # 画四角的小框
            points = [pts[0], pts[1], [pts[1][0], pts[0][1]], [pts[0][0], pts[1][1]] ]
#            print pts[0][0],pts[1][0], pts[0][1], pts[1][1]
            for pt in points:
                for th in range(self.thick):
                    cv2.rectangle(self.frame, (pt[0]-th, pt[1]-th), (pt[0]+th, pt[1]+th), (0,255,0), thickness = 1)
            
        
    def rect_done(self, x, y):
        # 画框完毕
        self.curRect.append([x, y])
#        print self.curRect
        if self.curRect[0][0]> self.curRect[1][0]:
            tmp = self.curRect[0][0]
            self.curRect[0][0] = self.curRect[1][0]
            self.curRect[1][0] = tmp
        if self.curRect[0][1] > self.curRect[1][1]:
            tmp = self.curRect[1][1]
            self.curRect[1][1] = self.curRect[0][1] 
            self.curRect[0][1] = tmp                        
        self.rects.append(self.curRect)
        self.update_boxImg()
        self.update_frame()
    
    def free_move(self, x, y):
        num = int(self.boxImg[y, x])
        if num >= 0:
            self.frame = copy.copy(self.bufframe)
            idx = int(num/3)
            self.chooseRect = idx
            self.chooseType = num%3
        else:
            self.chooseRect = -1
        self.update_frame(x, y)
        
    def select_conner(self, x, y):
        self.startPos = [x, y]
        points = self.rects[self.chooseRect]
        if abs(points[0][0] - x) <= self.thick:
            self.selectedX = 0
        else:
            self.selectedX = 1
        if abs(points[0][1] - y) <= self.thick:
            self.selectedY = 0
        else:
            self.selectedY = 1
        
        
    def move_conner(self, x, y):
        # 移动角
#        print x, y , self.startPos, self.selectedX, self.selectedY
        deltaX = x-self.startPos[0]
        deltaY = y-self.startPos[1]
        self.startPos[0] = x
        self.startPos[1] = y
        points = self.rects[self.chooseRect]
#        print deltaX, deltaY
        points[self.selectedX][0] = x
        points[self.selectedY][1] = y
#        points[0][self.selectedX] += deltaX
#        points[1][self.selectedY] += deltaY
        
    
    def draw_rect(self, event, x, y, flags, param):
        if self.rightClick == 0:
            if event == cv2.EVENT_LBUTTONUP:
                if self.rectFlag == 0:
                    # 开始画框
                    self.curRect.append([x, y])
                    self.rectFlag = 1
                elif self.rectFlag == 1:
                    # 画框
                    self.rectFlag = 0
                    if abs(x - self.curRect[0][0]) > self.minBox or abs(y-self.curRect[0][1]) > self.minBox:
                        self.rect_done(x, y)
                    else:
                        self.frame = copy.copy(self.bufframe)
                    self.curRect = []
                elif self.rectFlag == 2 or self.rectFlag == 3:
                    # 移动框，角结束
                    self.rectFlag = 0
                    self.selectedX = -1
                    self.selectedY = -1
    #                self.update_boxImg()
                
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.rectFlag == 1:
                    # 正在画框
                    self.frame = copy.copy(self.bufframe)
                    self.update_frame(x, y)
                elif self.rectFlag == 0:
                    # 鼠标自由移动
                    self.free_move(x, y)
                elif self.rectFlag == 2:
                    # 移动框
                    deltaX = x-self.startPos[0]
                    deltaY = y-self.startPos[1]
                    self.startPos[0] = x
                    self.startPos[1] = y
                    self.rects[self.chooseRect][0][0] += deltaX
                    self.rects[self.chooseRect][1][0] += deltaX
                    self.rects[self.chooseRect][0][1] += deltaY
                    self.rects[self.chooseRect][1][1] += deltaY
                    self.update_boxImg()
                    self.update_frame()
                elif self.rectFlag == 3:
                    # 移动角
                    self.move_conner(x, y)
                    self.update_boxImg()
                    self.update_frame(x, y)
                    
            elif event == cv2.EVENT_LBUTTONDOWN:
                if self.rectFlag == 0:
                    if self.chooseRect >=0:
    #                    print self.chooseType, x, y
                        if self.chooseType == 0:
                            # 移动框
                            self.rectFlag = 2
                            self.startPos = [x, y]
                        elif self.chooseType == 1:
                            # 移动角
                            self.rectFlag = 3
                            self.select_conner(x, y)
            elif event == cv2.EVENT_RBUTTONDOWN:
                if self.chooseRect >= 0:
                    self.labelRect = self.chooseRect
                    self.rightClick = 1
#        elif event == cv2.EVENT_RBUTTONUP:

    
    def labelling(self):
        cv2.namedWindow('image')        
        cv2.setMouseCallback("image", self.draw_rect)
        for name in os.listdir(self.imageDir):
            imgname = os.path.join(self.imageDir, name)
            self.frame = cv2.imread(imgname)
            shape = self.frame.shape
            self.shape = shape
#            print shape
            self.boxImg = np.zeros((shape[0], shape[1]))-1
#            print 'boximg', self.boxImg.shape
            self.bufframe = copy.copy(self.frame)
            while True:
                cv2.imshow("image", self.frame)
                key = cv2.waitKey(20)
                if key == 27:
                    break
            if cv2.waitKey(20) == 27:
                break
        cv2.destroyAllWindows() 

vl = videoLabel(videoDir, imageDir, labelName)
vl.labelling()



if __name__ == '__main__':
    videoDir = 'videos/'
    imageDir = 'images/'
    labelName = 'labels.txt'
