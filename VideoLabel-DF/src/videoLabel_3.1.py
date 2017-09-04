# -*- coding: utf-8 -*-
"""
Created on Wed Aug 02 22:21:40 2017

@author: dapengguai
"""


import cv2, os, copy
import pdb
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque


class videoLabel(object):
    def __init__(self, videoDir, imageDir, labelName):
        self.videoDir = videoDir
        self.imageDir = imageDir
        self.frame = None

        self.rectFlag = 0
        # 0，自由；1，画框，2，移动框, 3, 移动角


        self.curRect = []

        self.bufframe = None
        self.minBox = 10
        self.shape = None
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
        self.rightClick = 0    # ???

        self.length = 30    # 按键F减可以往后跳30张图片
        self.classes = {}    # 标注的类别，是数值变量，不同于self.labels，self.classes的键对应self.chooseRect，值对应类别名称
        self.font = cv2.FONT_HERSHEY_SIMPLEX    # 标注的字体样式
        self.fontsize = 1    # 标注字体大小
        self.key = deque(maxlen=1)  # 实现类别标注的相关设置（用于储存当前按键输入的类别）
        
        self.storename = []    # 用于按键A倒退、按键D前进的相关设置
        self.storerects = []    # Store rects of each frame
        self.prefix_template = r'./template_prefix.xml'    # 这两个是输出xml的相关设置
        self.object_template = r'./template_object.xml'
        self.outputimages = outputDir + 'images/'    # 输出images
        self.outputxmls = outputDir + 'xmls/'    # 输出xmls
        self.log = outputDir + 'outputlog.txt'    # 输出日志
        self.rectsCopy = None
        self.dr = None
        self.fc = False

    # 输出xml方法，
    def writeXML(self, imgsize, names, boxes, outname):
        ptem = open(self.prefix_template)
        ptemline = ptem.read()
        ptem.close()
        ptemline = ptemline.replace('$width$', str(imgsize[0]))
        ptemline = ptemline.replace('$height$', str(imgsize[1]))

        otem = open(self.object_template)
        otemline = otem.read()
        otem.close()
        org_object = copy.copy(otemline)

        outfile = open(outname, 'w')
        outfile.write(ptemline)
        for i, box in enumerate(boxes):
            otemline = copy.copy(org_object)
            otemline = otemline.replace('$name$', names[i])
            otemline = otemline.replace('$xmin$', str(box[0]))
            otemline = otemline.replace('$xmax$', str(box[2]))
            otemline = otemline.replace('$ymin$', str(box[1]))
            otemline = otemline.replace('$ymax$', str(box[3]))
            outfile.write(otemline)
        outfile.write('</annotation>')
        outfile.close()

    # 输出日志文件
    def writeLog(self, strings):
        log = open(self.log, 'a')
        log.write(strings + '\n')
        log.close()

    # 画图片中的框，四个小角，类别，还有输出图片，xml，日志操作，所以不同于update_frame里面的相关部分
    def draw_static(self, name, frame, shape, key, rects):
        for pts in rects:
            cv2.rectangle(frame, tuple(pts[0]), tuple(pts[1]), (0, 255, 0), thickness=self.linethick)
            points = [pts[0], pts[1], [pts[1][0], pts[0][1]], [pts[0][0], pts[1][1]]]
            for pt in points:
                for th in range(self.thick):
                    cv2.rectangle(frame, (pt[0] - th, pt[1] - th), (pt[0] + th, pt[1] + th), (0, 255, 0),
                                  thickness=1)

        for k, v in self.classes.items():
            box = rects[k]
            coord = (int((box[0][0] + box[1][0]) / 2), box[1][1])
            cv2.putText(frame, chr(v), coord, self.font, self.fontsize, self.colorList[v-48-1], 2, cv2.LINE_AA)

        cv2.imwrite(self.outputimages + name, frame)
        classes = list(map(chr, self.classes.values()))
        boxes = [[item[0][0], item[0][1], item[1][0], item[1][1]] for item in rects]
        try:
            self.writeXML(shape, classes, boxes, self.outputxmls + name[:-4] + '.xml')
        except IndexError:
            print('You forget label the category!')
        self.writeLog(name + ' , ' + chr(key))

    # 因为有后退、后退30张、前进操作，每次操作都要更新name,frame,bufframe,shape
    def update(self, idx):
        name = self.storename[idx]    # name从self.storename中取
        imgname = os.path.join(self.imageDir, name)
        self.frame = cv2.imread(imgname)
        self.bufframe = copy.copy(self.frame)
        self.shape = self.frame.shape
        return name, self.frame, self.bufframe, self.shape

    '''Interpolation'''
    def update_storerects(self, rects0, rects1, idx_f):
        '''
        Interpolation method
        :param rects0: rects of current frame,
        :param rects1: last rects of storerects,
        :param idx_f: last index of current index intervals,
        :return: None
        '''
        self.storerects = self.storerects[:idx_f - 30]
        for i in range(self.length):
            curRects = []
            for j in range(len(rects0)):
                # print('j:%s, len:%s' % (j, len(rects0)))
                rect0, rect1 = np.array(rects0[j]), np.array(rects1[j])
                # rect_gap = (rect1 - rect0) / self.length
                rect_gap = (rect1 - rect0) / float(self.length)
                rect_arr = (rect0 + rect_gap * i).astype('int32')
                curRects.append([list(rect_arr[0]), list(rect_arr[1])])
            self.storerects.append(curRects)
    '''Interpolation'''

    def parseLabel(self, inname):
        infile = open(inname)
        self.maxLabel = 0

        colorList = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 66),
                     (0, 122, 122), (122, 0, 122), (122,122,0)]
        for i, line in enumerate(infile):
            line = line.replace('\n', '')
            lenlab = len(line)
            if lenlab > self.maxLabel:
                self.maxLabel = lenlab
            # random.seed(i)
            # self.colorList.append((random.randint(0,255), random.randint(0,255), random.randint(0,255)))
            if len(line) > 0:
                self.labels.append(line) #string
                self.colorList.append(colorList[i])
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
    # 原self.boxImag的值就是-1，第一次更新后，使得self.boxImg中的[y1:y2,x1:x2]的部分为3，四个小角的部分为4

    def update_boxImg(self):
        self.boxImg = np.zeros((self.shape[0], self.shape[1])) - 1
        th = self.thick
        for i, pts in enumerate(self.rects):
            self.boxImg[pts[0][1]:pts[1][1], pts[0][0]:pts[1][0]] = i*3
            # [y1:y2,x1:x2]
            points = [pts[0], pts[1], [pts[0][0], pts[1][1]], [pts[1][0], pts[0][1]]]
            # [[x1,y1],[x2,y2],[x1,y2],[x2,y1]] = [左上，右下、左下、右上]
            for p in points:
                self.boxImg[p[1]-th: p[1]+th, p[0]-th: p[0]+th] = i*3+1
                
    def show_labels(self, x, y):
        for i, name in enumerate(self.labels):
            for j in range(self.labelHight):
                cv2.line(self.frame, tuple((x, y+j)), tuple((x+self.labelWidth, y+j)), self.colorList[i], thickness=1)

    ###
    def update_frame(self, x=-1, y=-1):
        self.frame = copy.copy(self.bufframe)
        # print('rectFlat:%s, chooseRect:%s, rect_len:%s' % (self.rectFlag, self.chooseRect, len(self.rects)))
        if self.rectFlag == 1:
            cv2.rectangle(self.frame, tuple(self.curRect[0]), (x, y), (0, 255, 0), thickness=self.linethick)
        if self.chooseRect >= 0:
            # print('updata_frame:', self.chooseType)
            points = self.rects[self.chooseRect]
            if self.chooseType == 0:
                # 移动框
                cv2.rectangle(self.frame, tuple(points[0]), tuple(points[1]), (255,255,255), thickness=self.linethick+2)
                if self.labelRect >= 0:
                    self.show_labels(x, y)
#                cv2.rectangle(self.frame, tuple(points[0]), tuple(points[1]), (0,255,0), thickness=self.linethick)
            elif self.chooseType == 1:
                # 移动角
                pt = [0, 0]
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

        # 画四角的小框以及大框
        for i, pts in enumerate(self.rects):
            cv2.rectangle(self.frame, tuple(pts[0]), tuple(pts[1]), (0, 255, 0), thickness=self.linethick)
            points = [pts[0], pts[1], [pts[1][0], pts[0][1]], [pts[0][0], pts[1][1]]]
            # point = [[x1,y1],[x2,y2],[x2,y1],[x1,y2]
            for pt in points:
                for th in range(self.thick):
                    cv2.rectangle(self.frame, (pt[0]-th, pt[1]-th), (pt[0]+th, pt[1]+th), (0, 255, 0), thickness=1)
        # 标注类别
        for k, v in self.classes.items():
            box = self.rects[k]
            coord = (int((box[0][0] + box[1][0]) / 2), box[1][1])
            cv2.putText(self.frame, chr(v), coord, self.font, self.fontsize, self.colorList[v-48-1], 2, cv2.LINE_AA)
        
    def rect_done(self, x, y):
        # 画框完毕
        self.curRect.append([x, y])
#        print self.curRect
        if self.curRect[0][0] > self.curRect[1][0]:
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

    ###
    def free_move(self, x, y):
        if self.boxImg is None:
            num = -1
        else:
            num = int(self.boxImg[y, x])
        if num >= 0:
            self.frame = copy.copy(self.bufframe)
            idx = int(num/3)
            self.chooseRect = idx
            self.chooseType = num % 3

            '''nothing to do'''
            if len(self.key) == 1:
                self.classes[self.chooseRect] = self.key[0]
                self.key.pop()
            '''nothing to do'''

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

    ####
    def draw_rect(self, event, x, y, flags, param):
        if self.rightClick == 0:
            # ???

            if event == cv2.EVENT_LBUTTONUP:
                # print('LBUTTONUP', 'Flag:', self.rectFlag, 'Rect:', self.chooseRect)
                # 画框的左上角时self.rectFla为0，画到右下角时self.rectFlag为1
                if self.rectFlag == 0:
                    # 0，自由；1，画框，2，移动框, 3, 移动边缘
                    # self.curRect.append([x, y])
                    # [x,y]为event触发时的坐标
                    # self.rectFlag = 1

                    '''if F is pressed'''
                    if self.fc == False:
                        self.curRect.append([x, y])
                        self.rectFlag = 1
                    '''if F is pressed'''

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
    #                self.update_boxImg()2

                    '''update the last rects of storerects'''
                    if self.storerects:
                        # self.storerects[-1] = self.rects
                        self.storerects[-1] = copy.deepcopy(self.rects)
                        # print 'len(elf.storects: ', len(self.storerects)
                        # print 'self.storects[0]: ', self.storerects[0]
                        # print 'self.storects[-1]:', self.storerects[-1]

                        print 'Moving... ',
                        # print self.storerects[0],
                        # print '-->',
                        # print self.storerects[-1]
                        # print '...',
                        print 'Done! '

                    '''update the last rects of storerects'''
                
            elif event == cv2.EVENT_MOUSEMOVE:
                # 首先触发该事件，若不进行其他操作:self.free_move -> self.update_frame(不会有其他任何操作)
                # print(self.rectFlag, self.chooseRect)
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
                # print('LBUTTONDOWN', self.rectFlag, self.chooseRect)
                if self.rectFlag == 0:
                    if self.chooseRect >= 0:
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

            # elif event == cv2.EVENT_RBUTTONUP:
                # key = cv2.waitKey()
                # cv2.putText(self.frame, chr(key), self.rects[self.chooseRect][0], cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255), 3)

    def labelling(self):
        '''
        A -- Skips to previous frame, or 
          -- Interpolates rects
        D -- Skips to next frame
        F -- Skips 30 frames

        1. 进行的所有操作能够实时显示；
        2. 进行的所有操作能够实时保存在self.storerects中，
           选择合适的时机写回磁盘。
        
        1. 误触鼠标左键启动画框，按esc取消； 

        :return: None
        '''

        cv2.namedWindow('image')
        cv2.setMouseCallback("image", self.draw_rect)

        idx = 0
        idx_f = 0
        
        self.storename = os.listdir(self.imageDir)

        name, self.frame, self.bufframe, self.shape = self.update(idx)

        # print('op_name,%s' % name)

        while True:
            cv2.imshow("image", self.frame)

            key = cv2.waitKey(20)

            if key in list(map(ord, self.labels)):
                '''if one Rect selected'''
                if self.chooseRect >= 0:
                    self.key.append(key)
                    self.classes[self.chooseRect] = self.key[0]
                    self.key.pop()
                    
                    # self.draw_static(name, self.frame, self.shape, key, self.rects)
                    self.update_frame()

            if key == 102:
                # 'f', 往后跳30张

                '''
                Once F pressed, the size of self.rects is fixed,
                no more rect!!!
                '''
                self.fc = True

                print 'Skipping to next 30st frame...',

                self.dr = 1
                idx = idx_f # front index
                for i in range(self.length):
                    
                    '''deep copy'''
                    rects = copy.deepcopy(self.rects)
                    # self.storerects.append(self.rects)
                    self.storerects.append(rects)
                    '''deep copy'''

                    self.draw_static(name, self.frame, self.shape, key, self.rects)
                    # self.update_boxImg()
                    # self.update_frame()
                    # print('Place wait! The current pic is %s' % i)
                    idx += 1
                    idx_f += 1
                    name, self.frame, self.bufframe, self.shape = self.update(idx)
                self.update_frame()

                '''debug'''
                # idx == idx_f
                # print 'len(elf.storects: ', len(self.storerects)
                # print 'self.storects[0]: ', self.storerects[0]
                # print 'self.storects[-1]:', self.storerects[-1]
                '''debug'''

                print 'Done! '

            '''bug'''
            # idx == idx_f - self.length
            '''bug'''

            if key == 100:
                # 'd', 进入下一张图片
                if self.fc == True and (idx+1) < idx_f:
                    idx += 1
                    name, self.frame, self.bufframe, self.shape = self.update(idx)
                    self.rects = self.storerects[idx]
                    print('D -- idx: %s, idx_f: %s, op_name: %s' % (idx, idx_f, name))

                    self.draw_static(name, self.frame, self.shape, key, self.rects)
                    # self.update_boxImg()
                    # self.update_frame()

                    '''debug'''
                    # print 'len(elf.storects): ', len(self.storerects)
                    # print 'self.storects[0]: ', self.storerects[0]
                    # print 'self.storects[-1]:', self.storerects[-1]
                    '''debug'''

            if key == 97:
                # 'a', 返回上一张图片

                '''after f'''
                '''bug'''
                # idx == idx_f
                '''bug'''
                if self.fc == True and idx > (idx_f-30):

                    '''Interpolation'''
                    if self.dr == 1:
                        print 'Interpolating...',
                        # self.update_storerects(self.rects, self.storerects[idx_f-1], idx_f)
                        self.update_storerects(self.storerects[idx_f-30], self.storerects[idx_f-1], idx_f)
                        print 'Done! '
                        
                        for i in range(self.length):
                            name, self.frame, self.bufframe, self.shape = self.update(idx_f-1-i)
                            # print(len(self.storerects), idx_f)
                            print('A -- idx: %s, idx_f: %s, op_name: %s' % (idx_f-1-i, idx_f, name))
                            rects = self.storerects[idx_f-1-i] 
                            self.draw_static(name, self.frame, self.shape, key, rects)
                            # self.update_frame()
                        
                        self.dr = 0
                    '''Interpolation'''

                    # import time 
                    # time.sleep(5)

                    idx -= 1
                    self.rects = self.storerects[idx]
                    name, self.frame, self.bufframe, self.shape = self.update(idx)
                    print('A -- idx: %s, idx_f: %s, op_name: %s' % (idx, idx_f, name))

                    self.draw_static(name, self.frame, self.shape, key, self.rects)
                    # self.update_frame()

                    self.writeLog(str(name) + ' , ' + chr(key))


                '''bug'''
                # idx = idx_f - self.length
                '''bug'''

            if key == 113:
                # 'q'，退出
                break
        cv2.destroyAllWindows()
        return


# vl = videoLabel(videoDir, imageDir, labelName)
# vl.labelling()


if __name__ == '__main__':
    videoDir = r'../videos'
    imageDir = r'../images'
    outputDir = r'../outputs/'
    labelName = r'./labels.txt'
    vl = videoLabel(videoDir, imageDir, labelName)
    vl.labelling()








