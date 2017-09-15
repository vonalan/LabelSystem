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

class VideoLabel(object):
    def __init__(self, videoDir, imageDir, labelName, outputDir):
        self.videoDir = videoDir
        self.imageDir = imageDir
        self.frame = None

        self.rectFlag = 0
        # 0，自由；1，画框，2，移动框, 3, 移动角

        self.rects = []
        self.curRect = []

        self.bufframe = None
        self.minBox = 10
        self.shape = None
        self.boxImg = None
        self.pixDict = {}

        self.thick = 10
        self.linethick = 3
        self.lineHighThick = 5
        self.chooseRect = -1
        self.chooseType = -1  # 0,选框， 1， 选角， 2，选边
        self.startPos = []
        self.chooseXY = [-1, -1]
        self.selectedX = -1
        self.selectedY = -1

        # label 相关
        self.labels = []
        self.labelRect = -1
        self.maxLabel = 0
        self.colorList = []
        self.labelHight = 20
        self.labelWidth = 0
        # self.extractFrames()
        self.parseLabel(labelName)
        self.rightClick = 0  # ???

        self.length = 30  # 按键F减可以往后跳30张图片
        self.mini_batch_size = self.length * 10  # 每次加载进内存的图片数量，应该是self.length的整数倍
        self.classes = {}  # 标注的类别，是数值变量，不同于self.labels，self.classes的键对应self.chooseRect，值对应类别名称
        self.font = cv2.FONT_HERSHEY_SIMPLEX  # 标注的字体样式
        self.fontsize = 1  # 标注字体大小
        self.key = deque(maxlen=1)  # 实现类别标注的相关设置（用于储存当前按键输入的类别）

        self.storename = []  # 用于按键A倒退、按键D前进的相关设置
        self.storerects = []  # Store rects of each frame
        self.storeclses = []

        self.prefix_template = r'./template_prefix.xml'  # 这两个是输出xml的相关设置
        self.object_template = r'./template_object.xml'

        self.inputDir = None
        self.outputDir = outputDir
        self.outputimages_dbg = os.path.join(outputDir, 'images_dbg/')  # 输出images，为了调试方便
        self.outputimages = os.path.join(outputDir, 'images/')  # 输出images
        self.outputxmls = os.path.join(outputDir, 'xmls/')  # 输出xmls

        self.log = outputDir + '/outputlog.txt'  # 输出日志
        self.record = None
        self.rectsCopy = None
        self.dr = False
        self.fc = False
        self.video = ''
        self.scale = 1.0
        self.gap = 25
        self.SC = 0 # skip control

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

    def writeRecord(self, strings):
        log = open(self.record, 'a')
        log.write(strings + '\n')
        log.close()

    # 输出日志文件
    def writeLog(self, strings):
        log = open(self.log, 'a')
        log.write(strings + '\n')
        log.close()

    def update_outputDir(self, video):
        self.outputimages_dbg = os.path.join(self.outputDir, video, 'imgs_dbg/')
        self.outputimages = os.path.join(self.outputDir, video, 'imgs/')
        self.outputxmls = os.path.join(self.outputDir, video, 'xmls/')
        self.log = os.path.join(self.outputDir, video, 'output.log')
        self.record = os.path.join(self.outputDir, video, 'record.log')

        if not os.path.exists(self.outputimages_dbg): os.makedirs(self.outputimages_dbg)
        if not os.path.exists(self.outputimages): os.makedirs(self.outputimages)
        if not os.path.exists(self.outputxmls): os.makedirs(self.outputxmls)

    # 画图片中的框，四个小角，类别，还有输出图片，xml，日志操作，所以不同于update_frame里面的相关部分
    def draw_static(self, name, frame, shape, key, rects):
        '''写入原图'''
        # cv2.imwrite(self.outputimages + name, frame)

        '''写入debug模式的图'''
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
            cv2.putText(frame, str(v-48), coord, self.font, self.fontsize, self.colorList[v - 48 - 1], 2, cv2.LINE_AA)
        cv2.imwrite(self.outputimages_dbg + name, frame)

        '''保存xml文件'''
        # classes = list(map(chr, self.classes.values()))
        classes = [str(v-48) for v in self.classes.values()]
        boxes = [[item[0][0], item[0][1], item[1][0], item[1][1]] for item in rects]
        try:
            self.writeXML(shape, classes, boxes, self.outputxmls + name[:-4] + '.xml')
        except IndexError:
            print('You forget label the category!')

        '''需要一次性将全部日志写入时，配合op_queue使用'''
        # self.writeLog(name + ' , ' + chr(key))
        # self.writeRecord(name)

    # 因为有后退、后退30张、前进操作，每次操作都要更新name,frame,bufframe,shape
    def update(self, idx):
        name = self.storename[idx]  # name从self.storename中取
        imgname = os.path.join(self.outputimages, name)
        self.frame = cv2.imread(imgname)
        self.bufframe = copy.deepcopy(self.frame)
        self.shape = self.frame.shape
        self.chooseRect = -1

        # ''''''
        # if len(self.storeclses) > 0:
        #     self.classes = copy.deepcopy(self.storeclses[idx])
        # ''''''

        return name, self.frame, self.bufframe, self.shape

    def update_storerects(self, rects0, rects1, numFrames):
        # self.storerects = self.storerects[:idx_f - self.length]
        self.storerects = []
        for i in range(numFrames):
            curRects = []
            for j in range(len(rects0)):
                rect0, rect1 = np.array(rects0[j]), np.array(rects1[j])
                rect_gap = (rect1 - rect0) / float(self.length)
                rect_arr = (rect0 + rect_gap * i).astype('int32')
                curRects.append([list(rect_arr[0]), list(rect_arr[1])])
            self.storerects.append(curRects)

    def flush_storerects_2(self, op_queue=None):
        '''
                将self.storerects从内存写如磁盘。
                op_queue: operation_queue，用于一次性将所有operation写入磁盘
                :return: None
                '''

        '''DBG'''
        op_queue = {'name': ['a', 'a', 'f', 'f']}
        key = op_queue['name'][0]
        '''DBG'''

        for idx in range(len(self.storerects)):
            rects = self.storerects[idx]
            name, self.frame, self.bufframe, self.shape = self.update(idx)
            self.draw_static(name, self.frame, self.shape, key, rects)

    def flush_storerects(self, op_queue=None):
        '''
        将self.storerects从内存写如磁盘。
        op_queue: operation_queue，用于一次性将所有operation写入磁盘
        :return: None
        '''

        '''DBG'''
        op_queue = {'name': ['a', 'a', 'f', 'f']}
        key = op_queue['name'][0]
        '''DBG'''

        for idx in range(len(self.storerects)):
            rects = self.storerects[idx]
            name, self.frame, self.bufframe, self.shape = self.update(idx)
            self.draw_static(name, self.frame, self.shape, key, rects)

    def parseLabel(self, inname):
        infile = open(inname)
        self.maxLabel = 0

        colorList = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 66),
                     (0, 122, 122), (122, 0, 122), (122, 122, 0),(255, 0, 0), (0, 0, 255), (255, 255, 66),
                     (0, 122, 122), (122, 0, 122), (122, 122, 0)]
        for i, line in enumerate(infile):
            line = line.replace('\n', '')
            lenlab = len(line)
            if lenlab > self.maxLabel:
                self.maxLabel = lenlab
            # random.seed(i)
            # self.colorList.append((random.randint(0,255), random.randint(0,255), random.randint(0,255)))
            if len(line) > 0:
                self.labels.append(line)  # string
                self.colorList.append(colorList[i])
        infile.close()
        self.labelWidth = self.maxLabel * 10

    def resizeFrame(self, frame):
        # 显示器参数
        # [高，宽]
        height = 1080 * 9 // 10
        width = 1920 * 9 // 10
        screen = [height, width]
        # screen = [270, 480]

        # [高，宽]
        scale = 1.0
        sizes = [frame.shape[0], frame.shape[1]]
        # dstsizes = sizes

        r0 = (sizes[0] - screen[0])/float(screen[0])
        r1 = (sizes[1] - screen[1])/float(screen[1])

        if r0 > 0 and r1 > 0:
            ratio = r1/r0
            if ratio > 1:
                scale *= (screen[1] / float(sizes[1]))
                sizes[1] = screen[1]
                sizes[0] = int(scale * sizes[0])
            else:
                scale *= (screen[0] / float(sizes[0]))
                sizes[0] = screen[0]
                sizes[1] = int(scale * sizes[1])
        elif r0 > 0 and r1 <=0:
            scale *= (screen[0] / float(sizes[0]))
            sizes[0] = screen[0]
            sizes[1] = int(scale * sizes[1])
        elif r0 <= 0 and r1 > 0:
            scale *= (screen[1] / float(sizes[1]))
            sizes[1] = screen[1]
            sizes[0] = int(scale * sizes[0])
        else:
            return frame

        # newsize
        # [宽，高]
        sizes = tuple(reversed(sizes))
        frame = cv2.resize(frame, sizes, interpolation=cv2.INTER_CUBIC)

        self.scale = scale
        return frame

    def extractFrames(self, factor=6):
        videoPath = os.path.join(self.videoDir, self.video)
        cap = cv2.VideoCapture(videoPath)

        # import random
        # random.seed(0)
        # idx, cnt, offset = 0, 0, random.randint(0, factor)
        idx, cnt, offset = 0, 0, 0

        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                if not (cnt + offset) % factor:
                    idx += 1
                    imgPath = os.path.join(self.outputimages, self.video + '_' + str(idx) + '.png')
                    frame = self.resizeFrame(frame)
                    # print frame.shape
                    cv2.imwrite(imgPath, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                cnt += 1
            else:
                break
        cap.release()

    def draw_circle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # print "asdfag"
            cv2.circle(self.frame, (x, y), 100, (255, 0, 0), -1)

    # 原self.boxImag的值就是-1，第一次更新后，使得self.boxImg中的[y1:y2,x1:x2]的部分为3，四个小角的部分为4

    def update_boxImg(self):
        self.boxImg = np.zeros((self.shape[0], self.shape[1])) - 1 # -1
        th = self.thick
        for i, pts in enumerate(self.rects):
            self.boxImg[pts[0][1]:pts[1][1], pts[0][0]:pts[1][0]] = i * 3
            # [y1:y2,x1:x2]
            points = [pts[0], pts[1], [pts[0][0], pts[1][1]], [pts[1][0], pts[0][1]]]
            # [[x1,y1],[x2,y2],[x1,y2],[x2,y1]] = [左上，右下、左下、右上]
            for p in points:
                self.boxImg[p[1] - th: p[1] + th, p[0] - th: p[0] + th] = i * 3 + 1

    def show_labels(self, x, y):
        for i, name in enumerate(self.labels):
            for j in range(self.labelHight):
                cv2.line(self.frame, tuple((x, y + j)), tuple((x + self.labelWidth, y + j)), self.colorList[i],
                         thickness=1)

    ###
    def update_frame(self, x=-1, y=-1):
        self.frame = copy.copy(self.bufframe)
        # print('rectFlat:%s, chooseRect:%s, rect_len:%s' % (self.rectFlag, self.chooseRect, len(self.rects)))
        if self.rectFlag == 1:
            cv2.rectangle(self.frame, tuple(self.curRect[0]), (x, y), (0, 255, 0), thickness=self.linethick)
        if self.chooseRect >= 0:
            # print('updata_frame:', self.chooseType)
            # print self.chooseRect
            points = self.rects[self.chooseRect]
            if self.chooseType == 0:
                # 移动框
                cv2.rectangle(self.frame, tuple(points[0]), tuple(points[1]), (255, 255, 255),
                              thickness=self.linethick + 2)
                if self.labelRect >= 0:
                    self.show_labels(x, y)
                    #                cv2.rectangle(self.frame, tuple(points[0]), tuple(points[1]), (0,255,0), thickness=self.linethick)
            elif self.chooseType == 1:
                # 移动角
                pt = [0, 0]
                '''selectedX, selectedY'''
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
                    pt = [points[self.selectedX][0], points[self.selectedY][1]]
                th = self.thick
                cv2.rectangle(self.frame, (pt[0] - th, pt[1] - th), (pt[0] + th, pt[1] + th), (255, 255, 255),
                              thickness=1)
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
                    cv2.rectangle(self.frame, (pt[0] - th, pt[1] - th), (pt[0] + th, pt[1] + th), (0, 255, 0),
                                  thickness=1)
        # 标注类别
        for k, v in self.classes.items():
            box = self.rects[k]
            coord = (int((box[0][0] + box[1][0]) / 2), box[1][1])
            cv2.putText(self.frame, str(v-48), coord, self.font, self.fontsize, self.colorList[v - 48 - 1], 2, cv2.LINE_AA)

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
            # cv2.imshow('fff', self.boxImg)

        if num >= 0:
            self.frame = copy.copy(self.bufframe)
            idx = int(num / 3)
            self.chooseRect = idx
            self.chooseType = num % 3

            # print len(self.rects), self.boxImg[y,x], self.chooseRect, self.chooseType


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
        deltaX = x - self.startPos[0]
        deltaY = y - self.startPos[1]
        self.startPos[0] = x
        self.startPos[1] = y
        points = self.rects[self.chooseRect] # points = [[xmin, ymin], [xmax, ymax]]
        #        print deltaX, deltaY
        points[self.selectedX][0] = x # reference
        points[self.selectedY][1] = y # reference

        # print self.rects[self.chooseRect], self.shape, (x,y)

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
                    if self.xc == 1:
                        self.curRect.append([x, y])
                        self.rectFlag = 1
                    '''if F is pressed'''

                elif self.rectFlag == 1:
                    # 画框
                    self.rectFlag = 0
                    if abs(x - self.curRect[0][0]) > self.minBox or abs(y - self.curRect[0][1]) > self.minBox:
                        self.rect_done(x, y)
                    else:
                        self.frame = copy.copy(self.bufframe)
                    self.curRect = []
                elif self.rectFlag == 2 or self.rectFlag == 3:
                    # 移动框，角结束
                    self.rectFlag = 0
                    self.selectedX = -1
                    self.selectedY = -1
                    # self.update_boxImg()

                    '''边框溢出问题'''
                    # min.x, max.x
                    if self.rects[self.chooseRect][0][0] > self.rects[self.chooseRect][1][0]:
                        self.rects[self.chooseRect][0][0], self.rects[self.chooseRect][1][0] = \
                            self.rects[self.chooseRect][1][0], self.rects[self.chooseRect][0][0]
                    # min.y, max.y
                    if self.rects[self.chooseRect][0][1] > self.rects[self.chooseRect][1][1]:
                        self.rects[self.chooseRect][0][1], self.rects[self.chooseRect][1][1] = \
                            self.rects[self.chooseRect][1][1], self.rects[self.chooseRect][0][1]

                    if self.rects[self.chooseRect][0][0] < 0:
                        # print "self.rects[self.chooseRect][0][0] < 0"
                        self.rects[self.chooseRect][0][0] = 0
                        self.rects[self.chooseRect][1][0] = self.gap
                    if self.rects[self.chooseRect][0][1] < 0:
                        # print "self.rects[self.chooseRect][0][0] < 0"
                        self.rects[self.chooseRect][0][1] = 0
                        self.rects[self.chooseRect][1][1] = self.gap

                    self.update_boxImg()

                    '''update the last rects of storerects'''
                    if self.storerects and self.dr == True:
                        self.storerects[-1] = copy.deepcopy(self.rects)
                        # print 'Moving... '
                        # print self.storerects[0],
                        # print '-->',
                        # print self.storerects[-1]
                        # print '...',
                        # print 'Done! '
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
                    deltaX = x - self.startPos[0]
                    deltaY = y - self.startPos[1]
                    self.startPos[0] = x
                    self.startPos[1] = y

                    '''边框溢出问题'''
                    # print self.rects[self.chooseRect], self.shape, (x,y)

                    # gap = 10
                    # min.x
                    if self.rects[self.chooseRect][0][0] < 0: # min.x
                        self.rects[self.chooseRect][0][0] = 0
                    elif self.rects[self.chooseRect][0][0] > self.shape[1] - self.gap:
                        self.rects[self.chooseRect][0][0] = self.shape[1] - self.gap
                    else:
                        self.rects[self.chooseRect][0][0] += deltaX

                    # min.y
                    if self.rects[self.chooseRect][0][1] < 0: # min.y
                        self.rects[self.chooseRect][0][1] = 0
                    elif self.rects[self.chooseRect][0][1] > self.shape[0] - self.gap:
                        self.rects[self.chooseRect][0][1] = self.shape[0] - self.gap
                    else:
                        self.rects[self.chooseRect][0][1] += deltaY

                    # max.x
                    if self.rects[self.chooseRect][1][0] < self.gap:  # max.x
                        self.rects[self.chooseRect][1][0] = self.gap
                    elif self.rects[self.chooseRect][1][0] > self.shape[1]:
                        self.rects[self.chooseRect][1][0] = self.shape[1]
                    else:
                        self.rects[self.chooseRect][1][0] += deltaX

                    # max.y
                    if self.rects[self.chooseRect][1][1] < self.gap:  # max.y
                        self.rects[self.chooseRect][1][1] = self.gap
                    elif self.rects[self.chooseRect][1][1] > self.shape[0]:
                        self.rects[self.chooseRect][1][1] = self.shape[0]
                    else:
                        self.rects[self.chooseRect][1][1] += deltaY
                    '''边框溢出问题'''

                    # self.rects[self.chooseRect][0][0] += deltaX
                    # self.rects[self.chooseRect][1][0] += deltaX
                    # self.rects[self.chooseRect][0][1] += deltaY
                    # self.rects[self.chooseRect][1][1] += deltaY
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
                    # elif event == cv2.EVENT_RBUTTONUP:
                    # key = cv2.waitKey()
                    # cv2.putText(self.frame, chr(key), self.rects[self.chooseRect][0], cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255), 3)

    def labelling(self):
        import time

        self.xc = 1

        existname = [xml[:-4] + '.png' for xml in os.listdir(self.outputxmls)]
        storename = os.listdir(self.outputimages)
        self.storename = [img for img in storename if img not in existname]
        self.storename = sorted(self.storename, key = lambda x : int((x.split('.')[1]).split('_')[1]))
        numFrames = len(self.storename)

        '''确保每一帧被检查过才能进行窗口移动'''
        # self.check = 0
        # self.storecheck = []

        idx_itv = [0,self.length]
        cur_idx = 0
        name, self.frame, self.bufframe, self.shape = self.update(cur_idx)

        # cv2.namedWindow('image', flags=cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('image', flags=cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("image", self.draw_rect)
        while True:
            cv2.imshow("image", self.frame)
            key = cv2.waitKey(20)

            if key == ord('x'):
                '''由于需要插值，故而F键锁定删除与插入，A键解除锁定'''
                if len(self.rects) and self.chooseRect >= 0:
                    if self.xc == 1:
                        # print(len(self.rects), len(self.classes))

                        bufferrects = copy.deepcopy(self.rects)
                        bufferclses = copy.deepcopy(self.classes)

                        if bufferclses.has_key(self.chooseRect):
                            del bufferclses[self.chooseRect]
                        self.classes = {}

                        del self.rects[self.chooseRect]
                        self.chooseRect = -1

                        count = 0
                        for i, rects in enumerate(bufferrects):
                            if bufferclses.has_key(i):
                                self.classes[count] = bufferclses[i]
                                count += 1

                        # print(len(self.rects), len(self.classes))
                        # print(self.classes)

                        if len(self.storerects) > 0:
                            self.storerects[cur_idx] = copy.deepcopy(self.rects)
                            self.storeclses[cur_idx] = copy.deepcopy(self.classes)
                        self.update_boxImg()
                        self.update_frame()

            # if key in list(map(ord, self.labels)) + list(map(ord, ['q','w','e'])):
            if key in [int(label)+48 for label in self.labels] + list(map(ord, ['q','w','e','r'])):
                if key == ord('q'): key = 58
                if key == ord('w'): key = 59
                if key == ord('e'): key = 60
                if key == ord('r'): key = 61

                if len(self.rects) and self.chooseRect >= 0:
                    self.key.append(key)
                    self.classes[self.chooseRect] = self.key[0]
                    self.key.pop()
                    self.update_boxImg()
                    self.update_frame()

                    '''bug bug bug'''
                    if len(self.storeclses) > 0:
                        self.storeclses[cur_idx] = copy.deepcopy(self.classes)
                    if len(self.storerects) > 0:
                        self.storerects[cur_idx] = copy.deepcopy(self.rects)
                    '''bug bug bug'''

            self.SC = 1 if len(self.rects) == len(self.classes) else 0

            if key in map(ord, ['g', 'a', 'd']):
                '''跳转帧之前，将当前帧写回缓冲区/磁盘'''
                if self.SC == 1: #
                    print('[G|A|D] -- Saving')
                    if len(self.storerects) and len(self.storeclses):
                        self.storerects[cur_idx] = copy.deepcopy(self.rects)
                        self.storeclses[cur_idx] = copy.deepcopy(self.classes)
                    self.draw_static(name, self.frame, self.shape, key, self.rects)

            # if key == 102:
            if key == ord('g'):
                # 'f', 调到下30帧
                if self.dr == False and self.SC == 1:
                    print("F")
                    if self.storerects:
                        numFrames -= len(self.storerects)
                        # self.flush_storerects_2()
                        self.storerects = []
                        self.storeclses = []
                        self.storename = self.storename[self.length:]

                    if numFrames == 0:
                        self.writeRecord(self.video)
                        break

                    self.dr = True
                    self.fc = True
                    self.xc = 0

                    '''flag = self.numFrames%self.length'''
                    # idx_itv = [idx + self.length for idx in idx_itv]  # index interval: [idx, idx + self.length)
                    if idx_itv[1] > numFrames: idx_itv[1] = numFrames
                    # print idx_itv

                    # print 'Skipping to next %d frame...'%(self.length),
                    for idx in range(idx_itv[0], idx_itv[1]):
                        '''跳转帧之前，将当前帧写回缓冲区'''
                        self.storerects.append(copy.deepcopy(self.rects))
                        self.storeclses.append(copy.deepcopy(self.classes))

                        '''从缓冲区获取帧'''
                        # self.rects = self.storerects[idx]
                        # self.classes = self.storeclses[idx]
                        # self.check = self.storecheck[idx]
                        name, self.frame, self.bufframe, self.shape = self.update(idx)
                        self.update_boxImg()
                        self.update_frame()

                        self.writeLog(str(name) + ' , ' + chr(key))
                        print('F -- idx: %s, idx_f: %s, op_name: %s' % (cur_idx, idx_itv[1], name))
                    # print 'Done! '
                    cur_idx = idx_itv[1] - 1

                    print len(self.storerects), numFrames

            if key == 100:
                # 'd', 进入下一张图片
                if self.fc == True and self.SC == 1:
                    cur_idx += 1
                    cur_idx = cur_idx if cur_idx < idx_itv[1] else idx_itv[1]-1
                    self.rects = self.storerects[cur_idx]
                    self.classes = self.storeclses[cur_idx]
                    name, self.frame, self.bufframe, self.shape = self.update(cur_idx)
                    self.update_boxImg()
                    self.update_frame()

                    print('D -- idx: %s, idx_f: %s, op_name: %s' % (cur_idx, idx_itv[1], name))
                    self.writeLog(str(name) + ' , ' + chr(key))

            if key == 97:
                # 'a', 返回上一张图片
                if self.fc == True and self.SC == 1:
                    print("A")
                    if self.dr == True:
                        # print 'Interpolating...',
                        self.update_storerects(self.storerects[idx_itv[0]], self.storerects[idx_itv[1] - 1], idx_itv[1])
                        # print 'Done! '

                        for idx in range(idx_itv[0], idx_itv[1]):
                            self.rects = self.storerects[idx]
                            self.classes = self.storeclses[idx]
                            name, self.frame, self.bufframe, self.shape = self.update(idx)
                            self.update_boxImg()
                            self.update_frame()

                            print('A -- idx: %s, idx_f: %s, op_name: %s' % (idx, idx_itv[1], name))
                            self.writeLog(str(name) + ' , ' + chr(key))
                        self.dr = False
                        self.xc = 1
                    else:
                        cur_idx -= 1
                        cur_idx = cur_idx if cur_idx >= idx_itv[0] else 0
                        self.rects = self.storerects[cur_idx]
                        self.classes = self.storeclses[cur_idx]
                        name, self.frame, self.bufframe, self.shape = self.update(cur_idx)
                        self.update_boxImg()
                        self.update_frame()

                        print('A -- idx: %s, idx_f: %s, op_name: %s' % (cur_idx, idx_itv[1], name))
                        self.writeLog(str(name) + ' , ' + chr(key))
        cv2.destroyAllWindows()
        time.sleep(1)

if __name__ == '__main__':
    # videoDir = r'F:\Users\Kingdom\Desktop\LabelSystem\VideoLabel-DF\videos' # 视频文件夹地址
    # imageDir = r'F:\Users\Kingdom\Desktop\LabelSystem\VideoLabel-DF\images' # 不用设置
    # outputDir = r'F:\Users\Kingdom\Desktop\LabelSystem\VideoLabel-DF\outputs' # images和xmls输出地址
    # labelName = r'.\labels.txt'

    # videoDir = r'D:\Users\Administrator\Desktop\HGR\hand_dataset\0907fuyangben\videos'  # 视频文件夹地址
    # imageDir = r'D:\Users\Administrator\Desktop\HGR\VideoLabel-DF\images'  # 不用设置
    # outputDir = r'D:\Users\Administrator\Desktop\HGR\hand_dataset\0907fuyangben\outputs'  # images和xmls输出地址
    # labelName = r'.\labels.txt'

    videoDir = r'../videos'  # 视频文件夹地址
    imageDir = r'../images'  # 不用设置
    outputDir = r'../outputs'  # images和xmls输出地址
    labelName = r'./labels.txt'

    '''settings'''
    sample_factor = 6  # 每6帧抽取一帧
    '''settings'''

    videoList = os.listdir(videoDir)
    for video in videoList:
        videoPath = os.path.join(videoDir, video)

        print videoPath
        vl = VideoLabel(videoDir, imageDir, labelName, outputDir)

        '''self.record 要完成一个视频才会保存下来，所以可以用来判断'''
        vl.update_outputDir(video)
        if (os.path.exists(vl.record)):
            print '\n'
            continue

        vl.video = video
        # vl.extractFrames(factor=sample_factor)

        vl.linethick = 1
        vl.lineHighThick = 3
        vl.length = 10  # 选择F键要跳转的帧数，debug使用

        vl.labelling()

        print '\n'

    '''
    1. 画框前定位，需要加横竖两条辅助线。
    4. 边框3像素改成1像素
    5. "q"退出
    6. "a"键和"f"键相互控制
    7. "f"误触

    5. 空白键切换激活的Rect
    6. 方向键移动边
    7. 工作日志记录
    8. 边框图片边缘溢出
    9. 无法及时捕获边框
    10. 可以自由地添加或者删除边框, 使用list()替代dict()
    11. 添加缩放工具
    12. 添加旋转工具
    13. 出现只有一帧时无法保存的尴尬情况
    '''