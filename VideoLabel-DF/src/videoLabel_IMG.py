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

class BBox(object):
    def __init__(self): 
        self.label = ''
        self.rect = [[-1,-1], [-1,-1]]

    def move(self, deltaX, deltaY):
        self.rect[0][0] += deltaX
        self.rect[0][1] += deltaY
        self.rect[1][0] += deltaX
        self.rect[1][1] += deltaY

    def check(self, minBox, shape):
        '''
        rect[0][0] >= 0 && rect[0][0] <= rect[1][0]
        rect[0][1] >= 0 && rect[0][1] <= rect[1][1]
        '''
        if self.rect[0][0] > self.rect[1][0]:
            self.rect[0][0], self.rect[1][0] = self.rect[1][0], self.rect[0][0]
        if self.rect[0][1] > self.rect[1][1]:
            self.rect[0][1], self.rect[1][1] = self.rect[1][1], self.rect[0][1]

        def chk(pos, t, l):
            if pos < t: pos = t
            if pos > l: pos = l
            return pos
        self.rect[0][0] = chk(self.rect[0][0], 0, shape[1]-minBox)
        self.rect[0][1] = chk(self.rect[0][1], 0, shape[0]-minBox)
        self.rect[1][0] = chk(self.rect[1][0], 0+minBox, shape[1])
        self.rect[1][1] = chk(self.rect[1][1], 0+minBox, shape[0])

    def points(self):
        pass

class FrameInfo(object):
    def __init__(self, name):
        self.name = name
        self.checked = 0
        self.boxes = []

class VideoLabel(object):
    def __init__(self, video, inputDir, outputDir):
        self.prefix_template = r'./template_prefix.xml'
        self.object_template = r'./template_object.xml'
        self.log = ''

        self.minBox = 10
        self.thick = 10
        self.linethick = 1
        self.lineHighThick = 3
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontsize = 1

        self.length = 10  # 按键F减可以往后跳10张图片

        self.inputDir = inputDir
        self.outputDir = outputDir

        self.video = video
        self.imgDir = ''
        self.xmlDir = ''

        self.labels = []
        self.colors = []

        self.name = ''
        self.boxes = []
        self.curRect = []
        self.frame = None 
        self.bufframe = None 
        self.boxImg = None 
        self.shape = None

        self.nameList = []
        self.storerects = []

        self.rectFlag = 0 # 0，自由；1，画框，2，移动框, 3, 移动角
        self.chooseRect = -1
        self.chooseType = -1 # 0,选框， 1， 选角， 2，选边
        self.startPos = []
        self.chooseXY = [-1, -1]
        self.selectedX = -1
        self.selectedY = -1
        self.rightClick = 0  # ???

        self.labelRect = -1
        self.labelHight = 20
        self.labelWidth = 0

        self._get_labels_and_colors_()
        self._update_dirs_()
        self._get_name_list_()

    def _get_labels_and_colors_(self):
        numbers = [str(i) for i in range(1,10)] # [1-9]
        letters = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o'] # [10-18]

        labels = numbers + letters
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255),
                  (255, 255, 66),(0, 122, 122), (122, 0, 122),
                  (122, 122, 0),(255, 0, 0), (0, 0, 255),
                  (255, 255, 66), (0, 122, 122), (122, 0, 122),
                  (122, 122, 0),(255, 0, 0), (0, 0, 255),
                  (255, 255, 66),(0, 122, 122), (122, 0, 122)]
        assert len(labels) == len(colors)

        # label = str(labels.index(key))
        # color = colors[labels.index(key)]
        self.labels = labels
        self.colors = colors

    def _get_name_list_(self):
        total = os.listdir(self.imgDir)
        exist = [xml[:-4] + '.png' for xml in os.listdir(self.xmlDir)]
        left = [img for img in total if img not in exist]
        self.nameList = sorted(left, key=lambda x: int((x.split('.')[1]).split('_')[1]))
        self.storerects = [FrameInfo(name) for name in self.nameList]
        self.length = min(self.length, len(self.nameList))

    def _update_dirs_(self):
        self.dbgDir = os.path.join(self.outputDir, self.video, 'dbgs/')
        self.imgDir = os.path.join(self.outputDir, self.video, 'imgs/')
        self.xmlDir = os.path.join(self.outputDir, self.video, 'xmls/')

        ''''''
        import shutil
        if os.path.exists(self.dbgDir): shutil.rmtree(self.dbgDir)
        if os.path.exists(self.xmlDir): shutil.rmtree(self.xmlDir)
        ''''''

        if not os.path.exists(self.dbgDir): os.makedirs(self.dbgDir)
        if not os.path.exists(self.imgDir): os.makedirs(self.imgDir)
        if not os.path.exists(self.xmlDir): os.makedirs(self.xmlDir)

        self.log = os.path.join(self.outputDir, video, 'output.log')

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
    def draw_static(self, name, frame, shape, key, boxes):
        '''写入原图'''
        # cv2.imwrite(self.outputimages + name, frame)

        '''写入debug模式的图'''
        for box in boxes:
            pts = box.rect
            cv2.rectangle(frame, tuple(pts[0]), tuple(pts[1]), (0, 255, 0), thickness=self.linethick)
            points = [pts[0], pts[1], [pts[1][0], pts[0][1]], [pts[0][0], pts[1][1]]]
            for pt in points:
                for th in range(self.thick):
                    cv2.rectangle(frame, (pt[0] - th, pt[1] - th), (pt[0] + th, pt[1] + th), (0, 255, 0),
                                  thickness=1)

        for box in boxes:
            if box.label:
                label = box.label
                rect = box.rect
                coord = (int((rect[0][0] + rect[1][0]) / 2), rect[1][1])
                cv2.putText(self.frame, label, coord, self.font, self.fontsize, self.colors[int(label)-1], 2,
                            cv2.LINE_AA)
        cv2.imwrite(self.dbgDir + name, frame)

        '''保存xml文件'''
        labels = [box.label for box in boxes]
        # rects = [[box.rect[0][0], box.rect[0][1], box.rect[1][0], box.rect[1][1]] for box in boxes]
        rects = [box.rect[0] + box.rect[1] for box in boxes]
        try:
            self.writeXML(shape, labels, rects, self.xmlDir + name[:-4] + '.xml')
        except IndexError:
            print('You forget label the category!')

        '''需要一次性将全部日志写入时，配合op_queue使用'''
        # self.writeLog(name + ' , ' + chr(key))
        # self.writeRecord(name)

    # 因为有后退、后退30张、前进操作，每次操作都要更新name,frame,bufframe,shape
    def update(self, idx):
        self.name = self.storerects[idx].name
        # self.checked = self.storerects[idx].checked
        self.boxes = self.storerects[idx].boxes

        imgname = os.path.join(self.imgDir, self.name)
        self.frame = cv2.imread(imgname)
        self.bufframe = copy.deepcopy(self.frame)
        self.shape = self.frame.shape
        self.chooseRect = -1


    def update_storerects(self, sidx, eidx, step):
        sboxes = self.storerects[sidx].boxes
        eboxes = self.storerects[eidx].boxes

        sArray, eArray = [], []
        for sbox, ebox in zip(sboxes, eboxes):
            sArray.append(sbox.rect)
            eArray.append(ebox.rect)
        sArray = np.array(sArray)
        eArray = np.array(eArray)
        dArray = (eArray - sArray)/float(eidx-sidx)

        # update all
        # for cidx in range(sidx, eidx):
        #     cArray = (eArray + dArray * cidx).astype('int32')
        #     cArray = cArray.tolist()
        #     for i, clist in enumerate(cArray):
        #         self.storerects[cidx].boxes[i].rect = clist

        # update one
        cArray = (sArray + dArray * (eidx-sidx-1)).astype('int32')
        cArray = cArray.tolist()
        for i, clist in enumerate(cArray):
            self.storerects[eidx-1].boxes[i].rect = clist


    def oriente(self, frame=None):
        rotate = frame

        count = 0
        while True:
            cv2.imshow('rotate', rotate)

            key = cv2.waitKey(20)
            if key == 32: # 空格
                rotate = self.rotate(rotate, oriention=1)
                count += 1
                print count%4
                continue
            elif key == 27: break
            else: pass
        cv2.destroyAllWindows()
        return count%4


    def rotate(self, image=None, oriention=0):
        assert len(image.shape) == 3
        for i in range(oriention):
            image = np.transpose(image, (1,0,2))
            image = cv2.flip(image,1)
        return image


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

    def extract_frame(self, factor=6):
        videoPath = os.path.join(self.vidDir, self.video)
        cap = cv2.VideoCapture(videoPath)

        # import random
        # random.seed(0)
        # idx, cnt, offset = 0, 0, random.randint(0, factor)
        idx, cnt, offset = 0, 0, 0

        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                if cnt == 0:
                    print u'按空格键调整方向，按ESC退出'
                    oriention = self.oriente(frame)
                    print u'方向--%d, 解压图片'%oriention

                if not (cnt + offset) % factor:
                    idx += 1
                    imgPath = os.path.join(self.imgDir, self.video + '_' + str(idx) + '.png')
                    frame = self.resizeFrame(frame)
                    # print frame.shape
                    frame = self.rotate(frame, oriention)
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
        rects = [box.rect for box in self.boxes]
        for i, pts in enumerate(rects):
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
        # print('rectFlat:%s, chooseRect:%s, rect_len:%s' % (self.rectFlag, self.chooseRect, len(self.boxes)))
        if self.rectFlag == 1:
            cv2.rectangle(self.frame, tuple(self.curRect[0]), (x, y), (0, 255, 0), thickness=self.linethick)
            print tuple(self.curRect[0]), (x, y)
        if self.chooseRect >= 0:
            # print('updata_frame:', self.chooseType)
            # print self.chooseRect
            points = self.boxes[self.chooseRect].rect
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
        rects = [box.rect for box in self.boxes]
        for i, pts in enumerate(rects):
            cv2.rectangle(self.frame, tuple(pts[0]), tuple(pts[1]), (0, 255, 0), thickness=self.linethick)
            points = [pts[0], pts[1], [pts[1][0], pts[0][1]], [pts[0][0], pts[1][1]]]
            # point = [[x1,y1],[x2,y2],[x2,y1],[x1,y2]
            for pt in points:
                for th in range(self.thick):
                    cv2.rectangle(self.frame, (pt[0] - th, pt[1] - th), (pt[0] + th, pt[1] + th), (0, 255, 0),
                                  thickness=1)
        # 标注类别
        for box in self.boxes:
            if box.label:
                label = box.label
                rect = box.rect
                coord = (int((rect[0][0] + rect[1][0]) / 2), rect[1][1])
                cv2.putText(self.frame, label, coord, self.font, self.fontsize, self.colors[int(label)-1], 2,
                            cv2.LINE_AA)

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

        curbox = BBox()
        curbox.rect = self.curRect
        self.boxes.append(curbox)
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
        else:
            self.chooseRect = -1

        self.update_frame(x, y)

    def select_conner(self, x, y):
        self.startPos = [x, y]
        points = self.boxes[self.chooseRect].rect
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
        self.boxes[self.chooseRect].rect[self.selectedX][0] = x
        self.boxes[self.chooseRect].rect[self.selectedY][1] = y


        # print self.boxes[self.chooseRect], self.shape, (x,y)

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
                    if self.XC == 1:
                        self.curRect.append([x, y])
                        self.rectFlag = 1
                    '''if F is pressed'''

                    # self.curRect.append([x, y])
                    # self.rectFlag = 1

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

                    '''边框溢出问题'''
                    # self.boxes[self.chooseRect].check()
                    # self.update_boxImg()
                    # self.update_frame()

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
                    self.boxes[self.chooseRect].move(deltaX, deltaY)
                    self.boxes[self.chooseRect].check(self.minBox, self.shape)
                    ''''''

                    self.update_boxImg()
                    self.update_frame()
                elif self.rectFlag == 3:
                    # 移动角
                    self.move_conner(x, y)
                    self.boxes[self.chooseRect].check(self.minBox, self.shape)
                    self.update_boxImg()
                    self.update_frame()

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

            # elif event == cv2.EVENT_RBUTTONDOWN:
            #     # if self.chooseRect >= 0:
            #     #     self.labelRect = self.chooseRect
            #     #     self.rightClick = 1
            #     pass
            # elif event == cv2.EVENT_RBUTTONUP:
            #     # key = cv2.waitKey()
            #     # cv2.putText(self.frame, chr(key), self.boxes[self.chooseRect][0], cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255), 3)
            #     pass

    def labelling(self):
        import time

        self.AC = 0  # A
        self.FC = 0  # F
        self.XC = 1  # insert | delete
        self.SC = 1  # skip control

        self.video = ''
        self.scale = 1.0

        # self.checked = 0

        sidx, eidx, cidx = -self.length, 0, 0

        self.update(0)
        self.update_boxImg()

        # cv2.namedWindow('image', flags=cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('image', flags=cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("image", self.draw_rect)
        while True:
            cv2.imshow("image", self.frame)
            # cv2.imshow("boxes", self.boxImg)
            key = cv2.waitKey(20)

            self.SC = 1
            for box in self.boxes:
                if not box.label:
                    self.SC = 0
                    break

            if key == ord('x'):
                '''由于需要插值，故而F键锁定删除与插入，A键解除锁定'''
                if len(self.boxes) and self.chooseRect >= 0:
                    if self.XC == 1:
                        del self.boxes[self.chooseRect]
                        self.chooseRect = -1
                        self.update_boxImg()
                        self.update_frame()

            if key in [ord(label) for label in self.labels]:
                print key

                if len(self.boxes) and self.chooseRect >= 0:
                    self.boxes[self.chooseRect].label = str(self.labels.index(chr(key)) + 1)
                    self.update_boxImg()
                    self.update_frame()

            if key in map(ord, ['g', 'a', 'd']):
                if self.SC == 1:  #
                    # if self.FC == 1: self.checked = 1
                    # self.checked = 1
                    self.storerects[cidx].name = self.name
                    self.storerects[cidx].checked = 1
                    self.storerects[cidx].boxes = copy.deepcopy(self.boxes)
                    self.draw_static(self.name, self.frame, self.shape, key, self.boxes)

            # if key == ord('G'):
            #     if self.AC == 0 and self.SC == 1:
            #         ''''''
            #         eidx = sidx
            #         sidx = max(sidx - self.length, 0)
            #         ''''''
            #
            #         if sidx >= eidx:
            #             # self.writeRecord(self.video)
            #             break

            if key == ord('g'):
                if self.AC == 0 and self.SC == 1:
                    ''''''
                    sidx = eidx
                    eidx = min(eidx + self.length, len(self.storerects))
                    ''''''

                    if sidx >= eidx:
                        # self.writeRecord(self.video)
                        break

                    self.AC = 1
                    self.FC = 1
                    self.XC = 0

                    # not necessary
                    for idx in range(sidx, eidx):
                        # self.storerects[idx].name = self.name
                        self.storerects[idx].checked = 0
                        self.storerects[idx].boxes = copy.deepcopy(self.boxes)

                        self.name = self.nameList[idx]
                        print('F --idx: %s, sidx: %s, eidx: %s, op_name: %s' % (idx, sidx, eidx, self.name))
                        # self.writeLog(str(self.name) + ' , ' + chr(key))

                    cidx = eidx - 1
                    self.name = self.nameList[cidx]
                    print('F --idx: %s, sidx: %s, eidx: %s, op_name: %s' % (cidx, sidx, eidx, self.name))

                    self.update(cidx)
                    self.update_boxImg()
                    self.update_frame()

            if key == ord('d'):
                if self.FC == 1 and self.SC == 1:
                    # # for symmetry
                    # if self.storerects[cidx+1].checked == 0:
                    #     self.update_storerects(eidx, cidx, -1)

                    cidx = min(cidx+1, eidx-1)
                    self.name = self.nameList[cidx]
                    print('D --idx: %s, sidx: %s, eidx: %s, op_name: %s' % (cidx, sidx, eidx, self.name))
                    # self.writeLog(str(self.name) + ' , ' + chr(key))

                    self.update(cidx)
                    self.update_boxImg()
                    self.update_frame()

            if key == ord('a'):
                if self.FC == 1 and self.SC == 1:
                    '''to update the previous frame'''
                    if cidx-1 > sidx and self.storerects[cidx-1].checked == 0:
                        self.update_storerects(sidx, cidx, 1)

                    cidx = max(cidx - 1, sidx)
                    self.name = self.nameList[cidx]
                    print('A --idx: %s, sidx: %s, eidx: %s, op_name: %s' % (cidx, sidx, eidx, self.name))
                    # self.writeLog(str(self.name) + ' , ' + chr(key))

                    self.update(cidx)
                    self.update_boxImg()
                    self.update_frame()

                    self.AC = 0
                    self.XC = 1
        cv2.destroyAllWindows()
        time.sleep(1)

if __name__ == '__main__':
    inputDir = r'F:\Users\kingdom\Documents\GIT\LabelSystem\VideoLabel-DF\videos'  # 视频文件夹地址
    outputDir = r'F:\Users\kingdom\Documents\GIT\LabelSystem\VideoLabel-DF\outputs'  # images和xmls输出地址

    videoList = os.listdir(inputDir)
    for video in videoList:
        videoPath = os.path.join(inputDir, video)

        print videoPath
        vl = VideoLabel(video, inputDir, outputDir)
        # vl.extract_frame()
        vl._update_dirs_()
        vl.length = 5
        vl.labelling()
    '''
    5. 空白键切换激活的Rect
    6. 方向键移动边
    7. 工作日志记录
    10. 可以自由地添加或者删除边框, 使用list()替代dict()
    11. 重写数据结构
    12. 与标签检查工具合并
    '''