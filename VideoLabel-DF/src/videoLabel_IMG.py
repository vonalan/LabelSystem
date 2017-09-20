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
    
    def points(self): 
        pass 
    
    def check(self): 
        pass 
    
    def move(self): 
        pass 

class FrameInfo(object):
    def __init__(self):
        self.name = ''
        self.checked = 0
        self.boxes = []

class VideoLabel(object):
    def __init__(self, videoDir, imageDir, labelName, outputDir):
        self.imgDir = ''
        self.xmlDir = ''
        self.vidDir = ''

        self.rectFlag = 0
        # 0，自由；1，画框，2，移动框, 3, 移动角

        self.name = ''
        self.checked = 0
        self.boxes = []
        self.curRect = []

        self.frame = None 
        self.bufframe = None 
        self.boxImg = None 
        self.shape = None 


        

        self.minBox = 10
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

        self.rightClick = 0  # ???

        # label 相关
        self.labels = []
        self.colors = []

        self.labelRect = -1
        self.maxLabel = 0
        # self.colorList = []
        self.labelHight = 20
        self.labelWidth = 0
        # self.extractFrames()
        self.parseLabel(labelName)
        

        self.length = 30  # 按键F减可以往后跳30张图片
        self.font = cv2.FONT_HERSHEY_SIMPLEX  # 标注的字体样式
        self.fontsize = 1  # 标注字体大小
        self.key = deque(maxlen=1)  # 实现类别标注的相关设置（用于储存当前按键输入的类别）

        self.prefix_template = r'./template_prefix.xml'  # 这两个是输出xml的相关设置
        self.object_template = r'./template_object.xml'  # 

        self.inputDir = None
        self.outputDir = outputDir
        # self.outputimages_dbg = os.path.join(outputDir, 'images_dbg/')  # 输出images，为了调试方便
        # self.outputimages = os.path.join(outputDir, 'images/')  # 输出images
        # self.outputxmls = os.path.join(outputDir, 'xmls/')  # 输出xmls

        self.log = outputDir + '/outputlog.txt'  # 输出日志
        self.record = None
        self.rectsCopy = None
        self.dr = False
        self.fc = False
        self.video = ''
        self.scale = 1.0
        self.gap = 25
        self.SC = 0 # skip control
        self.checked = 0

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
        self.name = self.storerects[idx].name
        self.checked = self.storerects[idx].checked
        self.boxes = self.storerects[idx].boxes

        imgname = os.path.join(self.outputimages, self.name)
        self.frame = cv2.imread(imgname)
        self.bufframe = copy.deepcopy(self.frame)
        self.shape = self.frame.shape
        self.chooseRect = -1


    def update_storerects(self, sidx, eidx):
        sboxes = self.storerects[sidx].boxes
        eboxes = self.storerects[eidx].boxes

        sArray, eArray = [], []
        for sbox, ebox in zip(sboxes, eboxes):
            sArray.append(sbox.rect)
            sArray.append(ebox.rect)
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
        cArray = (eArray + dArray * (eidx-1)).astype('int32')
        for i, clist in enumerate(cArray):
            self.storerects[eidx-1].boxes[i].rect = clist


    def update_storerects_1(self, rects0, rects1, numFrames):
        # self.storerects = self.storerects[:idx_f - self.length]
        self.storerects = []
        for i in range(numFrames): # 最后一帧不用修改
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


    def rotate_dev(self, image=None, angle=90):
        height = image.shape[0]
        width = image.shape[1]

        if angle%180 == 0:
            scale = 1
        elif angle%90 == 0:
            scale = float(max(height, width))/min(height, width)
        else:
            import math
            scale = math.sqrt(pow(height,2)+pow(width,2))/min(height, width)

        rheight = height/2
        rwidth = width/2

        rotateMat = cv2.getRotationMatrix2D((width/2, height/2), angle, scale)
        rotateImg = cv2.warpAffine(image, rotateMat, (width, height))
        # cv2.imshow('rotate', rotateImg)
        # cv2.waitKey(0)

        return rotateImg


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
                if cnt == 0:
                    print u'按空格键调整方向，按ESC退出'
                    oriention = self.oriente(frame)
                    print u'方向--%d, 解压图片'%oriention

                if not (cnt + offset) % factor:
                    idx += 1
                    imgPath = os.path.join(self.outputimages, self.video + '_' + str(idx) + '.png')
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
        for i, pts in enumerate(self.boxes):
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
        if self.chooseRect >= 0:
            # print('updata_frame:', self.chooseType)
            # print self.chooseRect
            points = self.boxes[self.chooseRect]
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
        for i, pts in enumerate(self.boxes):
            cv2.rectangle(self.frame, tuple(pts[0]), tuple(pts[1]), (0, 255, 0), thickness=self.linethick)
            points = [pts[0], pts[1], [pts[1][0], pts[0][1]], [pts[0][0], pts[1][1]]]
            # point = [[x1,y1],[x2,y2],[x2,y1],[x1,y2]
            for pt in points:
                for th in range(self.thick):
                    cv2.rectangle(self.frame, (pt[0] - th, pt[1] - th), (pt[0] + th, pt[1] + th), (0, 255, 0),
                                  thickness=1)
        # 标注类别
        for k, v in self.classes.items():
            box = self.boxes[k]
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
        self.boxes.append(self.curRect)
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

            # print len(self.boxes), self.boxImg[y,x], self.chooseRect, self.chooseType


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
        points = self.boxes[self.chooseRect]
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
        points = self.boxes[self.chooseRect] # points = [[xmin, ymin], [xmax, ymax]]
        #        print deltaX, deltaY
        points[self.selectedX][0] = x # reference
        points[self.selectedY][1] = y # reference

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
                    if self.boxes[self.chooseRect][0][0] > self.boxes[self.chooseRect][1][0]:
                        self.boxes[self.chooseRect][0][0], self.boxes[self.chooseRect][1][0] = \
                            self.boxes[self.chooseRect][1][0], self.boxes[self.chooseRect][0][0]
                    # min.y, max.y
                    if self.boxes[self.chooseRect][0][1] > self.boxes[self.chooseRect][1][1]:
                        self.boxes[self.chooseRect][0][1], self.boxes[self.chooseRect][1][1] = \
                            self.boxes[self.chooseRect][1][1], self.boxes[self.chooseRect][0][1]

                    if self.boxes[self.chooseRect][0][0] < 0:
                        # print "self.boxes[self.chooseRect][0][0] < 0"
                        self.boxes[self.chooseRect][0][0] = 0
                        self.boxes[self.chooseRect][1][0] = self.gap
                    if self.boxes[self.chooseRect][0][1] < 0:
                        # print "self.boxes[self.chooseRect][0][0] < 0"
                        self.boxes[self.chooseRect][0][1] = 0
                        self.boxes[self.chooseRect][1][1] = self.gap

                    self.update_boxImg()

                    '''update the last rects of storerects'''
                    if self.storerects and self.dr == True:
                        self.storerects[-1] = copy.deepcopy(self.boxes)
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
                    # print self.boxes[self.chooseRect], self.shape, (x,y)

                    # gap = 10
                    # min.x
                    if self.boxes[self.chooseRect][0][0] < 0: # min.x
                        self.boxes[self.chooseRect][0][0] = 0
                    elif self.boxes[self.chooseRect][0][0] > self.shape[1] - self.gap:
                        self.boxes[self.chooseRect][0][0] = self.shape[1] - self.gap
                    else:
                        self.boxes[self.chooseRect][0][0] += deltaX

                    # min.y
                    if self.boxes[self.chooseRect][0][1] < 0: # min.y
                        self.boxes[self.chooseRect][0][1] = 0
                    elif self.boxes[self.chooseRect][0][1] > self.shape[0] - self.gap:
                        self.boxes[self.chooseRect][0][1] = self.shape[0] - self.gap
                    else:
                        self.boxes[self.chooseRect][0][1] += deltaY

                    # max.x
                    if self.boxes[self.chooseRect][1][0] < self.gap:  # max.x
                        self.boxes[self.chooseRect][1][0] = self.gap
                    elif self.boxes[self.chooseRect][1][0] > self.shape[1]:
                        self.boxes[self.chooseRect][1][0] = self.shape[1]
                    else:
                        self.boxes[self.chooseRect][1][0] += deltaX

                    # max.y
                    if self.boxes[self.chooseRect][1][1] < self.gap:  # max.y
                        self.boxes[self.chooseRect][1][1] = self.gap
                    elif self.boxes[self.chooseRect][1][1] > self.shape[0]:
                        self.boxes[self.chooseRect][1][1] = self.shape[0]
                    else:
                        self.boxes[self.chooseRect][1][1] += deltaY
                    '''边框溢出问题'''

                    # self.boxes[self.chooseRect][0][0] += deltaX
                    # self.boxes[self.chooseRect][1][0] += deltaX
                    # self.boxes[self.chooseRect][0][1] += deltaY
                    # self.boxes[self.chooseRect][1][1] += deltaY
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

            # elif event == cv2.EVENT_RBUTTONDOWN:
            #     # if self.chooseRect >= 0:
            #     #     self.labelRect = self.chooseRect
            #     #     self.rightClick = 1
            #     pass
            # elif event == cv2.EVENT_RBUTTONUP:
            #     # key = cv2.waitKey()
            #     # cv2.putText(self.frame, chr(key), self.boxes[self.chooseRect][0], cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255), 3)
            #     pass
        else: pass

    def labelling(self):
        import time

        self.xc = 1

        '''准备工作'''
        existname = [xml[:-4] + '.png' for xml in os.listdir(self.outputxmls)]
        storename = os.listdir(self.outputimages)
        self.storename = [img for img in storename if img not in existname]
        self.storename = sorted(self.storename, key = lambda x : int((x.split('.')[1]).split('_')[1]))
        numFrames = len(self.storename)
        for name in self.storename:
            info = FrameInfo()
            self.storerects.append(info)



        '''确保每一帧被检查过才能进行窗口移动'''
        # self.check = 0
        # self.storecheck = []

        sidx, eidx, cidx = 0, 0, 0

        idx_itv = [0,self.length]
        cur_idx = 0

        self.update(0)

        # cv2.namedWindow('image', flags=cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('image', flags=cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("image", self.draw_rect)
        while True:
            cv2.imshow("image", self.frame)
            key = cv2.waitKey(20)

            if key == ord('x'):
                '''由于需要插值，故而F键锁定删除与插入，A键解除锁定'''
                if len(self.boxes) and self.chooseRect >= 0:
                    if self.xc == 1:
                        del self.boxes[self.chooseRect]
                        self.update_boxImg()
                        self.update_frame()

            '''准备工作'''
            extraLabels = ['q','w','e','r','t','y','u','i','o','p']
            if key in [int(label)+48 for label in self.labels] + list(map(ord, extraLabels)):
                for i, label in enumerate(extraLabels):
                    if key == ord(label): key = 58 + i

                if len(self.boxes) and self.chooseRect >= 0:
                    self.boxes[self.chooseRect].label = key
                    self.update_boxImg()
                    self.update_frame()

            if key in map(ord, ['g', 'a', 'd']):
                if self.SC == 1: #
                    if self.fc == True: self.checked = 1
                    if len(self.storerects):
                        self.storerects[cur_idx].name = self.name
                        self.storerects[cur_idx].checked = self.checked
                        self.storerects[cur_idx].boxes = copy.deepcopy(self.boxes)
                    self.draw_static(self.name, self.frame, self.shape, key, self.boxes)

            if key == ord('G'):
                pass

            if key == ord('g'):
                if self.dr == False and self.SC == 1:
                    ''''''
                    sidx += self.length
                    eidx = min(eidx + self.length, len(self.storerects))
                    ''''''

                    if sidx > eidx:
                        self.writeRecord(self.video)
                        break

                    self.dr = True
                    self.fc = True
                    self.xc = 0


                    # for idx in range(sidx, eidx):
                    #     self.storerects[idx].name = self.name
                    #     self.storerects[idx].checked = self.checked
                    #     self.storerects[idx].boxes = copy.deepcopy(self.boxes)
                    #
                    #     print('F -- idx: %s, idx_f: %s, op_name: %s' % (cur_idx, idx_itv[1], self.name))
                    #     self.writeLog(str(self.name) + ' , ' + chr(key))

                    cidx = eidx - 1

            if key == ord('d'):
                if self.fc == True and self.SC == 1:
                    # if self.storerects[cidx+1].checked == 0:
                    #     self.update_storerects(eidx, cidx, -1)

                    cidx = min(cidx+1, sidx)
                    print('D -- idx: %s, idx_f: %s, op_name: %s' % (cidx, eidx, self.name))
                    self.writeLog(str(self.name) + ' , ' + chr(key))

            if key == ord('a'):
                if self.fc == True and self.SC == 1:
                    if self.storerects[cidx-1].checked == 0:
                        self.update_storerects(sidx, cidx)

                    cidx = max(cidx - 1, sidx)
                    print('D -- idx: %s, idx_f: %s, op_name: %s' % (cidx, eidx, self.name))
                    self.writeLog(str(self.name) + ' , ' + chr(key))

            self.update(cidx)
            self.update_boxImg()
            self.update_frame()

        cv2.destroyAllWindows()
        time.sleep(1)

if __name__ == '__main__':
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

        import shutil
        shutil.rmtree(vl.outputxmls)
        os.mkdir(vl.outputxmls)

        if (os.path.exists(vl.record)):
            print '\n'
            continue

        vl.video = video
        # vl.extractFrames(factor=sample_factor)

        vl.linethick = 1
        vl.lineHighThick = 3
        vl.length = 10  # 选择F键要跳转的帧数，debug使用

        prompt = u"""
标注图片
F--跳帧
A--插值|前一帧
D--后一帧
"""
        print prompt
        vl.labelling()

        print '\n'

    '''
    5. 空白键切换激活的Rect
    6. 方向键移动边
    7. 工作日志记录
    10. 可以自由地添加或者删除边框, 使用list()替代dict()
    11. 重写数据结构
    12. 与标签检查工具合并
    '''