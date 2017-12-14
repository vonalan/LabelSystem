# -*- coding: utf-8 -*-
"""
Created on Wed Aug 09 09:35:12 2017

@author: Administrator
"""

import cv2
import os
import numpy as np
import copy
import xml.etree.ElementTree as et
import pdb

class xmlParser:
    def __init__(self, labelMapName=''):
        self.labelNum = None
        self.labelNameDict = {}
        self.labelIdDict = {}
        self.labelMapName = labelMapName
        if labelMapName != '':
            self.loadLabelMap(labelMapName)

    def loadLabelMap(self, labelMapName):
        self.labelmap = caffe_pb2.LabelMap()
        labFile = open(self.labelMapName)
        text_format.Merge(str(labFile.read()), self.labelmap)
        labFile.close()
        self.labelNum = len(self.labelmap.item)
        for item in self.labelmap.item:
            labName = item.name
            labId = item.label
            self.labelNameDict[labName] = labId
            self.labelIdDict[labId] = labName

    def getLabelIdx(self, lab):
        if self.labelNameDict == {}:
            self.loadLabelMap(self.labelMapName)
        return self.labelNameDict[lab]

    def parsing(self, filename):
        tree = et.parse(filename)
        root = tree.getroot()
        rets = []
        for obj in root.iter('object'):
            labelname = obj.find('name').text
#            label = self.getLabelIdx(labelname)
            label = labelname
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            rets.append( [label, xmin, ymin, xmax, ymax])
        if root.find('object') is None:
            rets = [[0,0,0,0,0]]
        return rets

    def parsing_sizes_dev(self, filename):
        tree = et.parse(filename)
        root = tree.getroot()
        sizes = root.find('size')
        width = int(sizes.find('width').text)
        height = int(sizes.find('height').text)

        return [width, height]

class labelVisual:
    def __init__(self, imgDir, xmlDir, prefix_template, object_template, logname):
        self.boxImg = None
        self.bufframe = None
        self.frame = None
        self.shape = None
        self.boxes = None
        self.selected = -1

        self.labels = []
        self.colors = []
        self._get_labels_and_colors_()

        self.xmlParser = xmlParser()
        ptem = open(prefix_template)
        self.ptemLine = ptem.read()
        ptem.close()
        otem = open(object_template)
        self.otemLine = otem.read()
        otem.close()
        self.visual(imgDir, xmlDir, logname)

        ''''''
        self.buffboxes = None
        self.scale = 1.0
        ''''''

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

    def updateFrame(self):
        if self.selected >=0:
            box = self.boxes[self.selected]
            cv2.rectangle(self.frame, tuple(box[1:3]), tuple(box[3:]), (255,255,255), thickness = 7)
        self.drawBoxes(self.boxes)

    def freeMove(self, x, y):
        if self.boxes is None:
            return
        self.frame = copy.deepcopy(self.bufframe)

        ''''''
        self.resize(self.frame) # self.scale && self.frame
        shape = self.frame.shape
        self.shape = shape
        self.boxImg = np.zeros((shape[0], shape[1]))-1
        ''''''

        idx = int(self.boxImg[y, x])
        if idx >= 0:
#            print idx
            self.selected = idx
        else:
            self.selected = -1
        self.updateFrame()

#            cv2.rectangle(self.frame, tuple(box[1:3]), tuple(box[3:]), self.colors[int(box[0])], thickness = 3)

    '''绑定一个鼠标事件'''
    def events(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.freeMove(x, y)

    def parseXml(self, xmlname):
        self.boxes = self.xmlParser.parsing(xmlname)
        for box in self.boxes:
            if not box[0].isdigit():
                print "标注错误，不应该有非数字的名称", xmlname
        return self.boxes

    def drawBoxes(self, boxes):
        if boxes is None:
            return
        for b, box in enumerate(boxes):
            # print int(box[0]) - 1
            cv2.rectangle(self.frame, tuple(box[1:3]), tuple(box[3:]), self.colors[int(box[0])-1], thickness = 3)
            for i in range(30):
                cv2.line(self.frame, (box[1], box[2]-i), (box [1]+40, box[2]-i), self.colors[int(box[0])-1], thickness = 1)
            cv2.putText(self.frame,box[0],(box[1], box[2]), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),2)
#            print box[2],box[4],  box[1], box[3]
            self.boxImg[box[2]:box[4], box[1]:box[3]] = b

    '''最为核心的部分'''
    def saveXml(self, xmlname):
        imgsize = [self.bufframe.shape[1], self.bufframe.shape[0]]
        org_object = copy.copy(self.otemLine)
        ptemLine = copy.copy(self.ptemLine)
        ptemLine = ptemLine.replace('$width$', str(imgsize[0]))
        ptemLine = ptemLine.replace('$height$', str(imgsize[1]))

        outfile = open(xmlname, 'w')
        outfile.write(ptemLine)
        for i, box in enumerate(self.buffboxes):
            otemLine = copy.copy(org_object)
            otemLine = otemLine.replace("$name$", box[0])
            otemLine = otemLine.replace('$xmin$', str(box[1]))
            otemLine = otemLine.replace('$xmax$', str(box[3]))
            otemLine = otemLine.replace('$ymin$', str(box[2]))
            otemLine = otemLine.replace('$ymax$', str(box[4]))
            outfile.write(otemLine)
        outfile.write('</annotation>')
        outfile.close()

    # def resize_v1(self, frame=None, scale=None):
    #     '''
    #     resize v1
    #
    #     :param frame:
    #     :param scale:
    #     :return:
    #     '''
    #     # (1080, 1920) * 0.8 or
    #     # (1920, 1080) * 0.8
    #
    #     tmpshape = [1080, 1920]
    #     tmpshape1 = [int(i * 0.8) for i in tmpshape]
    #     tmpshape2 = [int(i * (1080/float(1920))) for i in reversed(tmpshape1)]
    #
    #     orishape = (frame.shape[0], frame.shape[1])
    #     objshape = (tuple(tmpshape1)) if orishape[0] < orishape[1] else (tuple(tmpshape2))
    #
    #     ''''''
    #     if scale is not None:
    #         objshape = map(lambda x: int(x * scale), orishape)
    #         resizeFrame = cv2.resize(frame, tuple(reversed(objshape)), interpolation=cv2.INTER_CUBIC)
    #         return scale, resizeFrame
    #     ''''''
    #
    #     r0 = objshape[0] / float(orishape[0])
    #     r1 = objshape[1] / float(orishape[1])
    #     scale = min(r0, r1) # !!!
    #
    #     resizeFrame = frame
    #     if scale < 1.0:
    #         objshape = map(lambda x: int(x * scale), orishape)
    #         resizeFrame = cv2.resize(frame, tuple(reversed(objshape)), interpolation=cv2.INTER_CUBIC)
    #
    #     # return scale, resizeFrame
    #     self.scale = scale
    #     self.frame = resizeFrame

    '''只用来显示图片，而不能用于xmls写入'''
    def resize(self, frame=None, scale=None):
        '''
        resize v2

        to substitute
        cv2.namedWindow('image', flags=cv2.WINDOW_NORMAL)
        for user experience
        '''
        # (544,960) or
        # (960,544)

        orishape = (frame.shape[0], frame.shape[1])
        objshape = (544, 960) if orishape[0] < orishape[1] else (960, 544)

        ''''''
        if scale is not None:
            objshape = map(lambda x: int(x * scale), orishape)
            resizeFrame = cv2.resize(frame, tuple(reversed(objshape)), interpolation=cv2.INTER_CUBIC)
            return scale, resizeFrame
        ''''''

        r0 = objshape[0] / float(orishape[0])
        r1 = objshape[1] / float(orishape[1])
        scale = min(r0, r1) # 无论放大还是缩小，都应该使用min()?!!!

        resizeFrame = frame
        if scale < 1.0:
            objshape = map(lambda x: int(x * scale), orishape)
            resizeFrame = cv2.resize(frame, tuple(reversed(objshape)), interpolation=cv2.INTER_CUBIC)

        # return scale, resizeFrame
        self.scale = scale
        self.frame = resizeFrame

    '''只用来显示图片，而不能用于xmls写入'''
    def resizeBoxes(self, boxes):
        # self.boxes == self.bufferboxes

        if self.boxes is None:
            return

        scale = self.scale
        for _, box in enumerate(boxes):  # box = [label, xmin, ymin, xmax, ymax]
            box[1] = int(box[1] * scale) # floor
            box[2] = int(box[2] * scale)
            box[3] = int(box[3] * scale)
            box[4] = int(box[4] * scale)

        '''谢天谢地啊，写入xmls的是self.bufferdboxes'''
        self.boxes = boxes

    def visual(self, imgDir, xmlDir, logname):
        '''有些图片只能显示一部分，需要设置自适应窗口'''
        cv2.namedWindow('image')
        # cv2.setMouseCallback("image", self.events)

        nameList = os.listdir(imgDir)
        # nameList = sorted(nameList, key=lambda x: int((x.split('.')[1]).split('_')[1]))
        nameIdx = 0

        '''log文件可以保存更多的信息'''
        if os.path.isfile(logname):
            logfile = open(logname)
            nameIdx = int(logfile.read())
            logfile.close()

        '''根据局部性原理，可以记录前面图片的标签应用于当前图片'''
        while True:
            if nameIdx >= len(nameList): # nameIdx >= 0  and nameIdx <= len(nameList)
                nameIdx = len(nameList)-1
            if nameIdx < 0:
                nameIdx =0

            name = nameList[nameIdx] # load the picture and corresponding xml file
            imgname = os.path.join(imgDir, name)
            xmlname = os.path.join(xmlDir, name[:-4]+'.xml')
            print xmlname

            '''有些图片只能显示一部分，需要设置自适应窗口'''
            '''对于有多个目标的图片，Bounding Boxes可以顺序自动显示，而不需要鼠标激活'''
            ###
            # frame = cv2.imread(imgname)
            # frame, scale = ImageAdaption(frame)
            # self.scale = scale


            frame = cv2.imread(imgname)
            self.frame = frame # self.frame用于画画
            self.bufframe = copy.deepcopy(self.frame)  # self.bufframe用于备份以及xml文件写入

            self.resize(self.frame) # self.scale && self.frame
            shape = self.frame.shape
            self.shape = shape
            self.boxImg = np.zeros((shape[0], shape[1]))-1

            if os.path.isfile(xmlname):

                try:
                    self.boxes = self.parseXml(xmlname)
                except:
                    self.boxes = []
                    print('No Label!!!')
                else:pass

                ''''''
                # self.boxes 用于画画
                self.buffboxes = copy.deepcopy(self.boxes) # self.buffboxes用于备份以及xmls写入
                self.resizeBoxes(self.boxes)
                self.drawBoxes(self.boxes)

                ''''''

            while True:
                cv2.imshow("image", self.frame)
                key = cv2.waitKey(20)
                cv2.setMouseCallback("image", self.events)

                if key in [ord(label) for label in self.labels]:
                    # print str(self.labels.index(chr(key)) + 1)
                    self.boxes[self.selected][0] = str(self.labels.index(chr(key)) + 1) # 更改当前激活box的label
                    ''''''
                    self.buffboxes[self.selected][0] = str(self.labels.index(chr(key)) + 1) # 更改当前激活box的label
                    ''''''
                    self.updateFrame()
                    cv2.imshow("image", self.frame)
                    key = cv2.waitKey(20) # important

                if key == 32 or key == 100: # {30:'space', 100:'d'}, skip to next picture
                    nameIdx += 1
                    if os.path.isfile(xmlname):
                        self.saveXml(xmlname)
                    logfile = open(logname, 'w')
                    logfile.write(str(nameIdx-1))
                    logfile.close()
                    break
                elif key == 97: # {97:'a'}, skip to previous picture
                    nameIdx -=1
                    if os.path.isfile(xmlname):
                        self.saveXml(xmlname)
                    logfile = open(logname, 'w')
                    logfile.write(str(nameIdx-1))
                    logfile.close()
                    break
                elif key == 27: # {27:'esc'}
                    if os.path.isfile(xmlname):
                        self.saveXml(xmlname)
                    break
            if cv2.waitKey(20) == 27: # {27:'esc'}
                break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    imgdir = r'../images/' # 图片文件夹地址
    xmldir = r'../xmls/' # xml文件夹地址
    prefix_template = 'template_prefix.xml'
    object_template = 'template_object.xml'
    logname = r'./visual.log' # ！！！当一个文件夹首次被标注时，记得设为-2
    # labelVisual(imgdir, xmldir, prefix_template, object_template, logname)
    raise Exception(r'这个版本不再维护，请使用功能更强大的LabelVisual-DG！！！')



