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


class labelVisual:
    def __init__(self, imgDir, xmlDir, prefix_template, object_template, logname):
        self.boxImg = None
        self.bufframe = None
        self.frame = None
        self.shape = None
        self.boxes = None
        self.selected = -1
        self.colors = [(0, 0, 0), (0, 255, 0), (255, 0, 0), (0, 0, 255),
                       (255, 255, 66), (0, 122, 122), (122, 0, 122), (122, 0, 122), (122, 0, 122), (122, 0, 122)]
        self.xmlParser = xmlParser()
        ptem = open(prefix_template)
        self.ptemLine = ptem.read()
        ptem.close()
        otem = open(object_template)
        self.otemLine = otem.read()
        otem.close()
        # self.visual(imgDir, xmlDir, logname)

    def updateFrame(self):
        if self.selected >= 0:
            box = self.boxes[self.selected]
            cv2.rectangle(self.frame, tuple(box[1:3]), tuple(box[3:]), (255, 255, 255), thickness=7)
        self.drawBoxes(self.boxes)

    def freeMove(self, x, y):
        if self.boxes is None:
            return
        self.frame = copy.copy(self.bufframe)
        idx = int(self.boxImg[y, x])
        if idx >= 0:
            #            print idx
            self.selected = idx
        else:
            self.selected = -1
        self.updateFrame()

    #            cv2.rectangle(self.frame, tuple(box[1:3]), tuple(box[3:]), self.colors[int(box[0])], thickness = 3)

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
            cv2.rectangle(self.frame, tuple(box[1:3]), tuple(box[3:]), self.colors[int(box[0])], thickness=3)
            for i in range(30):
                cv2.line(self.frame, (box[1], box[2] - i), (box[1] + 40, box[2] - i), self.colors[int(box[0])],
                         thickness=1)
            cv2.putText(self.frame, box[0], (box[1], box[2]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            #            print box[2],box[4],  box[1], box[3]
            self.boxImg[box[2]:box[4], box[1]:box[3]] = b

    def saveXml(self, xmlname):
        imgsize = [self.frame.shape[1], self.frame.shape[0]]
        org_object = copy.copy(self.otemLine)
        ptemLine = copy.copy(self.ptemLine)
        ptemLine = ptemLine.replace('$width$', str(imgsize[0]))
        ptemLine = ptemLine.replace('$height$', str(imgsize[1]))

        outfile = open(xmlname, 'w')
        outfile.write(ptemLine)
        for i, box in enumerate(self.boxes):
            otemLine = copy.copy(org_object)
            otemLine = otemLine.replace("$name$", box[0])
            otemLine = otemLine.replace('$xmin$', str(box[1]))
            otemLine = otemLine.replace('$xmax$', str(box[3]))
            otemLine = otemLine.replace('$ymin$', str(box[2]))
            otemLine = otemLine.replace('$ymax$', str(box[4]))
            outfile.write(otemLine)
        outfile.write('</annotation>')
        outfile.close()

    def visual(self, imgDir, xmlDir, logname):
        cv2.namedWindow('image')
        cv2.setMouseCallback("image", self.events)
        nameList = os.listdir(imgDir)
        nameIdx = 0
        if os.path.isfile(logname):
            logfile = open(logname)
            nameIdx = int(logfile.read())
            logfile.close()
        while True:
            if nameIdx >= len(nameList):  # nameIdx >= 0  and nameIdx <= len(nameList)
                nameIdx = len(nameList) - 1
            if nameIdx < 0:
                nameIdx = 0
            name = nameList[nameIdx]  # load the picture and corresponding xml file
            imgname = os.path.join(imgDir, name)
            xmlname = os.path.join(xmlDir, name[:-4] + '.xml')
            print xmlname
            self.frame = cv2.imread(imgname)
            shape = self.frame.shape
            self.shape = shape
            #            print shape
            self.boxImg = np.zeros((shape[0], shape[1])) - 1
            #            print self.boxImg.shape
            self.bufframe = copy.copy(self.frame)  # deep copy
            if os.path.isfile(xmlname):
                self.boxes = self.parseXml(xmlname)
                self.drawBoxes(self.boxes)
            while True:
                cv2.imshow("image", self.frame)
                key = cv2.waitKey(20)
                if self.selected >= 0 and key < 58 and key > 48:  # {48:'0', ..., 57:'9', 58:':'}, change the label of gesture in the picture
                    #                    print key
                    print str(key - 48)
                    self.boxes[self.selected][0] = str(key - 48)
                    self.updateFrame()
                    cv2.imshow("image", self.frame)
                    key = cv2.waitKey(20)  # important
                if key == 32 or key == 100:  # {30:'space', 100:'d'}, skip to next picture
                    nameIdx += 1
                    if os.path.isfile(xmlname):
                        self.saveXml(xmlname)
                    logfile = open(logname, 'w')
                    logfile.write(str(nameIdx - 1))
                    logfile.close()
                    break
                elif key == 97:  # {97:'a'}, skip to previous picture
                    nameIdx -= 1
                    if os.path.isfile(xmlname):
                        self.saveXml(xmlname)
                    logfile = open(logname, 'w')
                    logfile.write(str(nameIdx - 1))
                    logfile.close()
                    break
                elif key == 27:  # {27:'esc'}
                    if os.path.isfile(xmlname):
                        self.saveXml(xmlname)
                    break
            if cv2.waitKey(20) == 27:  # {27:'esc'}
                break
        cv2.destroyAllWindows()


class labelSelect(labelVisual):
    def __init__(self, imgDir, xmlDir, prefix_template, object_template, logname):
        labelVisual.__init__(self, imgDir, xmlDir, prefix_template, object_template, logname)
        # self.select()

    def select(self, srcimgdir, srcxmldir, logname, dstimgdir, dstxmldir, objectives=[]):
        import shutil

        imgCount = [0, 0, 0, [0,0]] # {'noxml', 'nolabel', 'onelabel', {'purelabel', 'mixlabel'}}
        lblCount = len(objectives) * [0] # ['1', '2', '3', '4']

        imgList = os.listdir(srcimgdir)
        for idx, _ in enumerate(imgList):
            name = imgList[idx]  # load the picture and corresponding xml file
            imgpath = os.path.join(srcimgdir, name)
            xmlpath = os.path.join(srcxmldir, name[:-4] + '.xml')
            flag = os.path.exists(imgpath) and os.path.exists(xmlpath)

            if flag:
                xdstimgdir = ""
                xdstxmldir = ""

                try: # 有标签
                    self.boxes = self.parseXml(xmlpath)  # boxes is list of list

                    label = self.boxes[0][0]
                    check = 1  # 检查标签是否一致
                    for box in self.boxes:
                        if box[0] != label:
                            check = 0
                            break

                    if check == 1: # one or two
                        if len(self.boxes) == 1:
                            imgCount[2] += 1
                        else:
                            imgCount[3][0] += 1
                    else: # two
                        imgCount[3][1] += 1

                    for box in self.boxes:
                        label = box[0]

                        if label not in objectives:
                            print label, xmlpath
                            xdstimgdir = dstimgdir
                            xdstxmldir = dstxmldir

                        for i, obj in enumerate(objectives):
                            if label == obj:
                                lblCount[i] += 1

                except: # 无标签
                    imgCount[1] += 1
                    # print xmlpath
                else:
                    pass
                if xdstimgdir and xdstxmldir:
                    # if not os.path.exists(xdstimgdir): os.makedirs(xdstimgdir)
                    # if not os.path.exists(xdstxmldir): os.makedirs(xdstxmldir)
                    # shutil.copy(imgpath, xdstimgdir)
                    # shutil.copy(xmlpath, xdstxmldir)
                    # shutil.move(imgname, xdstimgdir)
                    # shutil.move(xmlname, xdstxmldir)
                    pass
            else:
                imgCount[0] += 1
                # print "Invalid xmlname: %s"%xmlname
                # raise Exception("Error")
        print imgCount
        print lblCount


if __name__ == '__main__':
    srcimgdir = r'D:\Users\Administrator\Desktop\HGR\hand_dataset\3hand_bk_20170818\3hand_bk_20170818_labelled\pure\jiandaoshitoubu\valid\imgs'
    srcxmldir = r'D:\Users\Administrator\Desktop\HGR\hand_dataset\3hand_bk_20170818\3hand_bk_20170818_labelled\pure\jiandaoshitoubu\valid\xmls'
    dstimgdir = r'D:\Users\Administrator\Desktop\HGR\hand_dataset\3hand_bk_20170818\3hand_bk_20170818_labelled\dbgs\imgs'
    dstxmldir = r'D:\Users\Administrator\Desktop\HGR\hand_dataset\3hand_bk_20170818\3hand_bk_20170818_labelled\dbgs\xmls'
    prefix_template = 'template_prefix.xml'
    object_template = 'template_object.xml'
    logname = 'visual.log'

    objectives = [str(i+1) for i in range(20)]
    # objectives = ['1','2','3','4']
    # labelVisual(srcimgdir, srcxmldir, prefix_template, object_template, logname)
    ls = labelSelect(srcimgdir, srcxmldir, prefix_template, object_template, logname)
    ls.select(srcimgdir, srcxmldir, logname, dstimgdir, dstxmldir, objectives)