# -*- coding: utf-8 -*-

# Related
# https://github.com/suriyasingh/pose-annotation-tool

# Requirements
# python 2
# numpy
# opencv-python

from __future__ import division

import os
import copy
import math
import shutil
import functools

import numpy as np
import cv2

import xmlParser

NUM_JOINTS = 15
JOINT_TEMPLATE = [-1, -1, -1, -1]
# JOINTS_TEMPLATE = NUM_JOINTS * JOINT_TEMPLATE

class Joint(object):
    def __init__(self):
        self.label = -1
        self.x = -1
        self.y = -1
        self.status = 1

class Person(object):
    def __init__(self):
        self.name = -1
        self.status = -1 # not used
        self.joints = []



class PoseAnnotation(object):
    def __init__(self, inputsDir, outputDir, bankupDir):
        self.xmlParser = None
        self.frame = None
        self.bufferFrame = None
        self.matte = None

        # TODO: len(self.joints) == num_persons * num_joints * len(joint_template)
        self.numJointsPerPerson = 15
        self.joints = [] # for manual_anno
        self.bufferJoints = [] # for auto_anno

        self.curJointIdx = -1
        # self.curJoint = []

        # TODO: persons
        # self.persons = 0
        self.curPersonIdx = -1
        self.curJoints = []

        self.names = []
        self.labels = []
        self.colors = []

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontsize = 0.75

        self.inputsDir = inputsDir
        self.outputDir = outputDir
        self.backupDir = backupDir

        self.imageList = []

        # self._prepare_file_system()
        self._get_labels_and_colors()

    def prepare_file_system(self, inputsDir='../inputs/', outputsDir='../outputs/'):
        self.inputImageDir = os.path.join(self.inputsDir, 'images')
        self.inputAnnoDir = os.path.join(self.inputsDir, 'annotations')
        self.outputImageDir = os.path.join(self.outputDir, 'images')
        self.outputAnnoDir = os.path.join(self.outputDir, 'annotations')
        # self.backupImageDir = os.path.join(self.backupDir, 'images')
        self.backupAnnoDir = os.path.join(self.backupDir, 'annotations')

        if not os.path.exists(self.outputImageDir): os.makedirs(self.outputImageDir)
        if not os.path.exists(self.outputAnnoDir): os.makedirs(self.outputAnnoDir)
        if not os.path.exists(self.backupAnnoDir): os.makedirs(self.backupAnnoDir)

    def _get_labels_and_colors(self):
        numbers = [str(i % 10) for i in range(1, 11)] # [1-10]
        letters = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p']  # [11-20]
        labels = numbers + letters
        names = [
            "Right ankle",
            "Right knee",
            "Right hip",
            "Left hip",
            "Left knee",
            "Left ankle",
            "Right wrist",
            "Right elbow",
            "Right shoulder",
            "Left shoulder",
            "Left elbow",
            "Left wrist",
            "Neck",
            "Nose",
            "Head top"]
        colors = [[255, 0, 0],
                  [255, 85, 0],
                  [255, 170, 0],
                  [255, 255, 0],
                  [170, 255, 0],
                  [85, 255, 0],
                  [0, 255, 0],
                  [0, 255, 85],
                  [0, 255, 170],
                  [0, 255, 255],
                  [0, 170, 255],
                  [0, 85, 255],
                  [0, 0, 255],
                  [85, 0, 255],
                  [170, 0, 255],
                  [255, 0, 255],
                  [255, 0, 170],
                  [255, 0, 85]]
        links = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [2, 6],
            [6, 7],
            [7, 8],
            [2, 9],
            [9, 10],
            [10, 11],
            [2, 12],
            [12, 13],
            [13, 14]]
        self.colors = colors
        self.labels = labels
        self.links = [tuple(item) for item in links]
        self.names = names

    def update(self, index):
        imagePath = os.path.join(self.inputImageDir, self.imageList[index])

        self.frame = cv2.imread(imagePath)
        self.bufferFrame = copy.copy(self.frame)

        srcTxtPath = os.path.join(self.inputAnnoDir, self.imageList[index][:-4] + '_auto_anno.txt')
        srcXmlPath = os.path.join(self.inputAnnoDir, self.imageList[index][:-4] + '_manual_anno.xml')

        if os.path.exists(srcTxtPath):
            '''only one person object'''
            _, self.bufferJoints = self.xmlParser.parse_txt_v1(srcTxtPath)
            self.joints = [[item[0], item[1], []] for item in self.bufferJoints]
            dstTxtPath = os.path.join(self.backupAnnoDir, self.imageList[index][:-4] + '_auto_anno.txt')
            # # shutil.copy(srcTxtPath, dstTxtPath)
            shutil.move(srcTxtPath, dstTxtPath)
        elif os.path.exists(srcXmlPath):
            '''multiple person objects'''
            self.bufferJoints = []
            _, self.joints = self.xmlParser.parse_xml(srcXmlPath)
        else:
            '''invalid files'''
            self.bufferJoints = []
            self.joints = [[item[0], item[1], []] for item in self.bufferJoints]

        # TODO: bug bug bug
        if len(self.bufferJoints):
            self.curBufferJoints = self.bufferJoints[0][-1]
        else:
            self.curBufferJoints = []

        if len(self.joints):
            self.curPersonIdx = 0
            self.curJoints = self.joints[self.curPersonIdx][-1]
        else:
            self.curPersonIdx = -1
            self.curJoints = []

    def update_matte(self):
        self.matte = np.zeros(self.frame.shape[:2]) - 7
        for idx, joint in enumerate(self.curJoints):
            if joint[-1] >= 0:
                color = 7 * idx
                cv2.circle(self.matte, (joint[1], joint[2]), 20, color, -1)

    def draw_static(self, index):
        frame = copy.copy(self.bufferFrame)

        # TODO: ADD
        # draw image
        for person in self.joints:
            curJoints = person[-1]

            # draw links between points
            for link in self.links:
                if link[1] >= len(self.labels):
                    continue

                idx0, idx1 = -1, -1
                for idx, joint in enumerate(curJoints):
                    if joint[0] == link[0]: idx0 = idx
                    if joint[0] == link[1]: idx1 = idx

                if idx0 >= 0 and idx1 >= 0:
                    joint0, joint1 = curJoints[idx0], curJoints[idx1]
                    if joint0[-1] >= 0 and joint1[-1] >= 0:
                        # TODO:
                        color0 = (112, 112, 112) if joint0[-1] == 0 else self.colors[link[0]]
                        color1 = (112, 112, 112) if joint1[-1] == 0 else self.colors[link[1]]
                        color = [int(0.5 * (c1 + c2)) for c1, c2 in zip(color0, color1)]
                        cv2.line(frame, (joint0[1], joint0[2]), (joint1[1], joint1[2]), color, 2)

                        center = int((joint0[1] + joint1[1]) / 2.0), int((joint0[2] + joint1[2]) / 2.0)
                        mainAxis = int(((joint0[1] - joint1[1]) ** 2 + (joint0[2] - joint1[2]) ** 2) ** 0.5 * 0.5)
                        subAxis = 3
                        algle = int(math.degrees(math.atan2(joint0[2] - joint1[2], joint0[1] - joint1[1])))

                        polygon = cv2.ellipse2Poly(center, (mainAxis, subAxis), algle, 0, 360, 1)
                        cv2.fillConvexPoly(frame, polygon, color)

            # draw_joint_points and labels
            for idx, joint in enumerate(curJoints):
                if joint[-1] >= 0:
                    # TODO:
                    radius = 20 if idx == self.curJointIdx else 5
                    color = (112, 112, 112) if joint[-1] == 0 else self.colors[idx % len(self.colors)]
                    cv2.circle(frame, (joint[1], joint[2]), radius, color, -1)

                    label = str(joint[0] + 1)
                    bgColor = self.colors[idx % len(self.colors)]
                    fgColor = (0, 0, 0)
                    if joint[3] == 0:
                        bgColor = (112, 112, 112)
                        fgColor = (255, 255, 255)
                    font_size = cv2.getTextSize(label, self.font, self.fontsize, 2)  # ((w,h), b)

                    # TODO: more proper position of text
                    # coords1 = (int(joint[1] - radius - font_size[0][0]) - 2, int(joint[2] - font_size[0][1]/2.0) - 2)
                    # coords2 = (int(joint[1] - radius) + 2, int(joint[2] + font_size[0][1] / 2.0) + 2)
                    coords1, coords2 = self.get_text_coordinates(joint, radius, font_size, frame.shape, label)

                    cv2.rectangle(frame, (coords1[0], coords1[1] - 2), (coords2[0], coords2[1] + 2), bgColor, -1)
                    cv2.putText(frame, label, (coords1[0], coords2[1]), self.font, self.fontsize, fgColor, 2)

        # write image
        image = self.imageList[index]
        imagePath = os.path.join(self.outputImageDir, image)
        cv2.imwrite(imagePath, frame)

        # TODO: bugs bugs bugs
        # # write txts
        # annoPath = os.path.join(self.outputAnnoDir, image[:-4] + '_manual_anno.txt')
        # # annoPath = os.path.join(self.inputAnnoDir, image[:-4] + '_auto_anno.txt')
        # self.xmlParser.write_txt(frame.shape, self.names, self.joints, annoPath)

        # write xmls
        annoPath = os.path.join(self.outputAnnoDir, image[:-4] + '_manual_anno.xml')
        self.xmlParser.write_xml(frame.shape, self.names, self.joints, annoPath)
        annoPath = os.path.join(self.inputAnnoDir, image[:-4] + '_manual_anno.xml')
        self.xmlParser.write_xml(frame.shape, self.names, self.joints, annoPath)

    def get_text_coordinates(self, joint, radius, font_size, shape, label=''):
        # y
        rowPair_1 = (int(joint[2] - radius - font_size[0][1]), int(joint[2] - radius))
        rowPair_2 = (int(joint[2] - font_size[0][1] / 2.0), int(joint[2] + font_size[0][1] / 2.0))
        rowPair_3 = (int(joint[2] + radius), int(joint[2] + radius + font_size[0][1]))
        rowPairs = [rowPair_1, rowPair_2, rowPair_3]

        # x
        colPair_1 = (int(joint[1] - radius - font_size[0][0]), int(joint[1] - radius))
        colPair_2 = (int(joint[1] - font_size[0][0] / 2.0), int(joint[1] + font_size[0][0] / 2.0))
        colPair_3 = (int(joint[1] + radius), int(joint[1] + radius + font_size[0][0]))
        colPairs = [colPair_1, colPair_2, colPair_3]

        minCoords, maxCoords = (-1, -1), (-1, -1)
        tranverse_order = [(1,0), (0, 1), (1,2), (2, 1), (2,0), (0,0), (0,2), (2,2)]
        for j, i in tranverse_order:
            minCoords, maxCoords = zip(colPairs[i], rowPairs[j])
            flags = (minCoords[0] <= maxCoords[0] and minCoords[1] <= maxCoords[1]) and \
                    (minCoords[0] >= 0 and maxCoords[0] <= shape[1]) and \
                    (minCoords[1] >= 0 and maxCoords[1] <= shape[0])
            if flags: return minCoords, maxCoords
        return minCoords, maxCoords

    def update_frame(self, x=-1, y=-1):
        self.frame = copy.copy(self.bufferFrame)

        # step 1 auto_anno
        for idx, joint in enumerate(self.curBufferJoints):
            radius = 3
            color = self.colors[idx % len(self.colors)]
            cv2.circle(self.frame, (joint[1], joint[2]), radius, color, -1)

        # step 2 manual_anno
        # draw links between points
        for link in self.links:
            if link[1] >= len(self.labels):
                continue

            idx0, idx1 = -1, -1
            for idx, joint in enumerate(self.curJoints):
                if joint[0] == link[0]: idx0 = idx
                if joint[0] == link[1]: idx1 = idx

            if idx0 >= 0 and idx1 >= 0:
                joint0, joint1 = self.curJoints[idx0], self.curJoints[idx1]
                if joint0[-1] >=0 and joint1[-1] >= 0:
                    # TODO:
                    color0 = (112, 112, 112) if joint0[-1] == 0 else self.colors[link[0]]
                    color1 = (112, 112, 112) if joint1[-1] == 0 else self.colors[link[1]]
                    color = [int(0.5 * (c1 + c2)) for c1, c2 in zip(color0, color1)]
                    cv2.line(self.frame, (joint0[1], joint0[2]), (joint1[1], joint1[2]), color, 2)

                    center = int((joint0[1] + joint1[1]) / 2.0), int((joint0[2] + joint1[2]) / 2.0)
                    mainAxis = int(((joint0[1] - joint1[1]) ** 2 + (joint0[2] - joint1[2]) ** 2) ** 0.5 * 0.5)
                    subAxis = 3
                    algle = int(math.degrees(math.atan2(joint0[2] - joint1[2], joint0[1] - joint1[1])))

                    polygon = cv2.ellipse2Poly(center, (mainAxis, subAxis), algle, 0, 360, 1)
                    cv2.fillConvexPoly(self.frame, polygon, color)

        # draw_joint_points and labels
        for idx, joint in enumerate(self.curJoints):
            if joint[-1] >=0:
                # TODO:
                radius = 20 if idx == self.curJointIdx else 5
                color = (112, 112, 112) if joint[-1] == 0 else self.colors[idx % len(self.colors)]
                cv2.circle(self.frame, (joint[1], joint[2]), radius, color,-1)

                label = str(joint[0] + 1)
                bgColor = self.colors[idx % len(self.colors)]
                fgColor = (0, 0, 0)
                if joint[3] == 0:
                    bgColor = (112, 112, 112)
                    fgColor = (255, 255, 255)
                font_size = cv2.getTextSize(label, self.font, self.fontsize, 2)  # ((w,h), b)

                # TODO: more proper position of text
                # coords1 = (int(joint[1] - radius - font_size[0][0]) - 2, int(joint[2] - font_size[0][1]/2.0) - 2)
                # coords2 = (int(joint[1] - radius) + 2, int(joint[2] + font_size[0][1] / 2.0) + 2)
                coords1, coords2 = self.get_text_coordinates(joint, radius, font_size, self.frame.shape, label)

                cv2.rectangle(self.frame, (coords1[0], coords1[1] - 2), (coords2[0], coords2[1] + 2), bgColor, -1)
                cv2.putText(self.frame, label, (coords1[0], coords2[1]), self.font, self.fontsize, fgColor, 2)

    def call_back_func(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.curJointIdx = int(self.matte[y, x] / 7)
        if event == cv2.EVENT_LBUTTONUP:
            if len(self.joints):
                if len(self.curJoints) < self.numJointsPerPerson:
                    joint = [-1, x, y, 1] # visible
                    joint[0] = len(self.curJoints)
                    self.curJoints.append(joint)
            else:
                print 'Create a person object first! '
        if event == cv2.EVENT_RBUTTONUP:
            if len(self.joints):
                if len(self.curJoints) < self.numJointsPerPerson:
                    joint = [-1, x, y, 0]  # invisible
                    joint[0] = len(self.curJoints)
                    self.curJoints.append(joint)
            else:
                print 'Create a person object first! '
        self.update_matte()
        self.update_frame()

    def annotation(self):
        self.prepare_file_system()
        self.imageList = os.listdir(self.inputImageDir)
        cidx = 0
        self.update(cidx)
        self.update_matte()

        # cv2.namedWindow('image', flags=cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('image', flags=cv2.WINDOW_NORMAL)
        # cv2.setMouseCallback('image', self.call_back_func) #binding the call_back_func
        while True:
            cv2.setMouseCallback('image', self.call_back_func)  # rebinding the call_back_func

            cv2.imshow('image', self.frame)
            # cv2.imshow('matte', self.matte)

            key = cv2.waitKey(20)

            # joint-level
            # delete
            if self.curJointIdx >= 0 and key == ord('x'):
                label = self.curJoints[self.curJointIdx][0]
                joints = self.curJoints[:label]
                pname = self.curPersonIdx
                status = self.joints[self.curPersonIdx][1]
                self.joints[self.curPersonIdx] = [pname, status, joints] # dereference
                self.curJoints = self.joints[self.curPersonIdx][-1] # reference
                self.curJointIdx = -1
            # add | change label
            if self.curJointIdx >= 0 and key in [ord(c) for c in self.labels]:
                label = self.labels.index(chr(key))
                if label != self.curJoints[self.curJointIdx][0]:
                    for idx, joint in enumerate(self.curJoints):
                        if joint[0] >= label: self.curJoints[idx][0] += 1
                self.curJoints[self.curJointIdx][0] = self.labels.index(chr(key))
            # skip current joint
            if key == ord('\t'): # tab
                if len(self.joints):
                    if len(self.curJoints) < self.numJointsPerPerson:
                        joint = [-1, -1, -1, -1]  # invalid
                        joint[0] = len(self.curJoints)
                        self.curJoints.append(joint)
                else:
                    print 'Create a person object first! '

            # TODO: skip control
            # PC = (len(self.curJoints) == self.numJointsPerPerson)
            # FC = functools.reduce(lambda x, y: x * len(y[-1]) == self.numJointsPerPerson, self.joints, len(self.joints))
            PC = 1
            FC = 1

            # person-level
            # switch
            if key == ord(' ') and PC: # blankspace
                if len(self.joints):
                    self.curPersonIdx = (self.curPersonIdx + 1) % len(self.joints)
                    # self.curBufferJoints = self.bufferJoints[self.curPersonIdx][-1] # reference
                    self.curJoints = self.joints[self.curPersonIdx][-1] # reference
                    # print 'idx: %d, total: %d, SC: %d'%(self.curPersonIdx, len(self.joints), PC)
                    # print self.joints
            if key == ord('g') and PC: # add
                # self.bufferJoints.append([len(self.bufferJoints), -1, []])
                self.joints.append([len(self.joints), -1, []])
                self.curPersonIdx = (self.curPersonIdx + 1) % (len(self.joints))
                # self.curBufferJoints = self.bufferJoints[self.curPersonIdx][-1]
                self.curJoints = self.joints[self.curPersonIdx][-1]  # reference
                # print 'idx: %d, total: %d' % (self.curPersonIdx, len(self.joints))
            if key == ord('h'): # delete
                if len(self.joints):
                    # del self.bufferJoints[self.curPersonIdx]
                    del self.joints[self.curPersonIdx]
                if not len(self.joints):
                    # self.bufferJoints = []
                    # self.joints = []
                    self.curPersonIdx = -1
                    # self.curBufferJoints = []
                    self.curJoints = []
                else:
                    self.curPersonIdx = (self.curPersonIdx - 1) % (len(self.joints))
                    # self.curBufferJoints = self.bufferJoints[self.curPersonIdx][-1]
                    self.curJoints = self.joints[self.curPersonIdx][-1]  # reference
                print 'idx: %d, total: %d' % (self.curPersonIdx, len(self.joints))

            # frame-level
            if key == ord('d') and FC:
                self.draw_static(cidx)
                cidx = min(cidx + 1, len(self.imageList) - 1)
                self.update(cidx)
                print '%s' % (self.imageList[cidx])
            if key == ord('a') and FC:
                self.draw_static(cidx)
                cidx = max(cidx - 1, 0)
                self.update(cidx)
                print '%s' % (self.imageList[cidx])
            if key == 27 and FC: # esc
                self.draw_static(cidx)
                break

            self.update_frame()
            self.update_matte()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    template_prefix = './template_prefix.xml'
    template_person = './template_person.xml'
    template_object = './template_object.xml'
    parser = xmlParser.XMLParser(template_prefix, template_person, template_object)

    inputDir = '../inputs/'
    outputDir = '../outputs/'
    backupDir = '../backups/'
    pa = PoseAnnotation(inputDir, outputDir, backupDir)
    pa.xmlParser = parser
    pa.annotation()
