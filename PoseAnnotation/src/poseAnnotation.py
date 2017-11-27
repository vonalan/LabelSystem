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

import numpy as np
import cv2

NUM_JOINTS = 15
JOINT_TEMPLATE = ['', -1, -1, -1]
# JOINTS_TEMPLATE = NUM_JOINTS * JOINT_TEMPLATE

class Joint(object):
    def __init__(self):
        self.label = ''
        self.pos = [0]



class PoseAnnotation(object):
    def __init__(self, inputsDir, outputDir):
        self.frame = None
        self.bufferFrame = None
        self.matte = None

        # TODO: len(self.joints) == num_persons * num_joints * len(joint_template)
        self.joints = []
        # self.bufferJoints = []
        # self.curJoint = [-1, -1, -1, -1]
        self.curJointIdx = -1

        # TODO: persons
        # self.persons = 0
        # self.curPersonIdx = -1

        self.names = []
        self.labels = []
        self.colors = []

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontsize = 0.75

        self.inputsDir = inputsDir
        self.outputDir = outputDir

        self.imageList = []

        # self._prepare_file_system()
        self._get_labels_and_colors()

    def prepare_file_system(self, inputsDir='../inputs/', outputsDir='../outputs/'):
        self.inputImageDir = os.path.join(self.inputsDir, 'images')
        self.inputAnnoDir = os.path.join(self.inputsDir, 'annotations')
        self.outputImageDir = os.path.join(self.outputDir, 'images')
        self.outputAnnoDir = os.path.join(self.outputDir, 'annotations')

        if not os.path.exists(self.outputImageDir): os.makedirs(self.outputImageDir)
        if not os.path.exists(self.outputAnnoDir): os.makedirs(self.outputAnnoDir)

    def _get_labels_and_colors(self):
        numbers = [str(i) for i in range(1, 10)]  # [1-10]
        letters = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p']  # [10-19]
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
        print(index)
        imagePath = os.path.join(self.inputImageDir, self.imageList[index])
        annoPath = os.path.join(self.inputAnnoDir, self.imageList[index][:-4] + '_auto_anno.txt')
        self.frame = cv2.imread(imagePath)
        self.bufferFrame = copy.copy(self.frame)

        try:
            with open(annoPath, 'r') as f:
                lines = [[int(item) for item in line] for line in
                         [line.strip().split(', ') for line in f]]
            self.joints = [[joint[2], joint[0], joint[1], -1] for joint in lines]
        except Exception:
            # joints = ['', -1, -1, -1]
            self.joints = []
        self.bufferJoints = []

    def update_matte(self):
        self.matte = np.zeros(self.frame.shape[:2]) - 7
        for idx, joint in enumerate(self.joints):
            color = 7 * idx
            cv2.circle(self.matte, (joint[1], joint[2]), 20, color, -1)

    def draw_static(self, index):
        image = self.imageList[index]
        imagePath = os.path.join(self.outputImageDir, image)
        annoPath = os.path.join(self.outputAnnoDir, image[:-4] + '_manual_anno.txt')

        # draw_joint_points
        frame = copy.copy(self.bufferFrame)
        for idx, joint in enumerate(self.joints):
            if joint[0] >= len(self.names):
                continue
            radius = 5
            thickness = -1
            if idx == self.curJointIdx: radius = 20
            # print joint
            cv2.circle(frame, (joint[1], joint[2]), radius, self.colors[idx % len(self.colors)], thickness)
            if joint[0] >= 0:
                cv2.putText(frame, str(joint[0] + 1), (joint[1], joint[2]), self.font, self.fontsize, (0, 0, 0), 1)
        # draw links between points
        for link in self.links:
            if link[1] >= len(self.labels):
                continue

            idx0, idx1 = -1, -1
            for idx, joint in enumerate(self.joints):
                if joint[0] == link[0]: idx0 = idx
                if joint[0] == link[1]: idx1 = idx
            # print(self.joints[idx0], self.joints[idx1])

            if idx0 >= 0 and idx1 >= 0:
                joint0, joint1 = self.joints[idx0], self.joints[idx1]
                color = [int(0.5 * (c1 + c2)) for c1, c2 in zip(self.colors[link[0]], self.colors[link[1]])]
                cv2.line(frame, (joint0[1], joint0[2]), (joint1[1], joint1[2]), color, 2)
        cv2.imwrite(imagePath, frame)

        with open(annoPath, 'w') as f:
            for i in range(len(self.names)):
                joint = [i, -1, -1, -1]

                idx = -1
                for j in range(len(self.joints)):
                    if i == self.joints[j][0]:
                        idx = j
                if idx >= 0: joint = self.joints[idx]

                line = str(joint).lstrip('[').rstrip(']')
                f.write(line)
                f.write('\n')

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

        # draw links between points
        for link in self.links:
            if link[1] >= len(self.labels):
                continue

            idx0, idx1 = -1, -1
            for idx, joint in enumerate(self.joints):
                if joint[0] == link[0]: idx0 = idx
                if joint[0] == link[1]: idx1 = idx

            if idx0 >= 0 and idx1 >= 0:
                joint0, joint1 = self.joints[idx0], self.joints[idx1]
                color = [int(0.5 * (c1 + c2)) for c1, c2 in zip(self.colors[link[0]], self.colors[link[1]])]
                cv2.line(self.frame, (joint0[1], joint0[2]), (joint1[1], joint1[2]), color, 2)

                center = int((joint0[1] + joint1[1]) / 2.0), int((joint0[2] + joint1[2]) / 2.0)
                mainAxis = int(((joint0[1] - joint1[1]) ** 2 + (joint0[2] - joint1[2]) ** 2) ** 0.5 * 0.5)
                subAxis = 3
                algle = int(math.degrees(math.atan2(joint0[2] - joint1[2], joint0[1] - joint1[1])))

                polygon = cv2.ellipse2Poly(center, (mainAxis, subAxis), algle, 0, 360, 1)
                cv2.fillConvexPoly(self.frame, polygon, color)

        # draw_joint_points
        for idx, joint in enumerate(self.joints):
            radius = 5
            thickness = -1
            if idx == self.curJointIdx: radius=20
            cv2.circle(self.frame, (joint[1], joint[2]), radius, self.colors[idx % len(self.colors)],thickness)
            # if joint[3] == 0:
            #     cv2.rectangle(self.frame, (joint[1] - 10, joint[2] - 10), ((joint[1] + 10, joint[2] + 10)),
            #                   self.colors[idx % len(self.colors)], 1)
            if joint[0] >= 0:
                label = str(joint[0] + 1)
                bgColor = self.colors[idx % len(self.colors)]
                fgColor = (0, 0, 0)
                if joint[3] == 0:
                    bgColor = (0, 0, 0)
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
            joint = [-1, x, y, 1] # visible
            self.joints.append(joint)
        if event == cv2.EVENT_RBUTTONUP:
            joint = [-1, x, y, 0]  # invisible
            self.joints.append(joint)
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
        cv2.setMouseCallback('image', self.call_back_func)
        while True:
            cv2.imshow('image', self.frame)
            cv2.imshow('matte', self.matte)
            key = cv2.waitKey(20)

            # delete
            if self.curJointIdx >= 0 and key == ord('x'):
                label = self.joints[self.curJointIdx][0]
                del self.joints[self.curJointIdx]
                for idx, joint in enumerate(self.joints):
                    if joint[0] > label: self.joints[idx][0] -= 1
                self.curJointIdx = -1

            # add label
            if self.curJointIdx >= 0 and key in [ord(c) for c in self.labels]:
                label = self.labels.index(chr(key))
                if label != self.joints[self.curJointIdx][0]:
                    for idx, joint in enumerate(self.joints):
                        if joint[0] >= label: self.joints[idx][0] += 1
                self.joints[self.curJointIdx][0] = self.labels.index(chr(key))

            # TODO: persons
            # switch person
            if key == 100: # blankspace
                # self.curPersonIdx = (self.curPersonIdx + 1 % len(self.persons))
                pass

            if key == ord('d'):
                cidx = min(cidx + 1, len(self.imageList) - 1)
                self.update(cidx)
            if key == ord('a'):
                cidx = max(cidx - 1, 0)
                self.update(cidx)
            if key == 27:
                self.draw_static(cidx)
                break

            self.update_frame()
            self.update_matte()

if __name__ == '__main__':
    inputDir = '../inputs/'
    outputDir = '../outputs'
    pa = PoseAnnotation(inputDir, outputDir)
    pa.annotation()
