# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import cv2

bbox = [-1, [[-1,-1], [-1, -1]]]
boxes = bbox * 3

class LabelImage(object):
    def __init__(self):
        self.frame = None
        self.bufferFrame = None

        self.boxes = []
        self.curBoxIdx = -1
        self.curBox = []