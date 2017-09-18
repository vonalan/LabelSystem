# -*- coding: utf-8 -*-

import os 
import sys 

import cv2

import workFrame as WF


Coordinate = WF.Coordinate
Rectagle = WF.Rectagle
BBox = WF.BBox
WorkFrame = WF.WorkFrame
FrameBlock = WF.FrameBlock


class FrameFlow(object):
    def __init__(self, video=''):
        '''global variables block'''
        self.labels = [str(i) for i in range(10)] 
        self.colors = [] 
        self.length = 1

        self.nameList = []

        self.frameBlock = None
        self.curFrame = None

        self.sidx = 0
        self.eidx = 0
        self.cidx = 0

        '''path related block'''
        self.imgDir = ''
        self.xmlDir = ''
        self.vidDir = ''
        self._update_output_dirs_()

        ''''''
        # self.choosedBox = -1 # which box is activated?
        # self.choosedType = -1 # 0,框， 1， 角， 2，边
        self.mouseFlag = 0 # 0，自由；1，画框，2，移动框, 3, 移动角
        # self.selectedX = -1
        # self.selectedY = -1
        self.startPos = Coordinate()
        ''''''


        '''control block'''
        self.FC = 0 # control F
        self.AC = 1 # control A 
        self.DC = 1 # control D 
        self.UC = 1 # control Update
        self.XC = 1 # control Delete
        self.CC = 0 # check_frame_flow


        ''''''
        self.name = name
        self.checked = False

        self.labels = []
        self.colors = []

        self.frame = self._initialize_frame_()
        self.bufferframe = copy.deepcopy(self.frame)
        self.shape = self.frame.shape

        self.boxes = []
        self.curBox = BBox()
        self.boxImage = np.zeros((self.shape[0], self.shape[1])) - 1
        self.minBox = 25

        ''''''
        self.choosedBox = -1  # which box is activated?
        self.choosedType = -1  # 0,框， 1， 角， 2，边
        self.mouseFlag = 0  # 0，自由；1，画框，2，移动框, 3, 移动角
        self.selectedX = -1
        self.selectedY = -1
        ''''''

        self.thick = 3
        self.linethick = 1

        self.labelRect = -1
        self.labelHight = 20
        self.labelWidth = 0

        self.font = cv2.FONT_HERSHEY_SIMPLEX  # 标注的字体样式
        self.fontsize = 1  # 标注字体大小

        self.startPos = Coordinate()

        self.rightClick = 0


    def update(self):
        pass

    def _update_output_dirs_(self): 
        pass 
    
    def _extract_frames_(self): 
        pass

    # def update_frame(self):
    #     self.curFrame.update_box_imgs()
    #     pass

    def draw_rect(self, event, x, y, flags, params):
        '''mouse'''
        if event == cv2.EVENT_LBUTTONUP:
            # print('LBUTTONUP', 'Flag:', self.mouseFlag, 'Rect:', self.choosedBox)
            # 画框的左上角时self.rectFla为0，画到右下角时self.rectFlag为1
            if self.mouseFlag == 0: # 0，自由；1，画框，2，移动框, 3, 移动边缘
                '''if F is pressed'''
                # if self.xc == 1:
                self.curFrame.curBox.mainRect.min.update(x=x, y=y)
                self.mouseFlag = 1

                '''if F is pressed'''
            elif self.mouseFlag == 1: # 画框
                self.mouseFlag = 0
                rect = self.curBox.mainRect
                cmin, cmax = rect.min, rect.max
                if abs(x - cmin.x) > self.minBox or abs(y - cmin.y) > self.minBox:
                    self.rect_done(x, y)
                else:
                    pass
                    self.frame = copy.deepcopy(self.bufferframe)
                self.curBox = BBox()
            elif self.mouseFlag == 2 or self.mouseFlag == 3:
                # 移动框，角结束
                self.mouseFlag = 0
                self.selectedX = -1
                self.selectedY = -1
                ''''''
                self.boxes[self.choosedBox].mainRect.check(shape=(self.shape[0], self.shape[1]))
                self.update_box_imgs()
                self.update_frame()
                ''''''
        elif event == cv2.EVENT_MOUSEMOVE:
            # 首先触发该事件，若不进行其他操作:self.free_move -> self.update_frame(不会有其他任何操作)
            # print('EVENT_MOUSEMOVE', 'Flag:', self.mouseFlag, 'Rect:', self.choosedBox)
            if self.mouseFlag == 1: # 正在画框
                self.frame = copy.deepcopy(self.bufferframe)
                self.update_frame(x, y)
            elif self.mouseFlag == 0: # 鼠标自由移动
                self.curFrame.update_status(x=x, y=y, mouseFlag=self.mouseFlag)

            elif self.mouseFlag == 2: # 移动框
                deltaX = x - self.startPos.x
                deltaY = y - self.startPos.y
                self.curFrame.startPos.update(x=x, y=y)
                ''''''
                self.curFrame.boxes[self.curFrame.choosedBox].mainRect.move_all(deltaX=deltaX, deltaY=deltaY)
                rect = self.curFrame.boxes[self.curFrame.choosedBox].mainRect
                cmin, cmax = rect.min, rect.max
                print [[cmin.x,cmin.y],[cmax.x,cmax.y]]
                self.curFrame.boxes[self.curFrame.choosedBox].mainRect.check(shape=(self.shape[0], self.shape[1]))
                ''''''
                self.update_box_imgs()
                self.update_frame()

            elif self.mouseFlag == 3: # 移动角
                self.move_conner(x, y)
                self.update_box_imgs()
                self.update_frame(x, y)
            else: pass
        elif event == cv2.EVENT_LBUTTONDOWN:
            # print('EVENT_LBUTTONDOWN', 'Flag:', self.mouseFlag, 'Rect:', self.choosedBox)
            if self.mouseFlag == 0:
                if self.choosedBox >= 0:
                    if self.choosedType == 0: # 移动框
                        self.mouseFlag = 2
                        self.startPos.update(x=x, y=y)
                    elif self.choosedType == 1: # 移动角
                        self.mouseFlag = 3
                        self.select_conner(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            pass
        elif event == cv2.EVENT_RBUTTONUP:
            pass
        else: pass

    def flow_rect(self):

        self.nameList = ['../images/test.png', '../images/test.png']
        if len(self.nameList) > self.length: self.length = len(self.nameList)

        self.curFrame = WorkFrame(self.nameList[0])
        self.cidx = 0

        cv2.namedWindow('image', flags=cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("image", self.draw_rect)
        while (True):
            cv2.imshow('image', self.curFrame.frame)
            cv2.imshow('boximg', self.curFrame.boxImage)
            key = cv2.waitKey(20)

            # if key in map(ord, self.labels):
            #     print(str(key-48))
            #     self.curFrame.update_bbox()
            #     self.update_frame()
            #
            # if key in map(ord, ['g','a','d']):
            #     self.curFrame.draw_static()
            #     print('[G|A|D] -- Saving')
            #
            # if key == ord('b'):
            #     pass
            #
            # if key == ord('g'):
            #     if self.AC == 1 and self.DC == 1:
            #         self.FC = 1
            #         self.UC = 1
            #         self.AC = 0
            #         self.DC = 0
            #         self.XC = 0
            #         print("F -- Duplication. ")
            #
            #     if self.wkBlock:
            #         flags = self.wkBlock.check()
            #         if not flags:
            #             self.wkBlock.flush()
            #             self.nameList = self.nameList[self.length:]
            #             if self.length < len(self.nameList): self.length = len(self.nameList)
            #         else:
            #             self.cidx = flags[-1]
            #             self.curFrame = self.wkBlock.wkFrames[self.cidx]
            #             self.update_frame()
            #     if self.length == 0:
            #         # self.writeLog()
            #         break
            #
            #     self.wkBlock = FrameBlock(self.nameList[:self.length])
            #     self.cidx = self.wkBlock.length - 1
            #     self.curFrame = self.wkBlock.wkFrames[self.cidx]
            #     self.update_frame()
            #
            # if key == ord('a'):
            #     if self.FC == 1:
            #         if self.UC == 1:
            #             self.UC = 0
            #             self.XC = 1
            #             print("A -- Interpolation. ")
            #
            #             if self.wkBlock and len(self.curFrame.boxes) == len(self.wkBlock.wkFrames[0].boxes):
            #                 self.wkBlock.update()
            #         else:
            #             self.AC = 1
            #             print('A -- Last. ')
            #
            #             self.cidx -= 1
            #             if self.cidx < 0: self.cidx = 0
            #             self.curFrame = self.wkBlock.wkFrames[self.cidx]
            #             self.update_frame()
            #
            # if key == ord('d'):
            #     if self.FC == 1 and self.AC == 1:
            #         self.DC = 1
            #         print('D -- Next. ')
            #
            #         self.cidx += 1
            #         if self.cidx > self.wkBlock.length - 1: self.cidx = self.wkBlock.length - 1
            #         self.curFrame = self.wkBlock.wkFrames[self.cidx]
            #         self.update_frame()
            #
            # if key == ord('x'):
            #     if self.XC == 1:
            #         print('X -- Delete. ')
            #         self.curFrame.update_bbox()

            if key == 27:
                break 
        cv2.destroyAllWindows()

    def main(self):
        self._extract_frames_()

if __name__ == "__main__": 
    FF = FrameFlow()
    FF.flow_rect()