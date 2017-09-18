# -*- coding: utf-8 -*-

import numpy as np
import cv2

import copy

import workFrame as WF 
WorkFrame = WF.WorkFrame
Coordinate = WF.Coordinate


class VideoLabel(WorkFrame): 
    def __init__(self, name=''):
        super(VideoLabel_dev, self).__init__(name=name)
        
        self.wkFrames = None 
        self.curFrame = None 
        
        self.labels = []
        self.colors = []
        self.thick = 3
        self.linethick = 1
        self.font = cv2.FONT_HERSHEY_SIMPLEX  # 标注的字体样式
        self.fontsize = 1  # 标注字体大小
        self.labelRect = -1
        self.labelHight = 20
        self.labelWidth = 0

        self.startPos = Coordinate()
        self.choosedBox = -1 # which box is activated?
        self.choosedType = -1 # 0,框， 1， 角， 2，边
        self.mouseFlag = 0 # 0，自由；1，画框，2，移动框, 3, 移动角
        self.rightClick = 0
        self.selectedX = -1
        self.selectedY = -1 
    
    def update(self, curFrame=None): 
        pass 
    
    def backup(self, wkFrame=None): 
        pass 

    def update_box_imgs(self, thick=None):
        self.boxImage = np.zeros((self.shape[0], self.shape[1])) - 1

        th = self.thick
        for i, box in enumerate(self.boxes):
            rect = box.mainRect
            cmin, cmax = rect.min, rect.max
            self.boxImage[cmin.y:cmax.y, cmin.x:cmax.x] = i * 3 # [y1:y2,x1:x2]
            for _, sRect in enumerate(box.get_sub_rects(thick=self.thick)):
                scmin, scmax = sRect.min, sRect.max
                self.boxImage[scmin.y:scmax.y, scmin.x:scmax.x] = i * 3 + 1

    def draw_static(self, name=None, frame=None, shape=None, key=None, rects=None):
        '''write origin image'''
        # cv2.imwrite(self.outputimages + name, frame)

        for i, box in enumerate(self.boxes):
            rect = box.mainRect
            cmin, cmax = rect.min, rect.max
            cv2.rectangle(frame, (cmin.x, cmin.y), (cmax.x, cmax.y), (0, 255, 0), thickness=self.linethick)

            for _, sRect in enumerate(box.get_sub_rects(thick=self.thick)):
                scmin, scmax = sRect.min, sRect.max
                for th in range(self.thick):
                    cv2.rectangle(frame, (scmin.x,scmin.y), (scmax.x,scmax.y), (0, 255, 0), thickness=1)

            label, color = box.label, box.color
            coord = ((int(cmin.x+cmax.x)/2), cmax.y)
            cv2.putText(frame, label, coord, self.font, self.fontsize, color, 2, cv2.LINE_AA)

        '''write dbg image'''
        # cv2.imwrite(self.outputimages_dbg + name, frame)

        '''write xml file'''
        # boxes = [box.mainRect.get_points() for box in self.boxes]
        # labels = [box.label for box in self.boxes]
        # try:
        #     self.writeXML(shape, labels, boxes, self.outputxmls + name[:-4] + '.xml')
        # except IndexError:
        #     print('You forget label the category!')

    def show_labels(self, x, y):
        for i, name in enumerate(self.labels):
            for j in range(self.labelHight):
                cv2.line(self.frame, tuple((x, y + j)), tuple((x + self.labelWidth, y + j)), self.colorList[i],
                         thickness=1)

    def update_frame(self, x=-1, y=-1):
        # print('rectFlat:%s, chooseRect:%s, rect_len:%s' % (self.rectFlag, self.chooseRect, len(self.rects)))
        self.frame = copy.deepcopy(self.bufferframe)
        if self.mouseFlag == 1: # 画框
            ''''''
            rect = self.curBox.mainRect
            cmin, cmax = rect.min, rect.max
            print '画框...', [(self.choosedBox, self.choosedType), (self.selectedX, self.selectedY), [(cmin.x,cmin.y),(cmax.x,cmax.y)], (x,y)]
            cv2.rectangle(self.frame, (cmin.x, cmin.y), (x, y), (0, 255, 0), thickness=self.linethick)
            ''''''

        if self.choosedBox >= 0:
            box = self.boxes[self.choosedBox]
            rect = box.mainRect
            cmin, cmax = rect.min, rect.max
            print '选中框...', [(self.choosedBox, self.choosedType), (self.selectedX, self.selectedY), [(cmin.x,cmin.y),(cmax.x,cmax.y)], (x,y)]
            if self.choosedType == 0: # 移动框
                cv2.rectangle(self.frame, (cmin.x, cmin.y), (cmax.x, cmax.y), (255, 255, 255),
                              thickness=self.linethick + 2)
                print '移动框...', [(self.choosedBox, self.choosedType), (self.selectedX, self.selectedY), [(cmin.x,cmin.y),(cmax.x,cmax.y)], (x,y)]
                '''???'''
                if self.labelRect >= 0:
                    self.show_labels(x, y)
                '''???'''
            elif self.choosedType == 1: # 移动角
                pt = Coordinate(x=0, y=0)
                '''selectedX, selectedY'''
                if self.selectedX == -1:
                    pt.x = cmin.x if abs(cmin.x - x) <= self.thick else cmax.x
                    pt.y = cmin.y if abs(cmin.y - y) <= self.thick else cmax.y
                else:
                    x = cmax.x if self.selectedX == 1 else cmin.x
                    y = cmax.y if self.selectedY == 1 else cmin.y
                    pt.update(x=x, y=y)
                th = self.thick
                cv2.rectangle(self.frame, (pt.x-th, pt.y-th), (pt.x+th,pt.y+th), (255, 255, 255), thickness=1)
                print '移动角...', [(self.choosedBox, self.choosedType), (self.selectedX, self.selectedY), [(cmin.x,cmin.y),(cmax.x,cmax.y)], (x,y)]
            elif self.choosedType == 2: # 移动边
                pass

        # 画四角的小框以及大框
        for i, box in enumerate(self.boxes):
            rect = box.mainRect
            cmin, cmax = rect.min, rect.max
            cv2.rectangle(self.frame, (cmin.x,cmin.y), (cmax.x,cmax.y), (0, 255, 0), thickness=self.linethick)

            for _, pt in enumerate(rect.get_points()):
                for th in range(self.thick): # 加粗显示
                    cv2.rectangle(self.frame, (pt.x-th, pt.y-th), (pt.x+th, pt.y+th), (0, 255, 0), thickness=1)

            label, color = box.label, box.color
            coord = (int((cmin.x + cmax.x) / 2), cmax.y)
            cv2.putText(self.frame, label, coord, self.font, self.fontsize, color, 2, cv2.LINE_AA)

    def rect_done(self, x, y): # 画框完毕
        self.curBox.mainRect.max.update(x=x, y=y)
        self.curBox.mainRect.check(shape=(self.shape[0], self.shape[1]))
        self.boxes.append(self.curBox)
        self.update_box_imgs()
        self.update_frame()
        ''''''
        # self.curBox = BBox()
        ''''''

    def free_move(self, x, y):
        num = -1 if self.boxImage is None else int(self.boxImage[y, x])
        if num >= 0:
            self.frame = copy.deepcopy(self.bufferframe)
            self.choosedBox = int(num/3)
            self.choosedType = num % 3
            print(self.choosedBox, self.choosedType)
        else:
            self.choosedBox = -1
        self.update_frame(x, y)

    def select_conner(self, x, y):
        self.startPos = Coordinate(x=x, y=y)
        ''''''
        rect = self.boxes[self.choosedBox].mainRect
        cmin, cmax = rect.min, rect.max
        ''''''
        # [[0,0], [0,1], [1,0], [1,1]]
        self.selectedX = 0 if abs(cmin.x - x) <= self.thick else 1
        self.selectedY = 0 if abs(cmin.y - y) <= self.thick else 1

    def move_conner(self, x, y):
        deltaX = x - self.startPos.x
        deltaY = y - self.startPos.y

        self.startPos.update(x=x, y=y)
        rect = self.boxes[self.choosedBox].mainRect
        cmin, cmax = rect.min, rect.max

        ''''''
        # map selected to conner
        if self.selectedX == 0: cmin.x = x
        if self.selectedY == 0: cmin.y = y
        if self.selectedX == 1: cmax.x = x
        if self.selectedY == 1: cmax.y = x
        ''''''

    def draw_rect(self, event, x, y, flags, params):
        '''mouse'''
        if self.rightClick == 0: # ???
            if event == cv2.EVENT_LBUTTONUP:
                # print('LBUTTONUP', 'Flag:', self.mouseFlag, 'Rect:', self.choosedBox)
                # 画框的左上角时self.rectFla为0，画到右下角时self.rectFlag为1
                if self.mouseFlag == 0: # 0，自由；1，画框，2，移动框, 3, 移动边缘
                    '''if F is pressed'''
                    # if self.xc == 1:
                    self.curBox.mainRect.min.update(x=x, y=y)
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
                    ''''''
                    rect = self.curBox.mainRect
                    cmin, cmax = rect.min, rect.max
                    print '自由...', [(self.choosedBox, self.choosedType), (self.selectedX, self.selectedY),
                                    [(cmin.x, cmin.y), (cmax.x, cmax.y)], (x, y)]
                    ''''''
                    self.free_move(x, y)
                elif self.mouseFlag == 2: # 移动框
                    deltaX = x - self.startPos.x
                    deltaY = y - self.startPos.y
                    self.startPos.update(x=x, y=y)
                    ''''''
                    self.boxes[self.choosedBox].mainRect.move_all(deltaX=deltaX, deltaY=deltaY)
                    rect = self.boxes[self.choosedBox].mainRect
                    cmin, cmax = rect.min, rect.max
                    # print [[cmin.x,cmin.y],[cmax.x,cmax.y]]
                    self.boxes[self.choosedBox].mainRect.check(shape=(self.shape[0], self.shape[1]))
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
        else: pass

    def flow_rect(self):
        '''keyboard'''
        cv2.namedWindow('image', flags=cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("image", self.draw_rect)
        while True:
            cv2.imshow('image', self.frame)
            key = cv2.waitKey()

            if key in map(ord, [str(i) for i in range(10)]):
                if len(self.boxes)>0 and self.choosedBox >= 0:
                    self.boxes[self.choosedBox].label = str(key-48)
                    self.boxes[self.choosedBox].color = (0,255,0)
                    self.update_box_imgs()
                    self.update_frame()

            self.update_box_imgs()
            self.update_frame()
            if key == 27: break
        cv2.destroyAllWindows()

if __name__ == "__main__":
    wf = WorkFrame('../images/test.png')