# -*- coding: utf-8 -*-

import numpy as np
import cv2

import copy


class Coordinate(object): 
    '''2-D Coordinates'''
    def __init__(self, x=-1, y=-1):
        self.x = x 
        self.y = y

    def to_tensor(self):
        '''1-D tensor'''
        tensor = [self.x, self.y]
        return tensor

    def from_tensor(self, tensor):
        self.x = tensor[0]
        self.y = tensor[1]

    def update(self,x=-1,y=-1):
        self.x = x
        self.y = y


class Rectagle(object):
    '''
    Rectagle
    [[xmin, ymin], [xmax, ymax]]
    '''
    def __init__(self, xmin=-1, ymin=-1, xmax=-1, ymax=-1, gap=25):
        self.min = Coordinate(x=xmin, y=ymin)
        self.max = Coordinate(x=xmax, y=ymax)
        self.gap = gap

    def to_tensor(self):
        '''2-D tensor'''
        tensor = [self.min.to_tensor(), self.max.to_tensor()]
        return tensor

    def from_tensor(self, tensor):
        self.min.from_tensor(tensor[0])
        self.max.from_tensor(tensor[1])

    def get_points(self):
        '''
        [[0,0],[0,1],[1,0],[1,1]]
        '''
        points = []
        points.append(self.min)
        points.append(Coordinate(x=self.min.x, y=self.max.y))
        points.append(Coordinate(x=self.max.x, y=self.min.y))
        points.append(self.max)
        return points

    def move_vertex(self):
        pass

    def move_edge(self):
        pass

    def move_all(self, deltaX=0, deltaY=0):
        self.min.x += deltaX
        self.min.y += deltaY
        self.max.x += deltaX
        self.max.y += deltaY

    def update(self, xmin=0, ymin=0, xmax=0, ymax=0):
        self.min.x = xmin
        self.min.y = ymin
        self.max.x = xmax
        self.max.y = ymax

    def check(self, shape=None):
        '''
        min.x >= 0 && min.x <= max.x
        min.y >= 0 && min.y <= max.y
        '''
        # swap min and max if neccessary
        if self.min.x > self.max.x:
            self.min.x, self.max.x = self.max.x, self.min.x
        if self.min.y > self.max.y:
            self.min.y, self.max.y = self.max.y, self.min.y

        # we know that self.max.x is 0 when self.mix.x lt 0
        if self.min.x < 0:
            self.min.x = 0
            self.max.x = self.gap
        # we know that self.max.x is 0 when self.mix.x lt 0
        if self.min.y < 0:
            self.min.y = 0
            self.max.y = self.gap
        # we know that self.min.x is shape[1] when self.max.x gt shape[1]
        if self.max.x > shape[1]:
            self.max.x = shape[1]
            self.min.x = shape[1] - self.gap
        # we know that self.min.y is shape[0] when self.max.y gt shape[0]
        if self.max.y > shape[0]:
            self.max.y = shape[0]
            self.min.y = shape[0] - self.gap


class BBox(object):
    '''
    Bounding box
    '''
    def __init__(self, xmin=-1, ymin=-1, xmax=-1, ymax=-1, label='', color=(0,0,0)):
        self.label = label
        self.color = color
        self.mainRect = Rectagle(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax) # [(xmin, ymin), (xmax, ymax)]
        # self.subRects = self.get_sub_rects() # 4*[(xmin, ymin), (xmax, ymax)]

    def get_sub_rects(self, thick=None):
        subRects = []
        for pt in self.mainRect.get_points():
            xmin = pt.x - thick 
            ymin = pt.y - thick 
            xmax = pt.x + thick 
            ymax = pt.y + thick 
            rect = Rectagle(xmin, ymin, xmax, ymax)
            subRects.append(rect)
        return subRects
    

class WorkFrame(object):
    def __init__(self, name=''):
        self.name = name
        self.checked = False

        self.labels = []
        self.colors = []

        self.frame = self._initialize_frame_()
        self.bufferframe = copy.deepcopy(self.frame)
        self.shape = self.frame.shape

        self.boxes = []
        self.curBox = BBox()
        self.boxImage = np.zeros((self.shape[0],self.shape[1])) - 1
        self.minBox = 25

        self.choosedBox = -1 # which box is activated?
        self.choosedType = -1 # 0,框， 1， 角， 2，边
        self.mouseFlag = 0 # 0，自由；1，画框，2，移动框, 3, 移动角

        ''''''
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

    def to_tensor(self):
        '''3-d tensor'''
        tensor = []
        for i, box in enumerate(self.boxes):
            tensor.append(box.mainRect.to_tensor())
        return tensor

    def from_tensor(self, tensor):
        for i, (box, t)in enumerate(zip(self.boxes, tensor)):
            box.mainRect.from_tensor(t)

    def show(self):
        cv2.namedWindow(self.name,cv2.WINDOW_AUTOSIZE)
        while(True): 
            cv2.imshow(self.name, self.frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    def _initialize_frame_(self, nameList=None):
        frame = cv2.imread(self.name)
        return frame 

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

class FrameBlock(object):
    def __init__(self, nameList=None):
        self.nameList = nameList
        self.wkFrames = self._initialize_frames_()
        self.length = len(self.wkFrames)

    def _initialize_frames_(self):
        wkFrames = []
        for idx, name in self.nameList: 
            frame = WorkFrame(name)
            wkFrames.append(frame)
        return wkFrames
    
    def update_old(self, sFrame=None, eFrame=None, steps=1):
        '''interoplation'''
        sFrame = self.wkFrames[0]
        eFrame = self.wkFrames[-1]
        steps = self.length
        assert len(sFrame.boxes) == len(eFrame.boxes)

        dFrame = copy.deepcopy(eFrame)
        for dbox, sbox, ebox in zip(dFrame.boxes, sFrame.boxes, eFrame.boxes):
            drect, srect, erect = dbox.mainRect, sbox.mainRect, ebox.mainRect
            drect.min.x = (erect.min.x - srect.min.x)/float(steps)
            drect.min.y = (erect.min.y - srect.min.y)/float(steps)
            drect.max.x = (erect.max.x - srect.max.x)/float(steps)
            drect.max.y = (erect.max.y - srect.max.y)/float(steps)

        for steps, cFrame in enumerate(self.wkFrames):
            if steps == 0 or steps == self.length - 1: continue
            for dbox, sbox, ebox in zip(dFrame.boxes, sFrame.boxes, eFrame.boxes):
                drect, srect, erect = dbox.mainRect, sbox.mainRect, ebox.mainRect
                erect.min.x = int(srect.min.x + drect.min.x * steps)
                erect.min.y = int(srect.min.y + drect.min.y * steps)
                erect.max.x = int(srect.max.x + drect.max.x * steps)
                erect.max.y = int(srect.max.y + drect.max.y * steps)

    def update(self):
        sArray = np.array(self.wkFrames[0].to_tensor())
        eArray = np.array(self.wkFrames[-1].to_tensor())
        dArray = (eArray - sArray)/float(self.length)

        for steps, cFrame in enumerate(self.wkFrames):
            if steps == 0 or steps == self.length - 1: continue
            cArray = sArray + dArray * steps
            cFrame.from_tensor(cArray.tolist())

    def flush(self): 
        '''flush buffer'''
        for idx, frame in enumerate(self.wkFrames):
            frame.draw_static()
        pass

    def check(self):
        flags = []
        for idx, frame in enumerate(self.wkFrames):
            if frame.checked == False:
                flags.append(idx)
        return flags

if __name__ == "__main__":
    wf = WorkFrame('../images/test.png')
    box = BBox(xmin=0,ymin=0,xmax=30,ymax=30)
    wf.boxes.append(box)
    wf.boxes.append(box)
    wf.boxes.append(box)
    t = wf.to_tensor()

    t = np.array(t)
    print t.shape
    print (t)

    t = t.tolist()
    for i in t:
        print i