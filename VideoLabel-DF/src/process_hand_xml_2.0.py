# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 10:58:05 2017

@author: Administrator
"""

import os
import pdb
import scipy.io as sio
from PIL import Image
import copy

def calcBoxes(boxes):
    num = boxes.shape[0]
    vocboxes = []
    for i in range(num):
        points = boxes[i][0][0]
        left = [5000,5000]
        right = [0,0]
        for j in range(4):
            if left[0] > points[j][0][0]:
                left[0] = points[j][0][0]
            if left[1] > points[j][0][1]:
                left[1] = points[j][0][1]
            if right[0] < points[j][0][0]:
                right[0] = points[j][0][0]
            if right[1] < points[j][0][1]:
                right[1] = points[j][0][1]
        vocboxes.append( [map(int,left), map(int, right)] )
    return vocboxes

    
def genXML(matdir, imgdir, xmldir, prefix_template, object_template):
    for name in os.listdir(matdir):
        if '.mat' not in name[-4:]:
            continue
        print name
        matname = matdir + name
        xmlname = xmldir + name[:-4] + '.xml'
        boxes = sio.loadmat(matname)['boxes'][0]
        img = Image.open(imgdir + name[:-4]+'.jpg')
        vocboxes = calcBoxes(boxes)
        writeXML(prefix_template, img.size, vocboxes, xmlname)

def ret2xml(retdir, imgdir, xmldir, prefix_template, object_template):
    for name in os.listdir(retdir):
        retname = retdir + name
        imgname = imgdir + name[:-4]
        xmlname = xmldir + name[:-4].replace('.png', '.xml')
        img = Image.open(imgname)
        imgsize = img.size
        retfile = open(retname)
        boxes = []
        names = []
        for line in retfile:
            line = line.replace('\n', '')
            items = line.split(',')
            name = items[1]
            xywh = map(float, items[3:])
            print xywh, imgsize
            box = [[0,0],[0,0]]
#            pdb.set_trace()
            box[0][1] = int((xywh[0]-xywh[2]/2)*imgsize[0])
            box[0][0] = int((xywh[1]-xywh[3]/2)*imgsize[1])
            box[1][1] = int((xywh[0]+xywh[2]/2)*imgsize[0])
            box[1][0] = int((xywh[1]+xywh[3]/2)*imgsize[1])
            boxes.append(box)
            names.append(name)
            print boxes, names
        retfile.close()
        writeXML(prefix_template, imgsize, names, boxes, xmlname)


def writeXML(imgsize, names, boxes, outname):
    ptem = open(prefix_template)
    ptemline = ptem.read()
    ptem.close()
    ptemline = ptemline.replace('$width$', str(imgsize[0]))
    ptemline = ptemline.replace('$height$', str(imgsize[1]))

    otem = open(object_template)
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



def txt_to_xml(imgdir, txtdir, outdir):
    for name in os.listdir(imgdir):
        imgname = imgdir + name
        txtname = txtdir + name[:-4] + '.txt'
        outname = outdir + name[:-4] + '.xml'
        img = Image.open(imgname)
        imgsize = img.size
        txtfile = open(txtname)
        names = []
        boxes = []
        for line in txtfile:
            line = line.replace('\n', '')
            line = line.replace('[', '')
            line = line.replace(']', '')
            line = line.replace("'", '')
            items = line.split(', ')
            names.append(items[0])
            boxes.append(map(int, items[1:]))
        txtfile.close()
        writeXML(imgsize, names, boxes, outname)







if __name__ == '__main__':

    '''
    prefix_template = 'd:/tmp/template_prefix.xml'
    object_template = 'd:/tmp/template_object.xml'

    imgdir = 'd:/tmp/output_images_dir/'
    txtdir = 'd:/tmp/output_points_dir/'
    outdir = 'd:/tmp/output_xmls_dir/'
    '''
    prefix_template = '/data6/wubing/data_extend/template_prefix.xml'
    object_template = '/data6/wubing/data_extend/template_object.xm.'

    imgdir = '/data6/wubing/data_extend/output_image_dir/'
    txtdir = '/data6/wubing/data_extend/output_point_dir/'
    outdir = '/data6/wubing/data_extend/output_xmls_dir'
    txt_to_xml(imgdir, txtdir, outdir)








