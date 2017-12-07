import os
import sys
import copy
import xml.etree.ElementTree as et
from PIL import Image

class XMLParser(object):
    def __init__(self, tempalte_prefix, template_object):
        self.prefix_template = tempalte_prefix
        self.object_template = template_object

    # TODO: besides boxes, the size of image also need to be returned to reconstruct the widht and height of boxes
    def parseXML(self, filename):
        tree = et.parse(filename)
        root = tree.getroot()

        sizes = root.find('size')
        width = int(sizes.find('width').text)
        height = int(sizes.find('height').text)
        sizes = (width, height)

        boxes = []
        for obj in root.iter('object'):
            # labelname = obj.find('name').text
            label = int(obj.find('name').text)
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            boxes.append([label, xmin, ymin, xmax, ymax])
        return sizes, boxes

    def writeXML(self, sizes, boxes, outname):
        # sizes = [shape[1], shape[0]]

        ptem = open(self.prefix_template)
        ptemline = ptem.read()
        ptem.close()
        ptemline = ptemline.replace('$width$', str(sizes[0]))
        ptemline = ptemline.replace('$height$', str(sizes[1]))

        otem = open(self.object_template)
        otemline = otem.read()
        otem.close()
        org_object = copy.copy(otemline)

        outfile = open(outname, 'w')
        outfile.write(ptemline)
        for i, box in enumerate(boxes):
            name = str(box[0])
            otemline = copy.copy(org_object)
            otemline = otemline.replace('$name$', name)
            otemline = otemline.replace('$xmin$', str(box[0]))
            otemline = otemline.replace('$xmax$', str(box[2]))
            otemline = otemline.replace('$ymin$', str(box[1]))
            otemline = otemline.replace('$ymax$', str(box[3]))
            outfile.write(otemline)
        outfile.write('</annotation>')
        outfile.close()

    # TODO: scale width and height of boxes to range of [0,1]
    def xml_to_txt(self, xml_path, txt_path):
        sizes, boxes = self.parseXML(xml_path)
        with open(txt_path, 'w') as f:
            for box in boxes:
                box = [str(i) for i in box]
                print box
                line = '\t'.join(box)
                f.write(line)
                f.write('\n')
        return sizes

    # TODO: reconstruct width and height boxes from range of [0, 1]
    def txt_to_xml(self, sizes, txtname, xmlname):
        with open(txtname, 'r') as f:
            lines = f.readlines()
            boxes = [[int(item) for item in line.strip().split()] for line in lines]
            print boxes
        self.writeXML(sizes, boxes, xmlname)

if __name__ == '__main__':
    template_prefix = './template_prefix.xml'
    template_object = './template_object.xml'
    parser = XMLParser(template_prefix, template_object)

    image_dir = r'E:\Backups\Datasets\alphamatting.com\LD\input_test_lowres'
    xml_dir = r'E:\Backups\Datasets\alphamatting.com\LD\boxes_test_lowres\xmls'
    txt_dir = r'E:\Backups\Datasets\alphamatting.com\LD\boxes_test_lowres\txts'

    for image_name in os.listdir(image_dir):
        # image_path = os.path.join(image_dir, image_name)
        # img = Image.open(image_path)

        xml_path = os.path.join(xml_dir, image_name[:-4] + '.xml')
        txt_path = os.path.join(txt_dir, image_name[:-4] + '.txt')

        sizes = parser.xml_to_txt(xml_path, txt_path)
        parser.txt_to_xml(sizes, txt_path, xml_path)