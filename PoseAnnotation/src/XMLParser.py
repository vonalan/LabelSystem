import os
import sys
import copy
import xml.etree.ElementTree as et
from PIL import Image

class XMLParser(object):
    def __init__(self, tempalte_prefix, template_person, template_object):
        self.prefix_template = tempalte_prefix
        self.person_template = template_person
        self.object_template = template_object

    # TODO: besides boxes, the size of image also need to be returned to reconstruct the widht and height of boxes
    def parse_xml(self, filename):
        # TODO: add
        pass
        # tree = et.parse(filename)
        # root = tree.getroot()
        #
        # sizes = root.find('size')
        # width = int(sizes.find('width').text)
        # height = int(sizes.find('height').text)
        # sizes = (width, height)
        #
        # boxes = []
        # for obj in root.iter('object'):
        #     # labelname = obj.find('name').text
        #     label = int(obj.find('name').text)
        #     bndbox = obj.find('bndbox')
        #     xmin = int(bndbox.find('xmin').text)
        #     ymin = int(bndbox.find('ymin').text)
        #     xmax = int(bndbox.find('xmax').text)
        #     ymax = int(bndbox.find('ymax').text)
        #     boxes.append([label, xmin, ymin, xmax, ymax])
        # return sizes, boxes

    def write_xml(self, shape, names, joints, filepath):
        sizes = [shape[1], shape[0]]

        ptem = open(self.prefix_template)
        ptemline = ptem.read()
        ptem.close()
        ptemline = ptemline.replace('$width$', str(sizes[0]))
        ptemline = ptemline.replace('$height$', str(sizes[1]))

        qtem = open(self.person_template)
        qtemline= qtem.read()
        qtem.close()
        org_person = copy.copy(qtemline)

        otem = open(self.object_template)
        otemline = otem.read()
        otem.close()
        org_object = copy.copy(otemline)

        outfile = open(filepath, 'w')
        outfile.write(ptemline)

        for curJoints in joints:
            qtemline = copy.copy(org_person)
            qtemline = qtemline.replace('$name', str(curJoints[0]))
            outfile.write(qtemline)
            for joint in curJoints[-1]:
                otemline = copy.copy(org_object)
                otemline = otemline.replace('$name$', str(joint[0]))
                otemline = otemline.replace('$status$', str(joint[-1]))
                otemline = otemline.replace('$x$', str(joint[1]))
                otemline = otemline.replace('$y$', str(joint[2]))
                outfile.write(otemline)
            outfile.write('    </person>\n')
        outfile.write('</annotation>\n')
        outfile.close()

    def parse_txt_v1(self, filepath, num_features=4):
        joints = []
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            # TODO: if no intersection between two consecutive person
            history = []
            curJoints = [-1, -1, []]
            count = 0
            for i, line in enumerate(lines):
                line = [int(item) for item in line.strip().split(',')]
                if line[-1] in history:
                    curJoints[0] = count
                    joints.append(curJoints)
                    count += 1
                    curJoints = [-1, -1, []]
                curJoints[-1].append([-1, line[0], line[1], 1])
            if len(curJoints):
                joints.append(curJoints)
        except Exception:
            pass
        return joints

    def parse_txt(self, filepath, num_features=4):
        joints = []
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            for line in lines:
                line = [int(item) for item in line.strip().split('\t')]
                curJoints = [line[i*num_features:(i+1)*num_features] for i in range(len(line)/num_features)]
                joints.append(curJoints)
        except Exception:
            pass
        return joints

    def write_txt(self, shape, names, total_joints, filepath):
        with open(filepath, 'w') as f:
            for _, curJoints in enumerate(total_joints):
                joints = []
                for j, name in enumerate(names):
                    joint = [j, -1, -1, -1]

                    idx = -1
                    for k, jnt in enumerate(curJoints[-1]):
                        if j == jnt[0]:
                            idx = j
                    if idx >= 0: joint = curJoints[-1][idx]
                    joints.extend(joint)

                line = ' '.join([str(item) for item in joints])
                f.write(line)
                f.write('\n')

    # TODO: scale width and height of boxes to range of [0,1]
    def xml_to_txt(self, xml_path, txt_path):
        # TODO: add
        pass
        # sizes, boxes = self.parseXML(xml_path)
        # with open(txt_path, 'w') as f:
        #     for box in boxes:
        #         box = [str(i) for i in box]
        #         print box
        #         line = '\t'.join(box)
        #         f.write(line)
        #         f.write('\n')
        # return sizes

    # TODO: reconstruct width and height boxes from range of [0, 1]
    def txt_to_xml(self, sizes, txtname, xmlname):
        # TODO: add
        pass
        # with open(txtname, 'r') as f:
        #     lines = f.readlines()
        #     boxes = [[int(item) for item in line.strip().split()] for line in lines]
        #     print boxes
        # self.writeXML(sizes, boxes, xmlname)

if __name__ == '__main__':
    template_prefix = './template_prefix.xml'
    template_person = './template_person.xml'
    template_object = './template_object.xml'
    parser = XMLParser(template_prefix, template_person, template_object)

    image_dir = r'E:\Backups\Datasets\alphamatting.com\LD\input_training_lowres'
    xml_dir = r'E:\Backups\Datasets\alphamatting.com\LD\boxes_training_lowres\xmls'
    txt_dir = r'E:\Backups\Datasets\alphamatting.com\LD\boxes_training_lowres\txts'

    for image_name in os.listdir(image_dir):
        # image_path = os.path.join(image_dir, image_name)
        # img = Image.open(image_path)

        xml_path = os.path.join(xml_dir, image_name[:-4] + '.xml')
        txt_path = os.path.join(txt_dir, image_name[:-4] + '.txt')

        sizes = parser.xml_to_txt(xml_path, txt_path)
        parser.txt_to_xml(sizes, txt_path, xml_path)