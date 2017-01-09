# -*- coding: utf-8 -*-
#!/usr/bin/env python

import os
import cv2
import fnmatch

dataset_path = '/data1/datasets/oxford_buildings/images/'

# walk through the folder
h = open("/home/deepinsight/blcf/oxford/list_oxford_horitzontal.txt", "w") # horitzontal
v = open("/home/deepinsight/blcf/oxford/list_oxford_vertical.txt", "w") # vertical

#dataset_path = '/media/disk1/DCC/video_first_frame_100W/'
#h = open("/home/yuanyong/data_sets/list_100w_horitzontal.txt", "w")
#v = open("/home/yuanyong/data_sets/list_100w_vertical.txt", "w")

for root, dirs, files in os.walk(dataset_path):
    for i, file in enumerate(files):
        # look for the files we want
        if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
            image = cv2.imread(os.path.join(dataset_path, file))
            (height, width, channel) = image.shape
            print "{}/{}: {}".format(i, len(files), file)
            if width >= height and channel == 3:
                h.write(os.path.join(dataset_path, file) + '\n')
            elif width < height and channel == 3:
                # width < hight
                v.write(os.path.join(dataset_path, file) + '\n')
            else:
                continue
h.close()
v.close()
