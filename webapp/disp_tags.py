#!/usr/bin/python
import sys
import cv2
import glob
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.misc import imread, imresize

sys.path.append('../')
from captioner1 import get_pred
from cap import pre_process
from captioner1 import vgg16
from caption_gen import get_captions
from imagenet_classes import class_names

def pre_process_video(fname):
    #print("In Pre Process")
    vidcap = cv2.VideoCapture(fname)
    num_frames = int(vidcap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    '''success,image = vidcap.read()
    count = 0
    success = True
    while success:
        success,image = vidcap.read()
        count += 1
    num_frames = count
    #print(num_frames)'''
    if(num_frames>10):
        num_frames /=10
    vidcap = cv2.VideoCapture(fname)
    success,image = vidcap.read()
    success = True
    count = 0
    count1 = 0
    base_name =  os.path.basename(fname)
    base_name = os.path.splitext(base_name)[0] 
    #print("file_name -> %s" % base_name)
    while success:
        success,image = vidcap.read()
        if(count%num_frames==0  and count1<10):
            #print(" writing image %s %d" % (base_name,count1))
            cv2.imwrite("./uploads/processed/%s_%d.jpg" % (base_name, count1), image)
            count1+=1
        count+=1


def get_imgs():
    path = "./uploads/processed/*.jpg"
    test_images = []
    files = glob.glob(path)
    for filename in files:
    	#print(filename)
    	test_images.append(filename)
    #print len(test_images)
    return test_images


if __name__ == '__main__':

	#print("Coming Soon!!!")

	video_path = "./uploads/*.avi"
	files = glob.glob(video_path)
	imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
	init_vars = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init_vars)	    

	for filename in files:
		pre_process_video(filename)
		imgs1 = get_imgs()
		imgs1 = pre_process(imgs1)
		#print(os.getcwd())
		vgg = vgg16(imgs, '../vgg16_weights.npz', sess)
		tags = get_pred(vgg,sess,imgs1)
		#print("got preds")
		#print len(tags)
		#print len(tags[0])
		
        out_file = "./uploads/tags.log"
        outf = open(out_file,"w")
        tagset = []
        for item in tags:
			for i in range(0,2):
				tagset.append(item[i][0])
        out = set(tagset)
        '''out = []
        window = 2
        item = 0
        while item < len(tags):
            j = item
            while j < item+window:
                for i in range(0,5):
                    tagset.append(tags[j][i][0])
                j+=1
            out.extend(set([x for x in tagset if tagset.count(x)>1]))
            item+=window

        out = set(out)'''

        #print(out)

        for item in out:
            outf.write(str(item)+".")
            #print(item) 
		#print tags
        outf.write("#")
        cap = get_captions(vgg,sess,imgs1)
        #print cap.shape
        cap = cap[0].split(" ")
        cap = set(cap)
        for item in cap:
            outf.write(item+".")
            #print(item)

        outf.close()
        outf = open(out_file,"r")
        data = outf.read()
        data = data.split("#")
        tags = data[0].split(".")    



