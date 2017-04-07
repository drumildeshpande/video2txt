import numpy as np
from matplotlib import pyplot as plt
from tsne import bh_sne
import os
import sys
import glob
import tensorflow as tf
import pandas as pd
from captioner1 import get_features
from captioner1 import get_pred
from captioner1 import vgg16
from scipy.misc import imread, imresize
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition

model_name = "caption_mod.h5"
max_time_steps = 20 

def load_data(dir_name):
    test_images = []
    path = "./" + dir_name + "*.jpg"
    print path
    files = glob.glob(path)
    #print len(files)
    for filename in files:
    	#print(filename)
    	test_images.append(filename)
    #print len(test_images)
    return test_images

def pre_process(test_images):
    images = []
    test = []
    for i in range(0,len(test_images)):
	#print test_images[i]
	vid_id = test_images[i].split("/")[4]
	#print vid_id
	vid_id = vid_id.split(".")[0].split("_")[0]
	test.append(vid_id)

    test = sorted(set(test))
    #print test
    for i in range(0,len(test)):
	#print test[i]
    	for test_image in test_images:
	    if(test[i] in test_image):
		vid_id = test_image.split("/")[4]
		vid_id = vid_id.split(".")[0].split("_")[0]
		if(test[i]==vid_id):
    	    	    img1 = imread(test_image, mode='RGB')
    	    	    img1 = imresize(img1, (224, 224))
	    	    images.append(img1)
    print('total number of images %d' % len(images))
    return np.array(images)

def comp_features(images, features):
    #print images
    size = features.shape[0]
    feat = np.zeros((4096,),dtype=np.float32)
    comp_feat = []
    for i in range(0,size):
	if(i!=0 and i%10==0):
	    comp_feat.append(feat/10)
	    feat = np.zeros((4096,),dtype=np.float32)
	else:
	    #print('#################')
	    #print(feat.shape)
   	    #print(features[i].shape)
	    feat = np.add(feat,features[i])
    comp_feat.append(feat)
    return np.array(comp_feat)

if __name__ == '__main__':

    #___________________________________Load Vocab________________________________________#    

    vocab = np.load('./data/youtube/vocab.npy')
    print ('Length of current vocab is %d' % len(vocab.item()))

    #___________________________________Model Parameters________________________________________#

    maxlen = 4096
    vocab_size = len(vocab.item())
    num_hidden = 128
    num_layers = 2
    learning_rate = 0.001
    out_dim = vocab_size
    num_iters = 100
    batch_size = 10
    activation = "softmax" #"sigmoid" "relu"  
    loss_func =  "binary_crossentropy" #"mse"
    img_size = 224	#images are 224*224
    img_color = 3
    dropout_rate = 0.2
    metric = "accuracy"
    optimizer = "adam"
    verbose_mode = 1


    #___________________________________Placeholders________________________________________#

    imgs = tf.placeholder(tf.float32, [None, img_size, img_size, img_color])    

    #___________________________________Load Data________________________________________#

    print('Loading data...')

    
    init_vars = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_vars)	      
    test_images = load_data(sys.argv[1])
    images = pre_process(test_images)  

    vgg = vgg16(imgs, 'vgg16_weights.npz', sess)
    
    print images.shape
    it = 0
    features = []
    while (it+40) <= images.shape[0]:
	img1 = images[it:it+40]
	#print img1.shape
	features.extend(get_features(vgg,sess,img1))
	it+=40
    
    features = np.asarray(features)
    print features.shape
    #features = get_features(vgg,sess,images)
    comp_feats = comp_features(images,features)
    x_ = comp_feats.astype('float64')
    #x_ = reshapeInput(comp_feats, max_time_steps)

    print(x_.shape)

    vis_data = bh_sne(x_)

    vis_x = vis_data[:]
    vis_y = vis_data[:]

    y_ = np.zeros(1880)
    for i in range(1000,1200):
    	y_[i] = 1

    for i in range(1200,1880):
    	y_[i] = 2

    plt.scatter(vis_x, vis_y, c=y_, cmap = plt.cm.get_cmap("jet",3))
    plt.colorbar(ticks=range(3))
    plt.clim(-0.5, 9.5)
    plt.show()

