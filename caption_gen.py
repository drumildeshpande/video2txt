from keras.preprocessing import sequence
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Embedding , Activation
from keras.layers import LSTM
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from keras.layers.normalization import BatchNormalization
import keras
import numpy as np
import os
import sys
import tensorflow as tf
import pandas as pd
from captioner1 import get_features
from captioner1 import get_pred
from captioner1 import vgg16
from scipy.misc import imread, imresize
import glob

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
	vid_id = test_images[i].split("/")[3]
	#print vid_id
	vid_id = vid_id.split(".")[0].split("_")[0]
	test.append(vid_id)

    test = sorted(set(test))
    print test
    for i in range(0,len(test)):
    	for test_image in test_images:
	    if(test[i] in test_image):
		vid_id = test_image.split("/")[3]
		vid_id = vid_id.split(".")[0].split("_")[0]
		if(test[i]==vid_id):
    	    	    img1 = imread(test_image, mode='RGB')
    	    	    img1 = imresize(img1, (224, 224))
	    	    images.append(img1)
    print('total number of images %d' % len(images))
    return np.array(images)

def load_labels(test_images, vocab_size, time_steps):
    true_y = []
    desc_path = "./data/youtube/small_y2.npy"
    vid_desc = np.load(desc_path)
    #print vid_desc
    vocab_path = "./data/youtube/small_vocab2.npy"
    vocab = np.load(vocab_path)
    test = []
    for i in range(0,len(test_images)):
	#print test_images[i]
	vid_id = test_images[i].split("/")[3]
	#print vid_id
	vid_id = vid_id.split(".")[0].split("_")[0]
	test.append(vid_id)

    test = sorted(set(test))
    for vid_id in test:
	print vid_id
	vid_label = vid_desc.item()[vid_id]
	if(vid_label.endswith(".")):
	    vid_label = vid_label[:-1]
	words = vid_label.split(" ")
        temp_words = words
	
	#while len(words)+len(temp_words) < time_steps:
	#    words.extend(temp_words)

	words.extend(words)
	words.extend(words)
	words.extend(words)
	words.extend(words)
	words = words[0:time_steps]

	#for i in range(len(words),time_steps):
	#    words.append(".")
	print words
	y = np.zeros((vocab_size,),dtype=np.float32)
	y1 = []
	count = 0
	for word in words:
	    y[vocab.item()[word]] = 1.0
            y1.append(y)
	    count+=1
	'''for word in words:
	    y[vocab.item()[word]]=1.0
	for i in range(0,time_steps):
	    y1.append(y)'''
	true_y.append(np.asarray(y1))
    #print true_y
    return np.array(true_y)


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

def reshapeInput(comp_feats, time_steps):
    x_ip = []
    for feat in comp_feats:
	ip = []
	for i in range(0,time_steps):
	    ip.append(feat)
	x_ip.append(np.asarray(ip))

    return np.asarray(x_ip)

def get_word(embedding):
    vocab_path = "../data/youtube/small_vocab2.npy"
    vocab = np.load(vocab_path)
    it = -1
    idx = np.argmax(embedding)
    #print idx
    return vocab.item().keys()[vocab.item().values().index(idx)]


def print_pred(preds, batch_size):
    #out = ["" for i in range(0,batch_size)]
    out = []
    for pred in preds:
	sen = ""
	for word in pred:
	    sen += str(get_word(word))
	    sen += " "
	out.append(np.asarray(sen))

    out = np.asarray(out)
    return out


def create_model():
    model = Sequential()
    model.add(LSTM(num_hidden, input_shape=(max_time_steps,maxlen),return_sequences=True))
    model.add(Activation('tanh'))  

    model.add(LSTM(num_hidden, return_sequences=True))
    #model.add(BatchNormalization())
    model.add(Activation('tanh'))
    #model.add(Dropout(dropout_rate)) #recurrent_dropout=0.2))
    model.add(Dense(vocab_size))
    model.add(Activation(activation))

    # try using different optimizers and different optimizer configs
    #model.compile(loss=loss_func, optimizer=optimizer, metrics=[metric])
    return model

def get_captions(vgg,sess,images):
    batch = 1
    print os.getcwd()
    gen_cap = load_model("../"+model_name)
    features = get_features(vgg,sess,images)
    comp_feats = comp_features(images,features)
    x_ = reshapeInput(comp_feats, max_time_steps)
    pred = gen_cap.predict(x_,batch_size=batch,verbose=1)
    out_predictions = print_pred(pred, batch) 
    print out_predictions
    print "Okay"


if __name__ == '__main__':

    #___________________________________Load Vocab________________________________________#    

    vocab = np.load('./data/youtube/small_vocab2.npy')
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
        
    features = get_features(vgg,sess,images)
    comp_feats = comp_features(images,features)

    x_train = reshapeInput(comp_feats, max_time_steps)
    x_test = x_train

    train_images = load_data(sys.argv[1])
    y_train = load_labels(test_images,vocab_size,max_time_steps)

    #y_train = sequence.pad_sequences(y_train, max_time_steps)

    y_test = y_train

    print(len(x_train), 'train sequences')
    print(x_train.shape, 'train shape')

    print(len(y_train), 'train labels')
    print((y_train.shape), 'train shape')


    #___________________________________Build Model________________________________________#

    cur_layers = 1
    print('Build model...')
    model = Sequential()
    model.add(LSTM(num_hidden, input_shape=(max_time_steps,maxlen),return_sequences=True))
    model.add(Activation('tanh'))  

    model.add(LSTM(num_hidden, return_sequences=True))
    #model.add(BatchNormalization())
    model.add(Activation('tanh'))
    #model.add(Dropout(dropout_rate)) #recurrent_dropout=0.2))
    model.add(Dense(vocab_size))
    model.add(Activation(activation))

    # try using different optimizers and different optimizer configs
    model.compile(loss=loss_func, optimizer=optimizer, metrics=[metric])

    #___________________________________Train Model________________________________________#


    #keras.callbacks.ModelCheckpoint('./', monitor='acc', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=10)

    print('Train...')

    tbcallback = TensorBoard(log_dir = './tmp', histogram_freq = 10, write_graph=True, write_images = False)
    history = model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=num_iters, callbacks=[tbcallback]) # ,validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_train, y_train, batch_size=batch_size)
    
    print('Test score:', score)
    print('Test accuracy:', acc)

    model.save(model_name)

    #___________________________________Visualization________________________________________#


    ''' plt.subplot(2,1,1)
    print(history.history.keys())
    plt.plot(history.history['acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    # plt.legend()

    plt.subplot(2,1,2)
    print(history.history.keys())
    plt.plot(history.history['loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    #plt.legend() 
    plt.show()
    '''

    #___________________________________Test Model________________________________________#

    pred = model.predict(x_train, batch_size = batch_size, verbose=verbose_mode)
    print pred.shape

    out_predictions = print_pred(pred, batch_size)	
    print(out_predictions)
