import numpy as np
import os
import sys
import tensorflow as tf
import pandas as pd
#import keras
#import tensorflow.nn.rnn_cell
from captioner1 import get_features
from captioner1 import get_pred
from captioner1 import vgg16
from scipy.misc import imread, imresize
import glob

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

'''def load_labels(test_images, vocab_size):
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

    test = set(test)
    for vid_id in test:
	#print vid_id
	vid_label = vid_desc.item()[vid_id]
	if(vid_label.endswith(".")):
	    vid_label = vid_label[:-1]
	words = vid_label.split(" ")
	words.append(".")
	y = np.zeros((vocab_size,),dtype=np.float32)
	for word in words:
	    y[vocab.item()[word]]=1.0
	prev_id = vid_id
	true_y.append(y)
    #print true_y
    return np.array(true_y)
'''

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
        #print test
    for i in range(0,len(test)):
        for test_image in test_images:
            if(test[i] in test_image):
                vid_id = test_image.split("/")[3]
                vid_id = vid_id.split(".")[0].split("_")[0]
                if(test[i]==vid_id):
                    img1 = imread(test_image, mode='RGB')
                    img1 = imresize(img1, (224, 224))
                    images.append(img1)
    #print('total number of images %d' % len(images))
    return np.array(images)
        
def print_pred(preds):
    for pred in preds:
	print('---------------------------------------------------------------------')
	for item in pred:	
	    print(item)

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

def transform(x, time_steps):
    x_input = []
    for i in range(0,x.shape[0]):
	temp = []
	for j in range(0,time_steps):
	    temp.append(x[i])
	x_input.append(np.asarray(temp))
    return np.asarray(x_input)

def get_word(embedding):
    vocab_path = "./data/youtube/small_vocab2.npy"
    vocab = np.load(vocab_path)
    it = -1
    idx = np.argmax(embedding)
    #print idx
    return vocab.item().keys()[vocab.item().values().index(idx)]

def parse_output(outputs, batch_size):
    out = ["" for i in range(0,batch_size)]
    for output in outputs:
	for i in range(0,output.shape[0]):
	   word = get_word(output[i])
	   out[i]+= word
	   out[i]+= " "
    print out


def RNN_node(x,wt,b):
    print("In def RNN")
    #print(x.get_shape())
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_hidden, forget_bias=1.0,state_is_tuple = True)
    multi_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*num_layers, state_is_tuple=True)
    outputs, states = tf.nn.rnn(multi_lstm, x, dtype=tf.float32)
    #print(outputs.get_shape())
    print(len(outputs))
    print(outputs[0].get_shape())
    print("-----------------")
    return [tf.matmul(output,wt['out']) + b['out'] for output in outputs]
    #return tf.matmul(outputs[-1],wt['out']) + b['out']

if __name__ == '__main__':
    
    vocab = np.load('./data/youtube/small_vocab2.npy')
    print ('Length of current vocab is %d' % len(vocab.item()))


    ##### Params ##########################################################  
    
    vocab_size = len(vocab.item())
    num_hidden = 24
    num_layers = 2
    max_time_steps = 20 
    learning_rate = 0.00001
    out_dim = vocab_size
    num_iters = 1000
    batch_size = 10

    ##### Placeholders ####################################################
    
    x = tf.placeholder(tf.float32,[None,max_time_steps,4096])
    y = tf.placeholder(tf.float32,[None,max_time_steps,out_dim])
    dropout = tf.placeholder(tf.float32)
    data = tf.placeholder(tf.float32,[None, max_time_steps, 4096])
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])


    ##### Weights #########################################################

    wt = {'out': tf.Variable(tf.random_normal([num_hidden, out_dim]))}  
    b = {'out' : tf.Variable(tf.random_normal([out_dim]))}
        
    #######################################################################
    
    softmax_w = tf.get_variable("softmax_w",[4096,num_hidden])
    softmax_b = tf.get_variable("softmax_b",[num_hidden])
    _X = tf.transpose(x,[1,0,2])
    _X = tf.reshape(_X,[-1,4096])
    print _X.get_shape()
    _X = tf.nn.relu(tf.matmul(_X,softmax_w)+softmax_b)
    _X = tf.split(0,max_time_steps,_X) 
    print (_X[0].get_shape())   
    pred = RNN_node(_X,wt,b)
    print pred
    #print(pred.get_shape())
    print(len(pred))
    print(pred[0].get_shape())
    #logits_pred = tf.nn.softmax(pred)
    logits_pred = tf.transpose(pred,[1,0,2])
    print logits_pred.get_shape()
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_pred, labels=y))
    print loss.get_shape()
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    correct_pred = tf.equal(tf.argmax(logits_pred,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    

    init_vars = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_vars)	      
    test_images = load_data(sys.argv[1])
    true_y = load_labels(test_images,vocab_size,max_time_steps)
    print true_y.shape
    #mask_y = get_masks(true_y)
    
    #true_y = true_y.transpose(1,0,2)  
    #print true_y.shape 

    images = pre_process(test_images)  
    #print images.shape
    #print images.dtype
    vgg = vgg16(imgs, 'vgg16_weights.npz', sess)
    
    #pred1 = get_pred(vgg,sess,images)
    #print len(pred)
    #print_pred(pred1)
    
    features = get_features(vgg,sess,images)
    comp_feats = comp_features(images,features)
    #print comp_feats
    print comp_feats.shape

    x_input = transform(comp_feats,max_time_steps)
    #print x_input
    print x_input.shape    

    tf.scalar_summary('loss',loss)
    tf.scalar_summary('accuracy',accuracy)
    summary_op = tf.merge_all_summaries()

    log_path = "./tmp"
    writer = tf.train.SummaryWriter(log_path, graph = tf.get_default_graph())
	

    print("__________________ Train _____________________________________")
    epoch = 0
    while epoch <= num_iters:
	#sess.run(optimizer, feed_dict={x: x_input, y: true_y}) 
	_,net_loss,acc,out,summary = sess.run([optimizer,loss,accuracy, pred, summary_op], feed_dict={x: x_input, y:true_y})
	#net_loss = sess.run(loss, feed_dict={x: x_input, y:true_y})
	#print out[0]    
	if(epoch%100==0):	
            print("------------------Mini Batch -----------------------")
	    print("Accuracy: %.6f" % acc)
	    #print(out)
	    print(out[0].shape)
	    parse_output(out, batch_size)
	    print("Loss: %.6f " % net_loss)
	    writer.add_summary(summary, epoch)
	epoch+=1
 
    
    print("Done")





