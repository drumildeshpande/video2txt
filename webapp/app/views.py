import os
from app import app
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, Markup, flash
from werkzeug import secure_filename
import sys
import cv2
import glob
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.misc import imread, imresize
import subprocess

sys.path.append('../')
from captioner1 import get_pred
from cap import pre_process
from captioner1 import vgg16
from imagenet_classes import class_names

@app.route('/')
@app.route('/index')
def index():
	return render_template('index.html')

@app.route('/upload', methods=['POST','GET'])
def upload():
	if(request.method == 'POST'):
		
		#Save the video file

		f = request.files['myfile']
		filename = secure_filename(f.filename)
		f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		#return redirect(url_for('uploaded_file', filename = filename))
		status = "Uploaded File: "+filename
		status = Markup(status)
		flash(status)

		#Process the video to get tags
		tags = "Dummy"
		tags = subprocess.Popen('/home/paperspace/Documents/video2txt/webapp/disp_tags.py',shell=False, stdout=subprocess.PIPE)
		tags = tags.communicate()[0]

		print(os.getcwd())
		out_file = "/home/paperspace/Documents/video2txt/webapp/uploads/tags.log"
		data = open(out_file,'r').read()
		data = data.split("#")
		tags = data[0].split(".")
		cap = data[1].split(".")
		cap = " ".join(cap)
		return render_template('upload.html', status = status, tags = tags, cap=cap)
	return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
	return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
