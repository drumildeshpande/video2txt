# **Video 2 Text**

This projects tackles the problem of generating tags and caption for decribing videos. 
The motivation for the project was the huge amounts of video data available on different websites (youtube, facebook, twitter, imdb etc.). 
But there is no real end to end system that can generate tags or captions for videos. 
These tags and captions could be leveraged in building better recommnedation systems for video, video search, video summarization etc.

## **Pre processing and Data**

The data set used are:
1) IageNet dataset - which contains 1.2 million images form 1000 classes.
The dataset can be found [here](http://image-net.org/download-images).

2) Youtube video dataset - which contains ~2k videos with multiple captions describing them.

	The videos in the dataset were sampled into frames which were resized into 224*224.
The imagenet data set can be used to pre train VGG-16 model and store the weights.
The weight can be reloded into the model for computation later. You can read more about VGG model [here](http://www.robots.ox.ac.uk/~vgg/research/very_deep/).
The weight loading part is done using the vgg16.py file.

## **Models**

	### Model to generate Tags :
	![model_tags](https://github.com/drumildeshpande/video2txt/blob/master/images/model_tags.jpg "Model Tags")

	### Model to generate Caption :
	![model_caption](https://github.com/drumildeshpande/video2txt/blob/master/images/model_cap.jpg "Model Caption")
