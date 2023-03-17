# Packages to be used in this program
import numpy
import tensorflow as tf
from keras.applications import MobileNetV2
import cv2
import pickle
import numpy as np
import scipy.io as sio
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
import os 

#Load pretrained weights of deep learning model
model = MobileNetV2(weights='imagenet', include_top=True)

#Define dataset path
dataset_directory = "Dataset\Train"
dataset_folder = os.listdir(dataset_directory)

#Feature extractions
DatabaseFeautres = []
DatabaseLabel = []
cc=0
for dir_counter in range(0,len(dataset_folder)):
    cc+=1
    print('Processing class:   ', cc, 'of 5')
    single_class_dir = dataset_directory + "/" + dataset_folder[dir_counter]
    all_videos_one_class = os.listdir(single_class_dir) 
    for single_video_name in all_videos_one_class:        
        video_path = single_class_dir + "/" + single_video_name
        capture = cv2.VideoCapture(video_path)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        video_features = []
        frames_counter = -1
        while(frames_counter < total_frames-1):

            frames_counter+=1
            ret, frame = capture.read()
            if (ret):
                frame = cv2.resize(frame, (224,224))
                img_data = image.img_to_array(frame)
                img_data = np.expand_dims(img_data, axis=0)
                img_data = preprocess_input(img_data)
                single_featurevector = model.predict(img_data)
                video_features.append(single_featurevector)
                if frames_counter%30 == 29:
                    temp = np.asarray(video_features)
                    DatabaseFeautres.append(temp)
                    DatabaseLabel.append(dataset_folder[dir_counter])
                    video_features = []

TotalFeatures= []
OneHotArray = []
for sample in DatabaseFeautres:
    TotalFeatures.append(sample.reshape([1,30000]))


TotalFeatures = np.asarray(TotalFeatures)
TotalFeatures = TotalFeatures.reshape([len(DatabaseFeautres),30000])

OneHotArray = []
kk=1;
for i in range(len(DatabaseFeautres)-1):
    OneHotArray.append(kk)
    if (DatabaseLabel[i] != DatabaseLabel[i+1]):
        kk=kk+1;

with open("OneHotArray.pickle", 'wb') as f:
  pickle.dump(OneHotArray, f)
    
OneHot=  np.zeros([len(DatabaseFeautres),5], dtype='int');


for i in range(len(DatabaseFeautres)-1):
    print(i)
    OneHot[i,OneHotArray[i]-1] = 1



np.save('Training_Features',TotalFeatures)
sio.savemat('Training_labels.mat', mdict={'TrainLabels': OneHot})




