# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 21:47:29 2017

@author: Administrator
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,OneHotEncoder,LabelEncoder,LabelBinarizer
from keras.models import Sequential
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten,Activation,BatchNormalization,Input,MaxPooling1D
from keras.layers import Conv1D
from keras.optimizers import Adam,RMSprop,Adadelta
from keras.callbacks import ModelCheckpoint,LearningRateScheduler




num_classes = 12 
num_input = 1

def read(path):

    flat_list = []
    data_list = []
    f = open(path)
    for data in f.readlines():
            data_list.append(list(data.strip('\n'))) # read file
    f.close()
    seq_max_len = len(max(data_list,key=len))
     
    for sublist in data_list:
        for item in sublist:
            flat_list.append(item) 
   
    le = LabelEncoder()
    le.fit(flat_list) # make dictionary
    
    return le,seq_max_len,data_list
    
def preprocess(le,seq_max_len,data_list):   
    result = np.full((len(data_list), seq_max_len), 26)  # pad with 26
    max_len_array = np.zeros([len(data_list)], dtype=np.int32)
    for i, x_i in enumerate(result):
        result[i,:len(data_list[i])] = le.transform(data_list[i]) # transform char to int
        max_len_array[i] = len(data_list[i])
        
    enc = OneHotEncoder(n_values = 27)   # one hot encoding
    result = enc.fit_transform(result).toarray()    
    seq_max_len_ohe =  result.shape[1]  
    return seq_max_len_ohe,result
        
########################  read and pre-process x_test and x_train ########        
le,seq_max_len,x_train_list = read("xtrain_obfuscated.txt") 
_,_,x_test_list = read("xtest_obfuscated.txt")
max_len, x_train = preprocess(le,seq_max_len,x_train_list) 
_,x_test = preprocess(le,seq_max_len,x_test_list)
###########################################################################


#######################   read lablels  ###############################
y_train = []
f = open('ytrain.txt')
for temp in f.readlines():
        y_train.append(temp.strip('\n'))
f.close()
y_train = list(map(int, y_train))


lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
#######################################################################



####################  train validation split   ###########################
train_data, val_data, train_label, val_label = train_test_split(
            x_train, y_train, test_size=0.1, random_state=30)
##########################################################################


###################   reshape   #######################################
train_data = train_data.reshape((train_data.shape[0], max_len, num_input))
val_data = val_data.reshape((val_data.shape[0], max_len, num_input))
x_test = x_test.reshape((x_test.shape[0], max_len, num_input))
#######################################################################


batch_size = 128
clf_1 = Sequential()
clf_1.add(Conv1D(128, kernel_size=55,strides=27,activation='relu',input_shape=(max_len, num_input),kernel_initializer='glorot_normal',name='cnn1',trainable=True))
clf_1.add(Dropout(0.2))
clf_1.add(Conv1D(128, kernel_size=9,strides=4,activation='relu',name='cnn2',trainable=True))
#clf_1.add(MaxPooling1D(3))
clf_1.add(Dropout(0.2))
clf_1.add(Conv1D(128, kernel_size=3,strides=1,activation='relu',name='cnn3',trainable=True))
#clf_1.add(MaxPooling1D(3))
clf_1.add(Dropout(0.2))
clf_1.add(Conv1D(128, kernel_size=3,strides=1,activation='relu',name='cnn4',trainable=True))
clf_1.add(Dropout(0.2))
clf_1.add(Conv1D(128, kernel_size=3,strides=1,activation='relu',name='cnn5',trainable=True))
clf_1.add(Dropout(0.2)) 
clf_1.add(Flatten())
clf_1.add(Dense(30,kernel_initializer='glorot_normal',name='dense_30',trainable=True))
clf_1.add(Activation('relu')) 
clf_1.add(Dense(num_classes,kernel_initializer='glorot_normal',activation='softmax'))

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
clf_1.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer=adam)
filepath_1_cnn = "/tmp/weights.best_1_cnn.hdf5"
checkpoint_1_cnn = ModelCheckpoint(filepath_1_cnn, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list_1_cnn = [checkpoint_1_cnn]
#clf_1.load_weights("/tmp/weights.best_1_cnn_5_layer.hdf5")
clf_1.fit(train_data, train_label,validation_data=(val_data,val_label), epochs=30, batch_size=batch_size, verbose=2,callbacks=callbacks_list_1_cnn) # volume


y_pred = clf_1.predict_classes(x_test, batch_size=batch_size)
np.savetxt('y_prediction.txt', y_pred, fmt="%d")
