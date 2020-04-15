

from __future__ import print_function
import os
import numpy as np
from PIL import Image
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from IPython.core.debugger import Pdb
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, Flatten, Dense, Dropout
from keras.layers.convolutional import MaxPooling2D
    # from hyperdash import monitor_cell
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras import optimizers
   
pdb = Pdb()
batch_size = 32
num_classes = 2
epochs = 25
import matplotlib.pyplot as plt
# input image dimensions
img_rows, img_cols = 2816, 1880
row=2816
col=1880
#
path2 = '/home/vamshi/Documents/ch'  
list2 = os.listdir(path2)
num_samples2=np.size(list2)

wid=3
#Creating array of reference images  
x_train = np.array([np.array(Image.open('/home/vamshi/Documents/ch'+ '/' + im2)).flatten()
              for im2 in list2],'f')
    
x_train = x_train.astype('float8')/255
X_healthy = np.reshape(x_train, (len(x_train), row, col,wid))
path2 = '/home/vamshi/Documents/ch2'  
list2 = os.listdir(path2)
num_samples2=np.size(list2)

    
#Creating array of reference images  
x_train = np.array([np.array(Image.open('/home/vamshi/Documents/ch2'+ '/' + im2)).flatten()
              for im2 in list2],'f')
    
x_train = x_train.astype('float8')/255
X_diseased = np.reshape(x_train, (len(x_train), row, col,wid))
print(X_healthy.shape)
print(X_diseased.shape)
X=np.concatenate((X_healthy, X_diseased), axis=0)
print(X.shape)
Y=np.zeros(10)
for i in range(5,10):
	Y[i]=1
print(Y.shape)
print(type(Y))
seed = 7
#for j in range(242):
	#X[i,:,:,:]=X_diseased[j,:,:,:]
	#i=i+1
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#print(x_train.dtype)
#print(y_train.dtype)
#print(y_train.shape)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#pdb.set_trace()
# the data, split between train and test sets
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#print(x_train.shape)
#plt.imshow(X_train[400])
#plt.show()
#print(y_train[400])
#plt.show()
#
print(X_train.dtype)
print(y_train.dtype)
input_shape = (img_rows, img_cols, 3)
#pdb.set_trace()
#if K.image_data_format() == 'channels_first':
  #  x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
 #   x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
 #   input_shape = (1, img_rows, img_cols)
#else:
 #   x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
 #   x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#    input_shape = (img_rows, img_cols, 1)

#x_train = x_train.astype('float32')
#x_test = x_test.astype('float32')
#x_train /= 255
#x_test /= 255
#print('x_train shape:', x_train.shape)
#print(x_train.shape[0], 'train samples')
#print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(8, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(8, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(16, activation='softmax'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])








print(history.history.keys())

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
