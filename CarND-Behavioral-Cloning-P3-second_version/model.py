import csv 
import cv2 
import numpy as np
import keras 

lines = [] 
with open ('./data/driving_log.csv') as csvfile: 
    reader = csv.reader(csvfile) 
    
 
    
    for line in reader: 
        # do not append the first line with 4th being 'steeeing' 
        if line[3] != 'steering': 
            lines.append(line)
            #print(line) 
            


            
# take all photo to images 
# loop though line[0] line[1] line[2]
            
images = []
measurements = [] 
for line in lines: 
    for i in range(3):
        source_path = line[i] 
        filename = source_path.split('/')[-1]
        current_path = './data/IMG/' + filename
        image  = cv2.imread(current_path) 
        
        
        if line[3] != 0 and i != 0: # filtter steering = 0 out and keep some
        #if line[3] !=0:    
            images.append(image) 
            # OUTPUT labels
            if i == 0: # center camera
                measurement = float(line[3]) 
                measurements.append(measurement) 
            elif i == 1: # left camera
                measurement = float(line[3]) + 0.4
                measurements.append(measurement)
            else: # right camera 
                measurement = float(line[3]) - 0.4
                measurements.append(measurement)

            
print(len(images))
print(len(measurements))
    
 

 # Convert images and measurements into numpy arrays 

X_train = np.array(images)
y_train = np.array(measurements)


# Build a Keras regression network by minimizing MSE 

from keras.models import Sequential 
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D


model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3), output_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160, 320, 3)))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
model.add(Dropout(0.3))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
model.add(Dropout(0.5))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
history_obj = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7) 

model.summary() 
model.save('modelnew.h5')


from keras.models import Model
import matplotlib.pyplot as plt 



#print(history_obj.history.keys())


### plot the training and validation loss for each epoch
plt.plot(history_obj.history['loss'])
plt.plot(history_obj.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()















