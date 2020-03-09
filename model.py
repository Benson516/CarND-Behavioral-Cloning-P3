import os
import csv
# import cv2
import numpy as np
# from  scipy import ndimage
import imageio

# data_path = '/opt/carnd_p3/data/'
data_path = '~/opt/carnd_p3/data/'

data_path = os.path.expanduser(data_path)

# Data importing
lines = []
with open(data_path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# Delete the first line in lines, since it's not a valid data
del lines[0]

# Moving averaging for steering angle
# N_h = 1
# N = 2*N_h + 1
# steering_angle_raw = [float(line[3]) for line in lines]
# _result = np.convolve(steering_angle_raw, np.ones((N,))/float(N), mode='full')
# steering_angle_averaged = _result[N_h:(-N_h)]
# print(len(steering_angle_raw))
# print(len(steering_angle_averaged))


images = []
measurements = []
print("Start loading data...")
for idx, line in enumerate(lines):
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = data_path + '/IMG/' + filename
    # image = cv2.imread(current_path)
    # image = ndimage.imread(current_path)
    image = imageio.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    # measurement = steering_angle_averaged[idx]
    measurements.append(measurement)
print("Finish loading data")

# Trainning data
#--------------------------------------#
X_train = np.array(images)
y_train = np.array(measurements)

# Train network
#--------------------------------------#
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# Create the model
model = Sequential()
model.add( Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)) )
model.add( Convolution2D(6,5,5,activation='relu') )
model.add( MaxPooling2D())
model.add( Convolution2D(16,5,5,activation='relu') )
model.add( MaxPooling2D())
model.add( Flatten() )
model.add( Dense(120) )
model.add( Dense(84) )
model.add( Dense(1) )

model.compile( loss='mse', optimizer='adam' )
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

# Save the model
model.save('model.h5')
