import os
import csv
import numpy as np
import sklearn
# import cv2
# from  scipy import ndimage
import imageio

# data_path = '/opt/carnd_p3/data/' # On GPU-enabled workspace
data_path = '~/opt/carnd_p3/data/' # On local machine

# Expand the path
data_path = os.path.expanduser(data_path)

# Data importing
samples = []
with open(data_path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Delete the first element in samples, since it's not a valid data
del samples[0]

# Moving averaging for steering angle
# N_h = 1
# N = 2*N_h + 1
# steering_angle_raw = [float(line[3]) for line in samples]
# _result = np.convolve(steering_angle_raw, np.ones((N,))/float(N), mode='full')
# steering_angle_averaged = _result[N_h:(-N_h)]
# print(len(steering_angle_raw))
# print(len(steering_angle_averaged))



# Split the train and test set 
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


# The batch generator
def generator(sample_list, batch_size=32, data_path="."):
    num_samples = len(samples)
    while True: # Eternal loop
        shuffle(sample_list)
        for offset in range(0, num_samples, batch_size):
            batch_samples = sample_list[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                current_path_center = data_path + batch_sample[0].split('/')[-1]
                # image_center = cv2.imread(current_path_center)
                # image_center = ndimage.imread(current_path_center)
                image_center = imageio.imread(current_path_center)
                angle_center = float(batch_sample[3])
                # Center
                images.append(image_center)
                angles.append(angle_center)
                # Flip
                # Right
                # Left
            # Convert to ndarray
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Training hyper parameters  
#--------------------------------------#
batch_size = 32
ch, row, col = 3, 160, 320 # Original image format


# Data generator
#--------------------------------------#
train_gen = generator(train_samples, batch_size=batch_size, data_path=data_path)
valid_gen = generator(validation_samples, batch_size=batch_size, data_path=data_path)

# Train network
#--------------------------------------#
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# Create the model
#--------------------------------------#
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
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
model.fit_generator(train_gen, /
                    step_per_epoch=ceil( len(train_samples)/batch_size), \
                    validation_data=valid_gen, \
                    validation_steps=ceil( len(validation_samples)/batch_size), \
                    epochs=5, verbose=1)

# Save the model
model.save('model.h5')
