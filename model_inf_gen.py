import os
import csv
import numpy as np
import sklearn
#---#
# import cv2
# from  scipy import ndimage
import imageio
#---#
import matplotlib.pyplot as plt

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

#
# Get a subset of data for rapid testing of the script funtionality
# num_sample = 100
# sklearn.utils.shuffle(samples)
# samples = samples[:num_sample]
#

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


# The batch generator with augmentaiton
def generator_aug(sample_list, batch_size=32, yield_size=None, data_path=".", aug_list=[], beta=0.2):
    # Process the augmentation list
    is_center_flip = "center_flip" in aug_list
    is_right = "right" in aug_list
    is_left = "left" in aug_list
    #
    if yield_size is None:
        yield_size = int(batch_size*1.5)
    #
    num_samples = len(sample_list)
    images = []
    angles = []
    while True: # Eternal loop
        sample_list = sklearn.utils.shuffle(sample_list)
        #
        # Loop over all samples
        for sample in sample_list:
            current_path_center = data_path + '/IMG/' + sample[0].split('/')[-1]
            current_path_left = data_path + '/IMG/' + sample[1].split('/')[-1]
            current_path_right = data_path + '/IMG/' + sample[2].split('/')[-1]
            # image_center = cv2.imread(current_path_center)
            # image_center = ndimage.imread(current_path_center)
            image_center = imageio.imread(current_path_center)
            angle_center = float(sample[3])
            # Center
            images.append(image_center)
            angles.append(angle_center)

            # Augmentation
            #-------------------------------#
            # Flip
            if is_center_flip:
                image_center_flip = np.fliplr(image_center)
                images.append(image_center_flip)
                angles.append(-1*angle_center)
            # Right
            if is_right:
                image_right = imageio.imread(current_path_right)
                # angle_right = angle_center * ( (1.0-beta) if angle_center > 0.0 else (1.0+beta))
                angle_right = angle_center - beta
                images.append(image_right)
                angles.append(angle_right)
            # Left
            if is_left:
                image_left = imageio.imread(current_path_left)
                # angle_left = angle_center * ( (1.0-beta) if angle_center < 0.0 else (1.0+beta))
                angle_left = angle_center + beta
                images.append(image_left)
                angles.append(angle_left)
            #-------------------------------#

            # yield
            if len(images) >= yield_size:
                # yield
                # Shuffle
                images, angles = sklearn.utils.shuffle(images, angles)
                # Convert to ndarray
                X_train = np.array(images[:batch_size])
                y_train = np.array(angles[:batch_size])
                # yield sklearn.utils.shuffle(X_train, y_train)
                yield (X_train, y_train)
                images = images[batch_size:]
                angles = angles[batch_size:]
            #


# Training hyper parameters
#--------------------------------------#
aug_list = []
# aug_list += ['center_flip']
aug_list += ['right']
aug_list += ['left']
#
batch_size = 32
num_epoch = 10 # 5
#
aug_multiple = 1 + len(aug_list)
ch, row, col = 3, 160, 320 # Original image format


# Data generator
#--------------------------------------#
train_gen = generator_aug(train_samples, batch_size=batch_size, data_path=data_path, aug_list=aug_list)
valid_gen = generator_aug(validation_samples, batch_size=batch_size, data_path=data_path, aug_list=[])



# Train network
#--------------------------------------#
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, LeakyReLU, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras import regularizers

# Create the model
#--------------------------------------#
model = Sequential()
model.add( Cropping2D(cropping=((50,20), (0,0)), input_shape=(row, col, ch) ) )
model.add( Lambda( lambda x: x / 255.0 - 0.5 ) )
# 1st model
# model.add( Convolution2D(6,5,5,activation=None) )
# model.add( LeakyReLU(alpha=0.2) )
# model.add( MaxPooling2D())
# model.add( Dropout(rate=0.5) )
# model.add( Convolution2D(16,5,5,activation=None) )
# model.add( LeakyReLU(alpha=0.2) )
# model.add( MaxPooling2D())
# model.add( Dropout(rate=0.5) )
# model.add( Flatten() )
# model.add( Dense(120, kernel_regularizer=regularizers.l2(0.01) ) )
# model.add( LeakyReLU(alpha=0.2) )
# model.add( Dense(84, kernel_regularizer=regularizers.l2(0.01) ) )
# model.add( LeakyReLU(alpha=0.2) )
# model.add( Dense(1) )

# 2nd model
model.add( Convolution2D(16,5,5,activation=None) )
model.add( LeakyReLU(alpha=0.2) )
model.add( MaxPooling2D())
model.add( Dropout(rate=0.5) )
model.add( Convolution2D(24,5,5,activation=None) )
model.add( LeakyReLU(alpha=0.2) )
model.add( MaxPooling2D())
model.add( Dropout(rate=0.5) )
model.add( Convolution2D(30,7,7,activation=None) )
model.add( LeakyReLU(alpha=0.2) )
model.add( MaxPooling2D())
model.add( Dropout(rate=0.5) )
model.add( Flatten() )
model.add( Dense(30, kernel_regularizer=regularizers.l2(0.01) ) )
model.add( LeakyReLU(alpha=0.2) )
model.add( Dense(11, activation='tanh', kernel_regularizer=regularizers.l2(0.01) ) )
model.add( Dense(6, activation='tanh', kernel_regularizer=regularizers.l2(0.01) ) )
model.add( Dense(2, kernel_regularizer=regularizers.l2(0.01) ) )
model.add( LeakyReLU(alpha=0.2) )
model.add( Dense(1) )

model.compile( loss='mse', optimizer='adam' )
# train_steps_epoch = np.ceil( aug_multiple*len(train_samples)/float(batch_size))
valid_steps_epoch = np.ceil( 1*len(validation_samples)/float(batch_size))

train_steps_epoch = 100 # Arbitrary number, since we use infinite-looped generator
# valid_steps_epoch = np.floor( 1*len(validation_samples)/float(batch_size)) # Remove the last step, since we are using infinite-looped generator
history_object = model.fit_generator( \
                    train_gen, \
                    steps_per_epoch=train_steps_epoch, \
                    validation_data=valid_gen, \
                    validation_steps=valid_steps_epoch, \
                    epochs=num_epoch, verbose=1)

# Save the model
#--------------------------------------#
model.save('model.h5')


# print the keys contained in the history object
#--------------------------------------#
print(history_object.history.keys())

# plot the training and validation loss for each epoch
#--------------------------------------#
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.yscale('log')
plt.show()
