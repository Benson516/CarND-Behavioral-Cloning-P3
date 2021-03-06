import os
import numpy as np
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

# The image to check
# sample = "center_2016_12_01_13_42_18_344.jpg"
# sample = "right_2016_12_01_13_42_18_344.jpg"
sample = "left_2016_12_01_13_42_18_344.jpg"

current_path_center = data_path + '/IMG/' + sample.split('/')[-1]

image_ = imageio.imread(current_path_center)

image_crop = image_[50:-20, :]

image_flip = np.fliplr(image_crop)

# plot
#--------------------------#
plt.figure()
plt.imshow(image_)
# crop
plt.figure()
plt.imshow(image_crop)
# crop, flip
plt.figure()
plt.imshow(image_flip)



# Show
plt.show()
