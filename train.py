import csv
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.core import Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
import matplotlib.pyplot as plt

'''
1. Read droving log csv file
    the cvs format is 
    Center_Image, Left_Image, Right_Image, Steering, Throttle, Brake, Speed
'''
lines = []
with open("data/driving_log_all.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        # line is a list
        lines.append(line)

'''
2. load image from file
'''
car_images = []
steer_angles = []

def process_image(img_path, current_directory):

    filename = img_path.split('/')[-1]
    current_path = current_directory + filename
    # print(current_path)
    return cv2.imread(current_path)

for line in lines:
    
    correction = 0.1
    # Steering
    steer_center = float(line[3])
    steer_left = steer_center + correction
    steer_right = steer_center - correction

    # filename = source_path.split('/')[-1]
    # current_path = "/data/IMG/" + filename
    # img_center = cv2.imread(line[0])
    # img_left = cv2.imread(line[1])
    # img_right = cv2.imread(line[2])
    # current_directory = "data/IMG/"
    # img_center = process_image(line[0], current_directory)
    # img_left = process_image(line[1], current_directory)
    # img_right = process_image(line[2], current_directory)

    img_center = cv2.imread(line[0])
    img_left = cv2.imread(line[1])
    img_right = cv2.imread(line[2])


    car_images.extend((img_center, img_left, img_right))
    steer_angles.extend((steer_center, steer_left, steer_right))


'''
3. data augmentation
'''
augmented_images = []
augmented_measurements = []

for image, measurement in zip(car_images, steer_angles):
    augmented_images.append(image)
    augmented_measurements.append(measurement)

    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(-measurement)



'''
4. Prepare Data
'''
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)


'''
5. Build CNN
'''
'''
keras.Lambda can be used to create arbitrary functions that operate on each image as it passes through the layer
keras.layers.core.Lambda(function, output_shape=None, mask=None, arguments=None)
outpu_shape can be inferred when using TensorFlow backend

keras.layers.convolutional.Convolution2D(nb_filter, nb_row, nb_col, init='glorot_uniform', activation='linear', 
                                            weights=None, border_mode='valid', subsample=(1, 1), dim_ordering='default', 
                                            W_regularizer=None, b_regularizer=None, activity_regularizer=None, 
                                            W_constraint=None, b_constraint=None, bias=True)
'''

model = Sequential()

model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320 , 3)))
model.add( Cropping2D( cropping=((70, 25), (0, 0)) ) )
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)
model.save('model.h5')
