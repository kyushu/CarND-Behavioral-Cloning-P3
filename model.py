import csv
import numpy as np
import cv2
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.callbacks import ModelCheckpoint
# from keras import optimizers
# from keras.layers.core import Activation
from keras.layers.convolutional import Convolution2D#, MaxPooling2D
import matplotlib.pyplot as plt
# import tensorflow as tf


from utils import  generate_new_log, mod_csv



def generator(samples, batch_size=32):
    """Generate batch sample
    samples: rows of csv file
    batch_size: batch size of data to feeding in neural network
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:

                # correction = 0.1
                image = cv2.imread(batch_sample[0])
                angle = float(batch_sample[3])
                images.append(image)
                angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)

            yield sklearn.utils.shuffle(X_train, y_train)


def getYChannel(image):
    mod_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)[:,:,0]
    mod_image = np.reshape(mod_image, (image.shape[0], image.shape[1], 1))
    return mod_image


'''
1. Read droving log csv file
    the cvs format is 
    Center_Image, Left_Image, Right_Image, Steering, Throttle, Brake, Speed
'''


# update data path first
main_data_dir = 'data/'
target_csv_file = 'driving_log.csv'
final_csv_file = 'driving_log_all.csv'
# mode= 1 : only center image  
#       2 : take all center, left and right image
#       3 : randomly choose center, left or right
mod_csv(main_data_dir, mode=3)
generate_new_log(main_data_dir, target_csv_file, final_csv_file)


lines = []
with open("data/driving_log_all.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        # line is a list
        lines.append(line)

sklearn.utils.shuffle(lines)
train_samples, validation_samples = train_test_split(lines, test_size=0.2) 

batch_size = 128
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


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
model.add( Cropping2D( cropping=( (50, 20), (0, 0)) , input_shape=(160, 320, 3) ) )
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320 , 3)))

model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
# model.add(Dropout(0.5))
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='elu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='elu'))
model.add(Dropout(0.2))
model.add(Dense(1))

# Add checkpoint callback 
filepath = "weights.hdf5" 

# if use file path format like "weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
# weight will be store at every epoch
# filepath = "weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
callbacks_list = [checkpoint]

# model.compile(loss='mse', optimizer='adam')
# optimizer = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
# model.summary()
model.compile(loss='mse', optimizer='adam')

# history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

history_object = model.fit_generator(train_generator, 
    samples_per_epoch= len(train_samples), 
    validation_data=validation_generator, 
    nb_val_samples=len(validation_samples), 
    verbose=1, 
    callbacks=callbacks_list,
    nb_epoch=15)
model.save('model.h5')


### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


# for eliminating wranings
# del tf.get_default_session
import gc; gc.collect() 