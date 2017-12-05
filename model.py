import math
import cv2
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Conv2D, Cropping2D
from sklearn.model_selection import train_test_split
from random import shuffle

TRAINING_DATA_PATH = 'E://mydata/'
CORRECTION = 0.2
BATCH_SIZE = 32
TRAIN_TEST_SPLIT_RATIO = 0.2


def convert_path(source):
    filename = source.split('\\')[-1]
    return '%sIMG/' % TRAINING_DATA_PATH + filename


def greyscale(X):
    return X[:, :, :, :1] / 3 + X[:, :, :, 1:2] / 3 + X[:, :, :, -1:] / 3


def reversing(X):
    return X[:, :, :, ::-1]


def greyscale_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 4
    shape[-1] = 1
    return tuple(shape)


def generator(samples, batch_size=BATCH_SIZE):
    num_samples = len(samples)
    while True:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            center_images = [cv2.imread(convert_path(s[0])) for s in batch_samples]
            center_angles = [float(s[3]) for s in batch_samples]
            left_images = [cv2.imread(convert_path(s[1])) for s in batch_samples]
            left_angles = [float(s[3]) + CORRECTION for s in batch_samples]
            right_images = [cv2.imread(convert_path(s[2])) for s in batch_samples]
            right_angles = [float(s[3]) - CORRECTION for s in batch_samples]

            sample_images = center_images + left_images + right_images
            sample_angles = center_angles + left_angles + right_angles
            flip_sample_images = [np.flip(img, 1) for img in sample_images]
            flip_sample_angles = [-m for m in sample_angles]

            # trim image to only see section with road
            X_sample_train = np.array(sample_images + flip_sample_images)
            y_sample_train = np.array(sample_angles + flip_sample_angles)
            yield sklearn.utils.shuffle(X_sample_train, y_sample_train)


with open('%sdriving_log.csv' % TRAINING_DATA_PATH) as csvfile:
    lines = [l.strip('\n').split(',') for l in csvfile.readlines()]

print(convert_path(lines[0][0]))
train_samples, validation_samples = train_test_split(lines, test_size=TRAIN_TEST_SPLIT_RATIO)
train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

# images = [cv2.imread(convert_path(line[0])) for line in lines]
# images += [cv2.imread(convert_path(line[1])) for line in lines]
# images += [cv2.imread(convert_path(line[2])) for line in lines]
# vf_images = [np.flip(img, 1) for img in images]
# images += vf_images
# measurements = [float(line[3]) for line in lines]
# measurements += [float(line[3]) + CORRECTION for line in lines]
# measurements += [float(line[3]) - CORRECTION for line in lines]
# measurements += [-m for m in measurements]
#
# X_train = np.array(images)
# y_train = np.array(measurements)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Lambda(reversing))
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='relu'))
model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='relu'))
model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(Flatten())
# model.add(Dropout(0.5))
model.add(Dense(100))
# model.add(Dropout(0.5))
model.add(Dense(50))
# model.add(Dropout(0.5))
model.add(Dense(10))
# model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_data = model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples) / BATCH_SIZE),
                                   epochs=6,
                                   validation_data=validation_generator,
                                   validation_steps=math.ceil(len(validation_samples) / BATCH_SIZE))
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5, verbose=1)

model.save('transfer_learning_model.h5')

print(history_data.history.keys())
plt.plot(history_data.history['loss'])
plt.plot(history_data.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
