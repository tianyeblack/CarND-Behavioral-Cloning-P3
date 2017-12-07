import math
import cv2
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Conv2D, Cropping2D
from sklearn.model_selection import train_test_split
from random import shuffle
from helper import *

BATCH_SIZE = 32
TRAIN_TEST_SPLIT_RATIO = 0.2


def generator(samples, batch_size=BATCH_SIZE):
    num_samples = len(samples)
    while True:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            sample_images = [cv2.merge(list(reversed(cv2.split(cv2.imread(s[0]))))) for s in batch_samples]
            sample_angles = [float(s[1]) for s in batch_samples]

            # trim image to only see section with road
            X_sample_train = np.array(sample_images)
            y_sample_train = np.array(sample_angles)
            yield sklearn.utils.shuffle(X_sample_train, y_sample_train)


lines = read_logs('augmented_driving_log.csv')
print(lines[0][0])

train_samples, validation_samples = train_test_split(lines, test_size=TRAIN_TEST_SPLIT_RATIO)
train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='relu'))
model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='relu'))
model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100))
# model.add(Dropout(0.5))
model.add(Dense(50))
# model.add(Dropout(0.5))
model.add(Dense(10))
# model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_data = model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples) / BATCH_SIZE),
                                   epochs=5,
                                   validation_data=validation_generator,
                                   validation_steps=math.ceil(len(validation_samples) / BATCH_SIZE))

model.save('transfer_learning_model.h5')

print(history_data.history.keys())
plt.plot(history_data.history['loss'])
plt.plot(history_data.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
