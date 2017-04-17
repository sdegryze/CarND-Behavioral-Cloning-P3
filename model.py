import numpy as np
import csv
import cv2
import os
import random
import pickle

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Dropout, Cropping2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Check the output of this command to verify the connection to the GPU is working correctly
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# Model parameters - change these as needed
steering_correction = 0.15
batch_size = 128
load_previous_weights = False
weights_filename = "model.h5"


lines = []
steering_angles = []
nr_samples = 0

with open("./data/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # ignore first line which contains the headers
    for line in reader:
        nr_samples += 1
        steering_angle = float(line[3])
        # Since the training dataset contains a disproportionate amount of datapoints with steering angle 0,
        # only include a smaller fraction (i.e., 5%) of the points with steering angle 0
        if abs(steering_angle) > 0.0 or random.random() > 0.95:
            steering_angles.append(steering_angle)
            lines.append(line)

train_lines, validation_lines = train_test_split(lines, test_size=0.2)

print("Total number of samples in csvfile: {0}; number of samples retained: {1}".format(nr_samples, len(lines)))
print("{0} samples in training dataset and {1} in validation dataset".format(len(train_lines), len(validation_lines)))


def get_path(line_entry):
    return os.path.join("./data/IMG/", os.path.basename(line_entry.strip()))


def get_augmented_row(line, flipped, angle_idx):
    source_path = get_path(line[angle_idx])
    image = cv2.imread(source_path)
    steering_angle = float(line[3])

    # Add steering angle adjustment to left/right cameras
    if angle_idx == 0:  # center camera
        steering_angle_corrected = steering_angle
    elif angle_idx == 1:  # left camera
        steering_angle_corrected = steering_angle + steering_correction
    elif angle_idx == 2:  # right camera
        steering_angle_corrected = steering_angle - steering_correction

    if flipped:
        steering_angle_corrected *= -1
        image = np.fliplr(image)

    return image.astype(np.float32), steering_angle_corrected


def get_data_generator(csv_data, batch_size=128, samples_per_epoch=None):
    if samples_per_epoch is None:
        samples_per_epoch = len(csv_data)

    batches_per_epoch = samples_per_epoch // batch_size
    batch_nr = 0

    while True:
        start_idx = batch_nr * batch_size
        end_idx = start_idx + batch_size - 1

        # initialize batch data
        X_batch = np.zeros((batch_size, 160, 320, 3), dtype=np.float32)
        y_batch = np.zeros((batch_size,), dtype=np.float32)

        # slice a `batch_size` sized chunk from the csv_data
        # and generate augmented data for each row in the chunk on the fly
        for idx, csv_line in enumerate(csv_data[start_idx:(end_idx+1)]):
            # perform image augmentation for each row with a random draw for camera and mirroring
            flipped = random.randint(0, 1) == 0
            angle_idx = random.randint(0, 2)

            X, y = get_augmented_row(csv_line, flipped, angle_idx)
            X_batch[idx], y_batch[idx] = X, y

        batch_nr += 1
        if batch_nr == batches_per_epoch - 1:
            # reset the index so that we can cycle over the csv_data again
            batch_nr = 0
        yield X_batch, y_batch

train_samples_per_epoch = (len(train_lines) // batch_size) * batch_size

# Go over validation data twice to have a more robust value of the validation loss
val_samples_per_epoch = (len(validation_lines) // batch_size) * batch_size * 2

train_generator = get_data_generator(train_lines,
                                     batch_size=batch_size,
                                     samples_per_epoch=train_samples_per_epoch)
validation_generator = get_data_generator(validation_lines,
                                          batch_size=batch_size,
                                          samples_per_epoch=val_samples_per_epoch)

model = Sequential()
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="elu", init='he_normal'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="elu", init='he_normal'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="elu", init='he_normal'))
model.add(Convolution2D(64, 3, 3, activation="elu", init='he_normal'))
model.add(Convolution2D(64, 3, 3, activation="elu", init='he_normal'))
model.add(Flatten())
model.add(Dropout(p=0.5))
model.add(Dense(100, activation='elu', init='he_normal'))
model.add(Dropout(p=0.5))
model.add(Dense(50, activation='elu', init='he_normal'))
model.add(Dense(10, activation='elu', init='he_normal'))
model.add(Dense(1))

if load_previous_weights:
    model.load_weights(weights_filename)
model.compile(loss='mse', optimizer=Adam(1e-4), metrics=['accuracy'])
checkpoint = ModelCheckpoint(weights_filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

history_object = model.fit_generator(train_generator,
                                     samples_per_epoch=train_samples_per_epoch,
                                     validation_data=validation_generator,
                                     nb_val_samples=val_samples_per_epoch,
                                     nb_epoch=200,
                                     callbacks=[checkpoint],
                                     verbose=1)

model.save('final_model.h5')

with open('history_object', 'wb') as handle:
    pickle.dump(history_object.history, handle, protocol=pickle.HIGHEST_PROTOCOL)