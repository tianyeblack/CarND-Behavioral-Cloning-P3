import csv
import cv2
import numpy as np
from helper import *

CORRECTION = 0.2


def preprocess_driving_log(lines, augment_with_all_cameras=False):
    paths = [convert_path(l[0]) for l in lines]
    angles = [float(l[3]) for l in lines]
    if augment_with_all_cameras:
        paths += [convert_path(l[1]) for l in lines]
        paths += [convert_path(l[2]) for l in lines]
        angles += [float(l[3]) + CORRECTION for l in lines]
        angles += [float(l[3]) - CORRECTION for l in lines]
    paths_and_angles = list(zip(paths, angles))

    print(paths_and_angles[-1])
    with open(get_path(TRAINING_DATA_PATH, 'preprocessed_driving_log.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(paths_and_angles)


def augment_by_flipping(lines):
    paths = [l[0] for l in lines]
    vertical_flip_image_paths = [get_path(get_directory_name(), 'vf_' + get_file_name(p, False)) for p in paths]

    angles = [float(l[1]) for l in lines]
    vertical_flip_angles = [-a for a in angles]

    paths_and_angles = list(zip(paths + vertical_flip_image_paths, angles + vertical_flip_angles))

    print(paths_and_angles[-1])
    with open(get_path(TRAINING_DATA_PATH, 'augmented_driving_log.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(paths_and_angles)
        for l in list(zip(paths, vertical_flip_image_paths)):
            image = cv2.imread(l[0])
            vertical_flip_image = np.flip(image, 1)
            cv2.imwrite(l[1], vertical_flip_image)


preprocess_driving_log(read_logs('driving_log.csv'), True)
augment_by_flipping(read_logs('preprocessed_driving_log.csv'))
