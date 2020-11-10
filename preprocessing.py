import gc
import os

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils import reverse_idx_category_dict, map_to_onehot

ucf_path = 'dataset/UCF-101'


def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]


def load_data_sample(name):
    video_path = os.path.join(ucf_path, name.split('_')[1], name)
    video = load_video(video_path)
    middle_frames = get_main_middle_frames(video)
    sample = np.array(middle_frames)  # / 255.0
    # sample = sample.astype(np.float16)
    return sample


def load_video(path, max_frames=0, resize=(224, 224)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return frames  # np.array(frames) / 255.0


def get_main_middle_frames(frames, offset=12, step=2):
    middle = len(frames) // 2
    middle_frames = frames[middle - offset:middle + offset:step]
    return middle_frames


def create_dataset():
    dir_names = next(os.walk(ucf_path))[1]
    x = []
    y = []
    for dir_ in dir_names:
        dir_path = os.path.join(ucf_path, dir_)
        file_names = next(os.walk(dir_path))[2]
        for file_name in file_names:
            x.append(file_name)
            y.append(reverse_idx_category_dict[dir_])
    return x, y


def split_dataset(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y, random_state=1)
    y_train = map_to_onehot(y_train)
    y_test = map_to_onehot(y_test)
    return x_train, x_test, y_train, y_test


def load_all_samples_for_dataset(x):
    return [load_data_sample(name) for name in tqdm(x)]


def test():
    pass


x, y = create_dataset()
x_train, x_test, y_train, y_test = split_dataset(x, y)

print(len(x_test))

x_train_samples = load_all_samples_for_dataset(x_train)

x_train_stacked = np.stack(x_train_samples, axis=0)
del x_train_samples
gc.collect()
np.save('dataset/x_train2.npy', x_train_stacked)

print('x_train shape:')
print(x_train_stacked.shape)
print(x_train_stacked[0].shape)

del x_train_stacked
gc.collect()

x_test_samples = load_all_samples_for_dataset(x_test)

x_test_stacked = np.stack(x_test_samples, axis=0)
del x_test_samples
gc.collect()
np.save('dataset/x_test2.npy', x_test_stacked)

print('x_test shape:')
print(x_test_stacked.shape)
print(x_test_stacked[0].shape)

del x_test_stacked
gc.collect()

y_train_stacked = np.stack(y_train, axis=0)
y_test_stacked = np.stack(y_test, axis=0)

print('y_train shape:')
print(y_train_stacked.shape)
print(y_train_stacked[0].shape)

print('y_test shape:')
print(y_test_stacked.shape)
print(y_test_stacked[0].shape)

np.save('dataset/y_train2.npy', y_train_stacked)
np.save('dataset/y_test2.npy', y_test_stacked)

# print(np.stack(y_test, axis=0).shape)
# print(np.stack(y_test, axis=0)[0].shape)
# print(np.stack(x_test_samples, axis=0).shape)
# print(np.stack(x_test_samples, axis=0)[0].shape)
#
# a = list(range(50))
# b = list(range(51))
#
# print(get_main_middle_frames(a))
# print(get_main_middle_frames(b))
# print(len(a))
# print(len(b))
# # format: name, frames, label
#

# print(load_data_sample('v_Biking_g02_c07.avi').dtype)
# print(load_data_sample('v_Biking_g02_c07.avi').shape)
