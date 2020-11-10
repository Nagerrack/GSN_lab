import itertools
import os

# import embed as embed
import imageio
import numpy as np
import cv2

dir_names = next(os.walk('dataset/UCF-101'))[1]
idx_category_dict = dict(zip(itertools.count(), dir_names))
reverse_idx_category_dict = dict(zip(dir_names, itertools.count()))


def idx_to_onehot(idx, cat_number=101):
    encoding = np.zeros(cat_number, dtype=float)
    encoding[idx] = 1.
    return encoding


def map_to_onehot(idxs, cat_number=101):
    mapped = []
    for idx in idxs:
        encoding = np.zeros(cat_number, dtype=float)
        encoding[idx] = 1.
        mapped.append(encoding)
    return mapped


def get_top_1(output):
    return np.argmax(output)


def get_top_n(output, n=5):
    return np.argsort(-output)[:n]


def map_to_categories(indices, map_dict=idx_category_dict):
    return list(map(map_dict.get, indices))


def map_to_indices(categories, map_dict=reverse_idx_category_dict):
    return list(map(map_dict.get, categories))


# def to_gif(images):
#     converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
#     imageio.mimsave('images/animation.gif', converted_images, fps=25)
#     return embed.embed_file('images/animation.gif')


a = np.array([1, 2, 31, 40, 5, 6, 7, 8, 9, 10])

print(get_top_1(a))
