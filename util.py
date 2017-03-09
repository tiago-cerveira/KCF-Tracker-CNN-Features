from __future__ import print_function
import argparse
import os
import time
import numpy as np
# import h5py
# from matplotlib import pyplot as plt
from Tkinter import Tk
from tkFileDialog import askdirectory
import os
import tensorflow as tf


def get_arguments():
    parser = argparse.ArgumentParser(description="data sequence choice")
    parser.add_argument("data_sequence")
    return parser.parse_args()


def get_list_seq_OTB100(directory):
    print("loading sequence:", directory)

    # gets a lis with all the image files in the correct order
    files = sorted(os.listdir("data_seq/" + directory + "/img/"))

    # adds the full path to all image files
    for i, file in enumerate(files):
        files[i] = os.path.abspath("data_seq/" + directory + "/img/" + file)
    return files

"""
The following functions handle timers
"""


def start_timer():
    return time.time()


def end_timer(start, message):
    duration = round(time.time() - start, 2)
    print("took", duration, "seconds to", message)
    return duration
""""""

"""
The following functions were used to write the data sequences, after feature extraction, to disk (deprecated)
"""


def write_dat(np_array_list, filename):
    file_object = open('binary/' + filename + '_features.dat', "wb")

    start = start_timer()

    for i in range(len(np_array_list)):
        np_array_list[i].tofile(file_object)
    file_object.close()

    end_timer(start, "write binary file to disk")


def write_npz(np_array_list, filename):
    start = start_timer()
    print(np_array_list[0].shape)

    for i in range(len(np_array_list)):
        # np_array_list[i] = np.delete(np_array_list[i], np.s_[3:], axis=2)
        np_array_list[i] *= 50
        np_array_list[i] = np_array_list[i].astype(np.uint8)

    print("shape of each frame features:", np_array_list[0].shape)

    np.savez('binary/' + filename + '_features.npz', *np_array_list)

    end_timer(start, "write binary file to disk")


def write_h5(np_array_list, filename):
    start = start_timer()

    with h5py.File('binary/' + filename + '_features.h5', 'w') as hf:
        for i, img in enumerate(np_array_list):
            hf.create_dataset('{:05}'.format(i), data=img)

    end_timer(start, "write binary file to disk")
""""""


def select_sequence():
    # Define the path to the image sequences folder
    base_path = '~/sequences'
    # Options for the UI sequence selection
    options = {'initialdir': base_path,
               'mustexist': False,
               'title': 'Enter the desired sequence folder and click OK button.'
               }
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    video_path = askdirectory(**options)  # show an "Open" dialog box and return the path to the selected file
    return video_path


def load_video_info(video_path):
    # if a valid video path was not provided
    if not isinstance(video_path, str):
        exit(-1)
    txt_file = 'groundtruth_rect.txt'
    groundtruth_file = video_path + '/' + txt_file
    groundtruth = np.loadtxt(groundtruth_file, delimiter=",")
    init_pos = groundtruth[0, :-2]
    init_pos = init_pos[::-1]
    target_sz = groundtruth[0, 2:]
    target_sz = target_sz[::-1]
    video_path += '/img/'
    img_files = [f for f in os.listdir(video_path) if os.path.isfile(os.path.join(video_path, f))]
    img_files.sort()
    return img_files, init_pos, target_sz, groundtruth, video_path


def preprocess_image(image, height, width,
                        central_fraction=1, scope=None):
  """Prepare one image for evaluation.

  If height and width are specified it would output an image with that size by
  applying resize_bilinear.

  If central_fraction is specified it would cropt the central fraction of the
  input image.

  Args:
    image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
      [0, 1], otherwise it would converted to tf.float32 assuming that the range
      is [0, MAX], where MAX is largest positive representable number for
      int(8/16/32) data type (see `tf.image.convert_image_dtype` for details)
    height: integer
    width: integer
    central_fraction: Optional Float, fraction of the image to crop.
    scope: Optional scope for name_scope.
  Returns:
    3-D float Tensor of prepared image.
  """
  with tf.name_scope(scope, 'eval_image', [image, height, width]):
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # Crop the central region of the image with an area containing 100% of
    # the original image.
    if central_fraction:
      image = tf.image.central_crop(image, central_fraction=central_fraction)

    if height and width:
      # Resize the image to the specified height and width.
      # image = tf.expand_dims(image, 0)
      # resize with bilinear interpolation
      image = tf.image.resize_images(image, [height, width])
      # image = tf.squeeze(image, [0])
    # image = tf.sub(image, 0.5)
    image -= 0.5
    # image = tf.mul(image, 2.0)
    image *= 2.0
    return image

