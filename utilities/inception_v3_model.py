import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
import inception_preprocessing
from util import *
import os
import numpy as np
# from matplotlib import pyplot as plt


slim = tf.contrib.slim
image_size = inception.inception_v3.default_image_size


def extract_features(sequence_list):
    with tf.Graph().as_default():

        filename_queue = tf.train.string_input_producer(sequence_list, shuffle=False)
        reader = tf.WholeFileReader()
        key, value = reader.read(filename_queue)
        image = tf.image.decode_jpeg(value, channels=3)

        processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
        processed_images = tf.expand_dims(processed_image, 0)

        # Create the model, use the default arg scope to configure the batch norm parameters.
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            tensor_out, endpoints = inception.inception_v3_base(processed_images, final_endpoint='Conv2d_2a_3x3')

        init_fn = slim.assign_from_checkpoint_fn(os.path.abspath('checkpoint/inception_v3.ckpt'),
                                                 slim.get_model_variables('InceptionV3'))

        with tf.Session() as sess:
            init_fn(sess)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)


            img_features = tensor_out.eval().squeeze()

            # print(img_features[0].shape)
            # for j in range(3):
            # plt.imshow(img_features[0][0, :, :, 10 + j])
            # plt._show()

            coord.request_stop()
            coord.join(threads)
    return img_features


def extract_features2(sequence_list):
    print("extracting features...")
    with tf.Graph().as_default():
        filename_queue = tf.train.string_input_producer(sequence_list, shuffle=False)
        reader = tf.WholeFileReader()
        key, value = reader.read(filename_queue)
        image = tf.image.decode_jpeg(value, channels=3)
        processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
        processed_images = tf.expand_dims(processed_image, 0)

        start = start_timer()

        # Create the model, use the default arg scope to configure the batch norm parameters.
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            tensor_out, endpoints = inception.inception_v3_base(processed_images, final_endpoint='Conv2d_2a_3x3')

        init_fn = slim.assign_from_checkpoint_fn(os.path.abspath('checkpoint/inception_v3.ckpt'),
                                                 slim.get_model_variables('InceptionV3'))

        end_timer(start, "create model and restore from checkpoint")

        with tf.Session() as sess:
            init_fn(sess)

            start = start_timer()

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            img_features = []
            for i in range(len(sequence_list)):
                img_features.append(tensor_out.eval().squeeze())
                # print(img_features[0].shape)
                # for j in range(3):
                # plt.imshow(img_features[0][0, :, :, 10 + j])
                # plt._show()

            coord.request_stop()
            coord.join(threads)

            duration = end_timer(start, "preprocess and extract features of all frames")
            print("total frames:", len(sequence_list))
            print("average FPS for extracting features:", round(len(sequence_list) / duration, 2), "[FPS]")
    return img_features
