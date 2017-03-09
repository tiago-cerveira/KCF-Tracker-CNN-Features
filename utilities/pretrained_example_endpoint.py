import numpy as np
import os
import tensorflow as tf

from datasets import imagenet
from tensorflow.contrib.slim.nets import inception
import inception_preprocessing
from matplotlib import pyplot as plt

slim = tf.contrib.slim

#batch_size = 3
image_size = inception.inception_v3.default_image_size
names = imagenet.create_readable_names_for_imagenet_labels()

with tf.Graph().as_default():
    filename_queue = tf.train.string_input_producer(
        ['/home/tiago/Desktop/img.jpeg', '/home/tiago/Desktop/img2.jpeg', '/home/tiago/Desktop/img3.jpeg'], shuffle=False)

    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    image = tf.image.decode_jpeg(value, channels=3)
    processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
    processed_images = tf.expand_dims(processed_image, 0)

    # Create the model, use the default arg scope to configure the batch norm parameters.
    with slim.arg_scope(inception.inception_v3_arg_scope()):
        tensor_out, endpoints = inception.inception_v3_base(processed_images, final_endpoint='Conv2d_2b_3x3')
        #logits, _ = inception.inception_v3(processed_images, num_classes=1001, is_training=False)
    #probabilities = tf.nn.softmax(logits)

    init_fn = slim.assign_from_checkpoint_fn('/home/tiago/PycharmProjects/DeepTrack/inception_v3.ckpt', slim.get_model_variables('InceptionV3'))

    with tf.Session() as sess:
        init_fn(sess)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(3):
            smt = tensor_out.eval()
            print(smt.shape)
            for j in range(3):

                plt.imshow(smt[0,:,:,10+j])
                plt._show()
            #print(np_image.shape, something[0].shape)

            # print stuff
            #for i in range(5):
            #    plt.imshow(something[0][0,:,:,2*i], cmap=plt.cm.gray)
            #    plt.figure(i+1)

            #plt.show()
            #print(np_image.shape)
            #plt.imshow(np_image)
            #plt.show()

        coord.request_stop()
        coord.join(threads)