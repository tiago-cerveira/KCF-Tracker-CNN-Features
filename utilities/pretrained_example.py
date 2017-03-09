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

img_list = ['/home/tiago/Desktop/img.jpeg', '/home/tiago/Desktop/img2.jpeg', '/home/tiago/Desktop/img3.jpeg', '/home/tiago/Desktop/img4.jpeg']

with tf.Graph().as_default():
    filename_queue = tf.train.string_input_producer(img_list, shuffle=False)

    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    image = tf.image.decode_jpeg(value, channels=3)
    processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
    processed_images = tf.expand_dims(processed_image, 0)

    # Create the model, use the default arg scope to configure the batch norm parameters.
    with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits, _ = inception.inception_v3(processed_images, num_classes=1001, is_training=False)
    probabilities = tf.nn.softmax(logits)

    init_fn = slim.assign_from_checkpoint_fn('/home/tiago/PycharmProjects/DeepTrack/inception_v3.ckpt', slim.get_model_variables('InceptionV3'))

    with tf.Session() as sess:
        init_fn(sess)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(4):
            prb = probabilities.eval()

            argmax = np.argmax(prb)
            print('Probability %0.2f%% => [%s]' % (prb[0][argmax], names[argmax]))
            #plt.imshow(img)
            #plt.show()
            #np_image, probabilities = sess.run([image, probabilities])
            #probabilities = probabilities[0, 0:]
            #classification = np.argmax(probabilities)
            #print('Probability %0.2f%% => [%s]' % (probabilities[classification], names[classification]))
            #print(np_image.shape)
            #plt.imshow(np_image)
            #plt.show()

        coord.request_stop()
        coord.join(threads)
