from __future__ import print_function
import cv2
import desiredResponse
import kernel
import numpy as np
# import fhog
import os
# os.environ['GLOG_minloglevel'] = '3'
# import caffe
import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
from PIL import Image
from util import *


slim = tf.contrib.slim
image_size = inception.inception_v3.default_image_size


# TODO : Get scale samples as gray, HoG and CNN features
def get_scale_sample(im, pos, base_target_sz, scale_factors, scale_window, scale_model_sz):

        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

        number_of_scales = len(scale_factors)
        n = scale_model_sz[1] * scale_model_sz[0]
        out = np.zeros((n.astype(int), number_of_scales))

        for s in xrange(0, number_of_scales):
            patch_sz = np.floor(base_target_sz * scale_factors[s])

            xs = np.floor(pos[1]) + np.arange(1, patch_sz[1]+1) - np.floor(patch_sz[1]/2)
            ys = np.floor(pos[0]) + np.arange(1, patch_sz[0]+1) - np.floor(patch_sz[0]/2)

            xs[xs < 0] = 0
            ys[ys < 0] = 0
            xs[xs >= im.shape[1]] = im.shape[1] - 1
            ys[ys >= im.shape[0]] = im.shape[0] - 1

            im_patch = im[np.ix_(ys.astype(int), xs.astype(int))]

            im_patch_resized = cv2.resize(im_patch, (scale_model_sz[1].astype(int), scale_model_sz[0].astype(int)))

            out[:, s] = im_patch_resized.flatten(1) * scale_window[s]

        return out


# def fhog_features(image, cell_size, cos_window):
#
#         c = fhog.pyOCV()
#         c.getMat(np.single(image), cell_size)
#
#         features = np.zeros((np.floor(cos_window.shape[1]).astype(int), np.floor(cos_window.shape[0]).astype(int), 31))
#
#         for i in xrange(0, 31):
#             features[:, :, i] = cv2.resize(c.returnMat(i), (np.floor(cos_window.shape[1]).astype(int),
#                                                             np.floor(cos_window.shape[0]).astype(int)))*cos_window
#
#         return features


# def gray_feat(im, cos_window):
#
#         im = np.divide(im, 255.0)
#         im = im - np.mean(im)
#         features = cos_window * im
#
#         return features


def image_segmentation(im, pos, patch_sz):
    xs = np.floor(pos[1]) + np.arange(1, patch_sz[1]+1) - np.floor(patch_sz[1]/2) - 1
    ys = np.floor(pos[0]) + np.arange(1, patch_sz[0]+1) - np.floor(patch_sz[0]/2) - 1

    xs[xs < 0] = 0
    ys[ys < 0] = 0
    xs[xs >= im.shape[1]] = im.shape[1] - 1
    ys[ys >= im.shape[0]] = im.shape[0] - 1

    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

    im_patch = im[np.ix_(ys.astype(int), xs.astype(int))]

    _, segmented_image = cv2.threshold(im_patch, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Remove the blobs originated from padding when the image is out-of-bounds, i.e. part of the ROI is outside the image frame.
    # segmented_image[ys == 0, :] = 0
    # segmented_image[:, xs == 0] = 0
    # segmented_image[ys == im.shape[0]-1, :] = 0
    # segmented_image[:, xs == im.shape[1]-1] = 0

    krnel = np.ones((3, 3), np.uint8)
    segmented_image = cv2.erode(segmented_image, krnel, iterations=1)

    return segmented_image


class Tracker:

    def __init__(self, im, params):

        self.parameters = params
        self.pos = self.parameters.init_pos
        self.target_sz = self.parameters.target_size

        self.graph = None

        # Initial target size
        self.init_target_sz = self.parameters.target_size
        # target sz at scale = 1
        self.base_target_sz = self.parameters.target_size

        # Window size, taking padding into account
        self.window_sz = np.floor(np.array((max(self.base_target_sz),
                                            max(self.base_target_sz))) * (1 + self.parameters.padding))

        sz = self.window_sz
        sz = np.floor(sz / self.parameters.cell_size)
        self.l1_patch_num = np.floor(self.window_sz / self.parameters.cell_size)

        # Desired translation filter output (2d gaussian shaped), bandwidth
        # Proportional to target size
        output_sigma = np.sqrt(np.prod(self.base_target_sz)) * self.parameters.output_sigma_factor / self.parameters.cell_size
        self.yf = np.fft.fft2(desiredResponse.gaussian_response_2d(output_sigma, self.l1_patch_num))

        # Desired output of scale filter (1d gaussian shaped)
        scale_sigma = self.parameters.number_of_scales / np.sqrt(self.parameters.number_of_scales) * self.parameters.scale_sigma_factor
        self.ysf = np.fft.fft(desiredResponse.gaussian_response_1d(scale_sigma, self.parameters.number_of_scales))

        # Cosine window with the size of the translation filter (2D)
        self.cos_window = np.dot(np.hanning(self.yf.shape[0]).reshape(self.yf.shape[0], 1),
                                 np.hanning(self.yf.shape[1]).reshape(1, self.yf.shape[1]))

        # Cosine window with the size of the scale filter (1D)
        if np.mod(self.parameters.number_of_scales, 2) == 0:
            self.scale_window = np.single(np.hanning(self.parameters.number_of_scales + 1))
            self.scale_window = self.scale_window[1:]
        else:
            self.scale_window = np.single(np.hanning(self.parameters.number_of_scales))

        # Scale Factors [...0.98 1 1.02 1.0404 ...] NOTE: it is not a incremental value (see the scaleFactors values)
        ss = np.arange(1, self.parameters.number_of_scales + 1)
        self.scale_factors = self.parameters.scale_step**(np.ceil(self.parameters.number_of_scales / 2.0) - ss)

        # If the target size is over the threshold then downsample
        if np.prod(self.init_target_sz) > self.parameters.scale_model_max_area:
            self.scale_model_factor = np.sqrt(self.parameters.scale_model_max_area/np.prod(self.init_target_sz))
        else:
            self.scale_model_factor = 1

        self.scale_model_sz = np.floor(self.init_target_sz*self.scale_model_factor)

        self.currentScaleFactor = 1

        self.min_scale_factor = self.parameters.scale_step**np.ceil(np.log(np.max(5.0 / sz)) / np.log(self.parameters.scale_step))
        self.max_scale_factor = self.parameters.scale_step**np.floor(np.log(np.min(im.shape[0:-1] / self.base_target_sz)) / np.log(self.parameters.scale_step))

        self.confidence = np.array(())
        self.high_freq_energy = np.array(())
        self.psr = np.array(())

        # Flag that indicates if the track lost the target or not
        self.lost = False

        self.model_alphaf = None
        self.model_xf = None
        self.sf_den = None
        self.sf_num = None

    def detect(self, im):
        # Extract the features to detect.
        xt = self.translation_sample(im, self.pos, self.window_sz, self.parameters.cell_size, self.cos_window,
                                     self.parameters.features, self.currentScaleFactor, False)
        # 2D Fourier transform. Spatial domain (x,y) to frequency domain.
        xtf = np.fft.fft2(xt, axes=(0, 1))

        # Compute the feature kernel
        if self.parameters.kernel.kernel_type == 'Gaussian':
            kzf = kernel.gaussian_correlation(xtf, self.model_xf, self.parameters.kernel.kernel_sigma, self.parameters.features)
        else:
            kzf = kernel.linear_correlation(xtf, self.model_xf, self.parameters.features)

        # Translation Response map. The estimated location is the argmax (response).
        translation_response = np.real(np.fft.ifft2(self.model_alphaf * kzf, axes=(0, 1)))

        if self.parameters.debug:
            row_shift, col_shift = np.floor(np.array(translation_response.shape)/2).astype(int)
            r = np.roll(translation_response, col_shift, axis=1)
            r = np.roll(r, row_shift, axis=0)
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            response_map = cv2.applyColorMap(r, cv2.COLORMAP_AUTUMN)
            cv2.imshow('image', response_map)
            cv2.waitKey(1)

        row, col = np.unravel_index(translation_response.argmax(), translation_response.shape)

        # Peak-to-sidelobe ratio. If this value drops below 6.0 assume the track lost the target.
        # response_aux = np.copy(translation_response)
        # response_aux[row-3:row+3, col-3:col+3] = 0
        # self.psr = np.append(self.psr, (np.max(translation_response) - np.mean(response_aux))/(np.std(response_aux)))
        # print self.psr
        self.psr = (np.max(translation_response) - np.mean(translation_response))/(np.std(translation_response))
        print("psr:", round(self.psr, 2))
        if self.psr < self.parameters.peak_to_sidelobe_ratio_threshold:
            self.lost = True
        else:
            self.lost = False

        if row > xtf.shape[0]/2:
            row = row - xtf.shape[0]
        if col > xtf.shape[1]/2:
            col = col - xtf.shape[1]

        # Compute the new estimated position in the image coordinates (0,0) <-> Top left corner
        self.pos = self.pos + self.parameters.cell_size * np.round(np.array((row, col)) * self.currentScaleFactor)
        # print np.array((col,row))

        # Return the position (center of mass in image coordinates) and the lost flag.
        return np.array([self.pos[0], self.pos[1], self.target_sz[0], self.target_sz[1]]), self.lost, xtf

    def train(self, im, start=True, xlf_from_detection=None):
        # Extract the features to train.
        if start:
            xl = self.translation_sample(im, self.pos, self.window_sz, self.parameters.cell_size, self.cos_window,
                                         self.parameters.features, self.currentScaleFactor, start)
            # 2D Fourier transform. Spatial domain (x,y) to frequency domain.
            xlf = np.fft.fft2(xl, axes=(0, 1))
        else:
            xlf = xlf_from_detection

        # Compute the features kernel
        # if self.parameters.kernel.kernel_type == 'Gaussian':
        #     kf = kernel.gaussian_correlation(xlf, xlf, self.parameters.kernel.kernel_sigma, self.parameters.features)
        # else:
        kf = kernel.linear_correlation(xlf, xlf, self.parameters.features)

        # Compute the optimal translation filter
        alphaf = self.yf / (kf + self.parameters.lmbda)

        # alpha= np.real(np.fft.ifft2(alphaf, axes=(0,1)))
        # row_shift, col_shift = np.floor(np.array(alphaf.shape)/2).astype(int)
        # alpha = np.roll(alpha, col_shift,axis=1)
        # alpha = np.roll(alpha,row_shift, axis=0)
        # cv2.imshow("alpha", alpha)
        # cv2.waitKey(0)

        # Extract the scale samples
        xs = get_scale_sample(im, self.pos, self.base_target_sz, self.currentScaleFactor * self.scale_factors, self.scale_window, self.scale_model_sz)

        # Compute the fourier transform from scale space to frequency
        xsf = np.fft.fft(xs, self.parameters.number_of_scales, axis=1)

        # Compute the optimal scale filter
        new_sf_num = self.ysf * np.conjugate(xsf)
        new_sf_den = np.sum(xsf * np.conjugate(xsf), axis=0)

        # If first frame create the model
        if start:
            self.model_alphaf = alphaf
            self.model_xf = xlf
            self.sf_den = new_sf_den
            self.sf_num = new_sf_num
        # Update the model in the consecutive frames.
        else:
            self.sf_den = (1 - self.parameters.learning_rate) * self.sf_den + self.parameters.learning_rate * new_sf_den
            self.sf_num = (1 - self.parameters.learning_rate) * self.sf_num + self.parameters.learning_rate * new_sf_num
            self.model_alphaf = (1 - self.parameters.learning_rate) * self.model_alphaf + self.parameters.learning_rate * alphaf
            self.model_xf = (1 - self.parameters.learning_rate) * self.model_xf + self.parameters.learning_rate * xlf

    def translation_sample(self, im, pos, model_sz, cell_size, cos_window, features, current_scale_factor, first):
        patch_sz = np.floor(model_sz * current_scale_factor)

        if patch_sz[0] < 1:
            patch_sz[0] = 2
        if patch_sz[1] < 1:
            patch_sz[1] = 2

        xs = np.floor(pos[1]) + np.arange(1, patch_sz[1]+1) - np.floor(patch_sz[1]/2) - 1
        ys = np.floor(pos[0]) + np.arange(1, patch_sz[0]+1) - np.floor(patch_sz[0]/2) - 1

        xs[xs < 0] = 0
        ys[ys < 0] = 0
        xs[xs >= im.shape[1]] = im.shape[1] - 1
        ys[ys >= im.shape[0]] = im.shape[0] - 1

        im_patch = im[np.ix_(ys.astype(int), xs.astype(int))]

        im_patch = cv2.resize(im_patch, (model_sz[1].astype(int), model_sz[0].astype(int)))

        if first:
            self.create_tf_graph(im_patch)

        out = self.cnn_features(im_patch, cos_window)

        return out

    def create_tf_graph(self, img):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.image = tf.placeholder(tf.float32, shape=img.shape)
            processed_image = preprocess_image(self.image, image_size, image_size)
            processed_images = tf.expand_dims(processed_image, 0)

            # Create the model, use the default arg scope to configure the batch norm parameters.
            with slim.arg_scope(inception.inception_v3_arg_scope()):
                self.tensor_out, endpoints = inception.inception_v3_base(processed_images, final_endpoint='Conv2d_4a_3x3')

            self.init_fn = slim.assign_from_checkpoint_fn(os.path.abspath('checkpoint/inception_v3.ckpt'),
                                                     slim.get_model_variables('InceptionV3'))

        self.session = tf.Session(graph=self.graph)
        self.init_fn(self.session)

    def cnn_features(self, frame, cos_window):

        img_features = self.session.run(self.tensor_out, feed_dict={self.image: frame}).squeeze()
        img_features = cv2.resize(img_features, (np.floor(cos_window.shape[1]).astype(int), np.floor(cos_window.shape[0]).astype(int)))
        img_features *= cos_window[:, :, np.newaxis]

        return img_features
