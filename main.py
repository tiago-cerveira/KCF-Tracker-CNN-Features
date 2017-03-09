from __future__ import print_function
import cv2
from matplotlib import pyplot as plt
from util import *
# import pickle
# from PIL import Image
import numpy as np
from tracker import Tracker
from kernel_params import Params



def main():

    # video_path = select_sequence()
    video_path = '/home/tiago/data_seq/Car2'

    parameters = Params()

    img_files, pos, target_sz, ground_truth, video_path = load_video_info(video_path)  # Load video info

    parameters.init_pos = np.floor(pos) + np.floor(target_sz / 2)                                # Initial position
    parameters.pos = parameters.init_pos                                                         # Current position
    parameters.target_size = np.floor(target_sz)                                                 # Size of target
    parameters.img_files = img_files                                                             # List of image files
    parameters.video_path = video_path                                                           # Path to the sequence

    num_frames = len(img_files)
    results = np.zeros((num_frames, 4))

    start = start_timer()

    # For each frame
    for frame in xrange(num_frames):

        # Read the image
        im = cv2.imread(video_path + img_files[frame], 1)


        # Initialize the tracker using the first frame
        if frame == 0:
            tracker1 = Tracker(im, parameters)
            tracker1.train(im, True)
            results[frame, :] = np.array(
                (pos[0] + np.floor(target_sz[0] / 2), pos[1] + np.floor(target_sz[1] / 2), target_sz[0], target_sz[1]))
        else:
            results[frame, :], lost, xtf = tracker1.detect(im)  # Detect the target in the next frame
            if not lost:
                tracker1.train(im, False, xtf)  # Update the model with the new infomation
        if parameters.visualization:
            # Draw a rectangle in the estimated location and show the result
            cvrect = np.array((results[frame, 1] - results[frame, 3] / 2,
                               results[frame, 0] - results[frame, 2] / 2,
                               results[frame, 1] + results[frame, 3] / 2,
                               results[frame, 0] + results[frame, 2] / 2))
            cv2.rectangle(im, (cvrect[0].astype(int), cvrect[1].astype(int)),
                          (cvrect[2].astype(int), cvrect[3].astype(int)), (0, 255, 0), 2)
            cv2.imshow('Window', im)
            cv2.waitKey(1)
            print(frame, end='\t')

    duration = end_timer(start, "to complete tracking")
    fps = round(num_frames/duration, 2)
    print(fps, "FPS")

    np.savetxt('results.txt', results, delimiter=',', fmt='%d')

    if parameters.debug:
        plt.figure()
        plt.plot(tracker1.confidence)
        plt.draw()

        plt.figure()
        plt.plot(tracker1.psr)
        plt.draw()

        plt.figure()
        plt.plot(tracker1.high_freq_energy)
        plt.show()

if __name__ == "__main__":
    main()
