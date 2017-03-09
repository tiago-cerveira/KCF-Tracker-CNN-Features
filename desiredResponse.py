import numpy as np


def gaussian_response_2d(output_sigma, sz):
    rs, cs = np.mgrid[1:sz[0]+1, 1:sz[1]+1]
    rs -= np.floor(sz[0]/2)
    cs -= np.floor(sz[1]/2)
    y = np.exp(-0.5 * (np.square(rs) + np.square(cs)) / np.square(output_sigma))
    # Show the desired output y
    # import cv2
    # cv2.imshow("Desired output", y)
    # cv2.waitKey(0)
    row_shift, col_shift = -np.floor(np.array(y.shape)/2).astype(int)+1
    y = np.roll(y, col_shift, axis=1)
    y = np.roll(y, row_shift, axis=0)
    return y


def gaussian_response_1d(sigma, sz):
    ss = np.arange(1, sz + 1) - np.ceil(sz / 2.0)
    ys = np.exp(-0.5 * (ss ** 2) / sigma**2)
    return ys
