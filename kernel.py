import numpy as np

def linear_correlation(xf, yf, features):

    # Deprecated code. To be removed.
    # if features == 'HoG' or features == 'CNN':
    #     kf = np.sum(xf * np.conjugate(yf), axis=2) / np.size(xf)
    # else:
    #     kf = xf * np.conjugate(yf) / np.size(xf)

    if len(xf.shape) == 3:
        kf = np.sum(xf * np.conjugate(yf), axis=2) / np.size(xf)
    elif len(xf.shape) == 2:
        kf = xf * np.conjugate(yf) / np.size(xf)
    else:
        raise ValueError("Length of shape of xyf (number of feature channels) should be either 2 or 3.")

    return kf


def gaussian_correlation(xf, yf, sigma, features):

    n = xf.shape[0]*xf.shape[1]
    xx = np.linalg.norm(xf)**2 / n
    yy = np.linalg.norm(yf)**2 / n

    xyf = xf * np.conjugate(yf)

    # Deprecated code. To be removed.
    # if features == 'HoG' or features == 'CNN':
    #     xy = np.sum(np.real(np.fft.ifft2(xyf, axes=(0, 1))), axis=2)
    # else:
    #     xy = np.real(np.fft.ifft2(xyf, axes=(0, 1)))

    if len(xyf.shape) == 3:
        xy = np.sum(np.real(np.fft.ifft2(xyf, axes=(0, 1))), axis=2)
    elif len(xyf.shape) == 2:
        xy = np.real(np.fft.ifft2(xyf, axes=(0, 1)))
    else:
        raise ValueError("Length of shape of xyf (number of feature channels) should be either 2 or 3.")

    gaussianmean = (xx + yy - 2 * xy) / np.size(xf)
    gaussianmean[gaussianmean < 0] = 0

    kf = np.fft.fft2(np.exp(-1 / sigma**2 * gaussianmean))

    return kf
