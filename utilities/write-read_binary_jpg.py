import numpy as np
from scipy import misc
from matplotlib import pyplot as plt
import os

sequence = "Biker"

files = sorted(os.listdir("data_seq/" + sequence + "/img/"))
for i, file in enumerate(files):
    files[i] = os.path.abspath("data_seq/" + sequence + "/img/" + file)

img = []
for file in files:
    img.append(misc.imread(file))

# print(img[0].shape)
# plt.imshow(img[0])
# plt.show()

file_object = open('binary/' + sequence + '.dat', "wb")

# write bunary file
for i in range(len(img)):
    img[i].tofile(file_object)
file_object.close()

# read binary file
file_object2 = open('binary/' + sequence + '.dat', "rb")

recovered_img = np.fromfile(file_object2, dtype=(np.uint8, (360, 640, 3)))
# print(recovered_img.shape)
#
# plt.imshow(recovered_img[0][:, :, :])
# plt.show()

