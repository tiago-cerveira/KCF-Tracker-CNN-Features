import numpy as np
from scipy import misc
from matplotlib import pyplot as plt
import os
import h5py

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

with h5py.File('binary/' + sequence + '.h5', 'w') as hf:
    for i, im in enumerate(img):
        hf.create_dataset('{:04}'.format(i), data=im)

with h5py.File('binary/' + sequence + '.h5', 'r') as hf:
    for item in hf:
        print(item)
        print(hf[item].shape)
        plt.imshow(hf[item])
        plt.show()
