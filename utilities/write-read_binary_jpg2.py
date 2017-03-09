import numpy as np
from scipy import misc
from matplotlib import pyplot as plt
import os

# sequence = "CarScale"
#
# files = sorted(os.listdir("data_seq/" + sequence + "/img/"))
# for i, file in enumerate(files):
#     files[i] = os.path.abspath("data_seq/" + sequence + "/img/" + file)
#
# img = []
# for file in files:
#     img.append(misc.imread(file))
#
# # print(img[0].shape)
# # plt.imshow(img[0])
# # plt.show()
#
# np.savez('binary/' + sequence + '.npz', *img)

# new = np.load('binary/' + sequence + '.npz')
new = np.load('binary/Car2_features.npz')
keys = new.keys()
sorted_keys = sorted(keys, key=lambda x:int(x[4:]))
# print(sorted_keys)

img = new[sorted_keys[0]]
print(img[23, 75, 3])

# for i in range(len(new.files)):
#     img = new[sorted_keys[i]]
#     plt.imshow(img)
#     plt.show()
