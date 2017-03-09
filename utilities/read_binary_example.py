import numpy as np
from matplotlib import pyplot as plt

file_object = open("binary/Biker_features.dat", "rb")

array = np.fromfile(file_object, dtype=(np.float32, (147, 147, 64)))
print(array.shape)
for i in range(3):
    plt.imshow(array[17][:, :, i+25])
    plt.show()