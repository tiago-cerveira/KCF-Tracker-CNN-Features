import os

list_images = sorted(os.listdir())

with open('images.txt', 'w') as f:
    for img in list_images:
        if img[0].isdigit():
            f.write(os.path.abspath(img) + '\n')


