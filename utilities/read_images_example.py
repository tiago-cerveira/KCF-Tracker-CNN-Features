import tensorflow as tf
from matplotlib import pyplot as plt

filename_queue = tf.train.string_input_producer(['/home/tiago/Desktop/img.jpeg', '/home/tiago/Desktop/img2.jpeg', '/home/tiago/Desktop/img3.jpeg'], shuffle=False) #  list of files to read

reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)

my_img = tf.image.decode_jpeg(value) # use png or jpg decoder based on your files.

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)

    # Start populating the filename queue.

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(3): #length of your filename list
        image = my_img.eval() #here is your image Tensor :)

        print(image.shape)
        plt.imshow(image)
        plt.show()
    coord.request_stop()
    coord.join(threads)