import tensorflow as tf
import keras.datasets.mnist as mnist
import keras.utils as k_utils

MODEL_DIR = "./models"

mnist_train, mnist_test = mnist.load_data()
(x_train, y_train) = mnist_train
(x_test, y_test) = mnist_test
# Scale the images to the [0,1] range
x_train = tf.cast(x_train, dtype=tf.float32) / 255.0
x_test = tf.cast(x_test, dtype=tf.float32) / 255.0

# Make the images from shape (28,28) to (28,28,1)
x_train = tf.expand_dims(x_train, axis=-1)  # -1 adds a dimension in the last position
x_test = tf.expand_dims(x_test, axis=-1)
assert x_train[0].shape == (28, 28, 1)
# Now we have the labels in the form [5 0 4 ... 5 6 8]. We want that each one is represented as a vector (10,1)
y_train = k_utils.to_categorical(y_train, num_classes=10)
y_test = k_utils.to_categorical(y_test, num_classes=10)
assert y_train[0].shape == (10,)