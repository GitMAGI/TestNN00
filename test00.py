import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from log import Log

log_lib = Log.get_instance()

def GettingStartedTF2():

    # Print the TensorFlow Version
    log_lib.debug("Tensor Flow Version %s" % tf.__version__)

    # Load and prepare the MNIST dataset. Convert the samples from integers to floating-point numbers
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Build the tf.keras.Sequential model by stacking layers. Choose an optimizer and loss function for training
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train and evaluate the model
    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test)

    return

def ImageClassification():
    # Print the TensorFlow Version
    log_lib.debug("Tensor Flow Version %s" % tf.__version__)

    # Load MNIST dataset
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    log_lib.debug("Train Images %d" % len(train_images))
    log_lib.debug("Train Labels %d" % len(train_labels))
    log_lib.debug("Test Images %d" % len(test_images))
    log_lib.debug("Test Labels %d" % len(test_labels))

    return