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

def ClothingDataSetExploring():
    # Print the TensorFlow Version
    log_lib.debug("Tensor Flow Version %s" % tf.__version__)

    # Load MNIST dataset (Training + Test)
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    log_lib.debug("Train Images %d with size of %dx%d" % train_images.shape)
    log_lib.debug("Train Labels %d" % len(train_labels))    
    log_lib.debug("Test Images %d with size of %dx%d" % test_images.shape)
    log_lib.debug("Test Labels %d" % len(test_labels))

    # All Images belong to hese classes
    class_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Plotting the first image of the training set
    
    plt.figure()
    plt.imshow(train_images[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()

    # Normalization
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Displaying the first 25 Images of the training set
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()

    return