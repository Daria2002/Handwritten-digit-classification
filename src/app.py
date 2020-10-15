import os
import tensorflow as tf
import tensorflow as tfjs
from tensorflow import keras

def load_mnist_dataset():
    # split the minst dataset into train and test datasets
    (train_img, train_label), (test_img, test_label) = keras.datasets.mnist.load_data()
    # reshape the input vector to a 4-dims numpy arrays and normalize the input by dividing the RGB codes to 255
    train_img = train_img.reshape([-1, 28, 28, 1])
    test_img = test_img.reshape([-1, 28, 28, 1])
    train_img = train_img/255.0
    test_img = test_img/255.0
    # convert class vectors to binary class matrices using one-hot encoding
    train_label = keras.utils.to_categorical(train_label)
    test_label = keras.utils.to_categorical(test_label)

def define_the_cnn_model():
    model = keras.Sequential([
        keras.layers.Conv2D(32, (5, 5), padding="same", input_shape=[28, 28, 1]),
        keras.layers.MaxPool2D((2,2)),
        keras.layers.Conv2D(64, (5, 5), padding="same"),
        keras.layers.MaxPool2D((2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

def train_and_save_the_model():
    load_mnist_dataset()
    define_the_cnn_model()

if __name__ == "__main__":
    train_and_save_the_model()
