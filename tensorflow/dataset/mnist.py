import os

import tensorflow as tf


def main():
    (x_train, y_train), (x_test, y_test) \
        = tf.keras.datasets.mnist.load_data(path='mnist.npz')

    print('==== x_train ====')
    print(type(x_train))

    print('==== y_train ====')
    print(type(y_train))

    print('==== x_test ====')
    print(type(x_test))

    print('==== y_test ====')
    print(type(y_test))


if __name__ == '__main__':
    main()
