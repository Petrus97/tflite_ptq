import tensorflow as tf
import keras.datasets.mnist as mnist

def main():
    dataset = mnist.load_data()
    print(dataset)


if __name__ == "__main__":
    main()