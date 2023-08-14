#!/usr/bin/env python

from feynman_torch import FeynmanModel


def main():
    # Usage:
    model = FeynmanModel(3)
    model.train(dataset_dir="./dataset/train/", epochs=10, batch_size=32)

if __name__ == "__main__":
    main()
