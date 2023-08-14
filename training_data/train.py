#!/usr/bin/env python

import logging
from feynman_torch import FeynmanModel

from colored import Fore, Back, Style

logging.basicConfig(
    level=logging.DEBUG,
    format=f"{Fore.rgb(100, 100, 100)}%(asctime)s {Fore.rgb(255, 255, 255)}{Back.rgb(80, 200, 80)} FeynMAN {Style.reset} [{Fore.rgb(255, 240, 240)}%(levelname)s{Style.reset}] %(message)s",
)

def main():
    # Usage:
    logging.info("Creating model...")
    model = FeynmanModel(3)
    logging.info("START TRAINING!")
    model.train(dataset_dir="./dataset/train/", epochs=10, batch_size=32)
    logging.info("END TRAINING!")


if __name__ == "__main__":
    main()
