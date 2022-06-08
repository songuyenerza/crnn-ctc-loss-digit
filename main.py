import argparse
from cgi import test
from itertools import count
from pathlib import Path
import random

from PIL import Image

import PIL.ImageOps
from numpy import quantile  

from get_mnist import load_mnist
from num_seq_generator import generate_numbers_sequence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence", help="Number to be generated")
    parser.add_argument(
        "min_spacing",
        help="Minimum pixel spacing between consecutive digits (inclusive)",
        type=int,
    )
    parser.add_argument(
        "max_spacing",
        help="Maximum pixel spacing between consecutive digits (exclusive)",
        type=int,
    )
    parser.add_argument(
        "image_width", help="Pixel width of the generated image", type=int
    )
    parser.add_argument("-o", help="Path to load/store MNIST dataset")
    args = parser.parse_args()

    if args.min_spacing < 0:
        raise ValueError("Minimum spacing can't be negative")
    if args.max_spacing < 0:
        raise ValueError("Maximum spacing can't be negative")
    if args.image_width < 1:
        raise ValueError("Image width can't be less than 1")
    if args.max_spacing <= args.min_spacing:
        raise ValueError("Maximum spacing has to be greater than minimum space")

    out_path = Path(args.o) if args.o is not None else Path("data")
    if not out_path.exists():
        out_path.mkdir()
    if not out_path.is_dir():
        raise ValueError("Path to MNIST dataset is not a directory")
    print(args.sequence)
    sequence = list(map(int, str(args.sequence)))
    train_imgs, train_labels, _, _ = load_mnist(out_path)
    number_sequence_image = generate_numbers_sequence(
        sequence,
        (args.min_spacing, args.max_spacing),
        args.image_width,
        train_imgs,
        train_labels,
    )

    im = Image.fromarray(number_sequence_image * 255)
    im = im.convert("L")
    im = PIL.ImageOps.invert(im)
    # print(args.sequence)
    count= random.randint(0,9999999999999)
    name = 'datagen'+'/' + str(args.sequence)+"_" + str(count) +'.png'
    im.save(name, quality= 100)
