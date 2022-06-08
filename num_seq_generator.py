from typing import Iterable, Tuple

import numpy as np
from PIL import Image
import random

# changed function signature slightly for performance
def generate_numbers_sequence(
    digits: Iterable[int],
    spacing_range: Tuple,
    image_width: int,
    data: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """
    Generate an image that contains the sequence of given numbers, spaced
    randomly using an uniform distribution.

    Parameters
    ----------
    digits:
        A list-like containing the numerical values of the digits from which
        the sequence will be generated (for example [3, 5, 0]).
    spacing_range:
        A (minimum, maximum) pair (tuple), representing the min and max spacing
        between digits. Unit should be pixel.
    image_width:
        Specifies the width of the image in pixels.
    data:
        A dataset represented as a numpy array containing the images to use to
        generate number sequences.
    labels:
        Labels corresponding to the dataset represented as a numpy array of integers.

    Returns
    -------
    The image containing the sequence of numbers. Images should be represented
    as floating point 32bits numpy arrays with a scale ranging from 0 (black) to
    1 (white), the first dimension corresponding to the height and the second
    dimension to the width.
    """

    _is_valid_input_arguments(digits, data, labels)

    rng = np.random.default_rng()
    spacing = rng.integers(spacing_range[0], spacing_range[1] + 1, len(digits) - 1)
    spacing_imgs=[]
    for i in range(len(digits)-1):
        # spacing_img = [
        #     np.zeros((28, space)).astype("float32") if space != 0 else None
        #     for space in spacing
        # ]

        space= random.randint(spacing_range[0], spacing_range[1] + 1)
        spacing_img= np.zeros((28, space)).astype("float32")
        spacing_imgs.append(spacing_img)
    # print(spacing_imgs)

    img_seq = []
    width = data.shape[1]
    height = data.shape[2]
    for i, label in enumerate(digits):
        random_image_from_label = rng.choice(data[labels == label], 1, replace=False)
        formatted_image = random_image_from_label.reshape(width, height).astype(
            "float32"
        )
        # print(random_image_from_label.dtype())
        
        if i != 0 and spacing_imgs[i - 1] is not None:
            img_seq.extend([spacing_imgs[i - 1], formatted_image])
        else:
            img_seq.append(formatted_image)

    return _concatenate_images(img_seq, image_width)


def _is_valid_input_arguments(
    digits: Iterable[int], data: np.ndarray, labels: np.ndarray
):
    if len(data.shape) != 3:
        raise ValueError(
            f"Wrong number of dimensions. Expected 3 dimensions [n_samples, width, height], "
            f"found {len(data.shape)}"
        )
    if len(labels.shape) != 1:
        raise ValueError(
            f"Wrong number of dimensions. Expected 1 dimension [n_samples], found {len(labels.shape)}"
        )
    if data.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Number of samples of data and labels is different. Found {data.shape[0]} samples and "
            f"{labels.shape[0]} labels"
        )
    if not set(digits).issubset(set(labels)):
        raise ValueError("Some of the provided digits do not appear on the dataset")


def _concatenate_images(img_seq: Iterable[np.ndarray], width: int) -> np.ndarray:
    final_image = np.hstack(img_seq).astype("float32")
    resized_image = Image.fromarray(final_image)
    resized_image = resized_image.resize((width, 28))
    normalized_image = np.array(resized_image) / 255
    return normalized_image.astype("float32")
