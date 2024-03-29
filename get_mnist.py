from pathlib import Path
from random import random
from urllib import request, parse
import gzip
import pickle
import random
import numpy as np
import cv2


# Although I made some changes to conform to PEP8 and add some more flexibility
# attribution to the original code: https://github.com/hsjeong5/MNIST-for-Numpy/blob/master/mnist.py
_MNIST_IMAGE_FILES = (
    ("training_images", "train-images-idx3-ubyte.gz"),
    ("test_images", "t10k-images-idx3-ubyte.gz"),
)
_MNIST_LABEL_FILES = (
    ("training_labels", "train-labels-idx1-ubyte.gz"),
    ("test_labels", "t10k-labels-idx1-ubyte.gz"),
)
_MNIST_FILES = (*_MNIST_IMAGE_FILES, *_MNIST_LABEL_FILES)


def _download_mnist(data_path: Path) -> None:
    """
    Download MNIST dataset files that are not already downloaded.
    """
    base_url = "http://yann.lecun.com/exdb/mnist/"
    downloaded = False
    for key, file_name in _MNIST_FILES:
        file_path = data_path / file_name
        if not file_path.exists():
            print(f"Downloading, {file_name}. This process may take a few minutes...")
            request.urlretrieve(
                parse.urljoin(base_url, file_name), data_path / file_name
            )
            downloaded = True
    if downloaded:
        print("Download complete.")


def _save_mnist(data_path: Path):
    """
    Compress MNIST dataset into a pickle file.
    """
    mnist = {}
    for key, file_name in _MNIST_IMAGE_FILES:
        with gzip.open(data_path / file_name, "rb") as f:
            mnist[key] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(
                -1, 28, 28
            )

    for key, file_name in _MNIST_LABEL_FILES:
        with gzip.open(data_path / file_name, "rb") as f:
            mnist[key] = np.frombuffer(f.read(), np.uint8, offset=8)

    with open(data_path / "mnist.pkl", "wb") as f:
        pickle.dump(mnist, f)
    print("Save complete.")


def _clean_mnist(data_path: Path):
    """
    Clean original MNIST dataset.
    """
    for _, file_name in _MNIST_FILES:
        complete_path = data_path / file_name
        complete_path.unlink(missing_ok=True)

def rotate(image, angle):
    (h, w) = (28,28)
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    image = cv2.warpAffine(image, M, (nW, nH))

#    image = cv2.resize(image, (w,h))
    return image


def load_mnist(data_path: Path):
    """
    Download, store and return MNIST dataset.

    Parameters
    ----------
    data_path:
        A path pointing to a valid directory to store the dataset.

    Returns
    -------
    train_imgs:
        Numpy array containing MNIST training images
    train_labels:
        Numpy array containing MNIST training labels
    test_imgs:
        Numpy array containing MNIST test images
    test_labels:
        Numpy array containing MNIST test labels
    """
    if not data_path.is_dir():
        raise ValueError("Data path is not a directory")

    mnist_pkl = 'D:/ANlab/sequence-generator/data/mnist.pkl'
    # if not mnist_pkl.exists():
    #     _download_mnist(data_path)
    #     _save_mnist(data_path)
    #     _clean_mnist(data_path)
    with open(mnist_pkl, "rb") as f:
        mnist = pickle.load(f)
    # -----------------edit..............
    # img = mnist["training_images"]
    # label= mnist["training_labels"]
    # imgs= []
    # labels=[]
    # for i in range(len(img)):
    #     x = random.randint(-4,4)
    #     img[i]=rotate(img[i], angle = x)
    #     imgs.append(img[i])
    #     labels.append(label[i])

    # return(imgs, labels)
# -------------------------------
    return (
        mnist["training_images"],
        mnist["training_labels"],
        mnist["test_images"],
        mnist["test_labels"],
    )
