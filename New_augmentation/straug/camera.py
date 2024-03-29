"""
Camera sensors have many imperfections and tunable settings. 
1) Contrast, 
2) Brightness, 
3) JpegCompression 
and 
4) Pixelate. 

Contrast enables us to distinguish the different objects that compose an image. 
Brightness is directly affected by scene luminance. 
JpegCompression is the side effect of image compression. 
Pixelate is exhibited by increasing the resolution of an image.

Reference: https://github.com/hendrycks/robustness
Hacked together for STR by: Rowel Atienza
"""

from io import BytesIO

import numpy as np
import skimage as sk
from PIL import Image, ImageOps, ImageEnhance
from skimage import color

'''
    PIL resize (W,H)
    cv2 image is BGR
    PIL image is RGB
'''


class Contrast:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        # c = [0.4, .3, .2, .1, .05]
        c = [0.4, .3, .2]
        if mag < 0 or mag >= len(c):
            index = self.rng.integers(0, len(c))
        else:
            index = mag
        c = c[index]
        img = np.asarray(img) / 255.
        means = np.mean(img, axis=(0, 1), keepdims=True)
        img = np.clip((img - means) * c + means, 0, 1) * 255

        return Image.fromarray(img.astype(np.uint8))

class LightAndShadow:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        brightness_range = [-0.7, 0.2]  # adjust brightness by up to 20%
        contrast_range = [0.7, 1.3]  # adjust contrast by a factor of up to 1.5

        # adjust brightness
        brightness_factor = self.rng.uniform(*brightness_range)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1 + brightness_factor)

        # # adjust contrast
        # contrast_factor = self.rng.uniform(*contrast_range)
        # enhancer = ImageEnhance.Contrast(img)
        # img = enhancer.enhance(contrast_factor)

        return img


class Brightness:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        # W, H = img.size
        c = [.1, .2, .3, .4, .5]
        # c = [.1, .2, .3]
        if mag < 0 or mag >= len(c):
            index = self.rng.integers(0, len(c))
        else:
            index = mag
        c = c[index]

        n_channels = len(img.getbands())
        isgray = n_channels == 1

        img = np.asarray(img) / 255.
        if isgray:
            img = np.expand_dims(img, axis=2)
            img = np.repeat(img, 3, axis=2)

        img = sk.color.rgb2hsv(img)
        img[:, :, 2] = np.clip(img[:, :, 2] + c, 0, 1)
        img = sk.color.hsv2rgb(img)

        # if isgray:
        #    img = img[:,:,0]
        #    img = np.squeeze(img)

        img = np.clip(img, 0, 1) * 255
        img = Image.fromarray(img.astype(np.uint8))
        if isgray:
            img = ImageOps.grayscale(img)

        return img
        # if isgray:
        # if isgray:
        #    img = color.rgb2gray(img)

        # return Image.fromarray(img.astype(np.uint8))


class JpegCompression:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        # c = [25, 18, 15, 10, 7]
        c = [25, 18, 15]
        if mag < 0 or mag >= len(c):
            index = self.rng.integers(0, len(c))
        else:
            index = mag
        c = c[index]
        output = BytesIO()
        img.save(output, 'JPEG', quality=c)
        return Image.open(output)


class Pixelate:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        w, h = img.size
        # c = [0.6, 0.5, 0.4, 0.3, 0.25]
        c = [0.6, 0.5, 0.4]
        if mag < 0 or mag >= len(c):
            index = self.rng.integers(0, len(c))
        else:
            index = mag
        c = c[index]
        img = img.resize((int(w * c), int(h * c)), Image.BOX)
        return img.resize((w, h), Image.BOX)
