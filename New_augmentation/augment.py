import cv2
import numpy as np
import os
from torchvision.transforms import functional as F
from torchvision.transforms import AugMix
import time
import random
import threading


def apply_random_contrast(image, contrast_range=(0.5, 1.5)):
    """
    Apply random contrast adjustment to the input image.

    Args:
        image (numpy.ndarray): Input image.
        contrast_range (tuple): Tuple containing the minimum and maximum contrast values.
                                Random contrast adjustment will be sampled from this range.

    Returns:
        numpy.ndarray: Image with random contrast adjustment applied.
    """
    # Convert image to float32
    image = image.astype(np.float32)

    # Generate random contrast adjustment factor
    contrast_factor = np.random.uniform(contrast_range[0], contrast_range[1])

    # Calculate mean pixel value of the image
    mean_pixel_value = np.mean(image)

    # Apply contrast adjustment
    image = (image - mean_pixel_value) * contrast_factor + mean_pixel_value

    # Clip pixel values between 0 and 255
    image = np.clip(image, 0, 255)

    # Convert image back to uint8
    image = image.astype(np.uint8)

    return image

def apply_random_brightness(image, brightness_range=(0.5, 1.5)):
    """
    Apply random brightness adjustment to the input image.

    Args:
        image (numpy.ndarray): Input image.
        brightness_range (tuple): Tuple containing the minimum and maximum brightness values.
                                  Random brightness adjustment will be sampled from this range.

    Returns:
        numpy.ndarray: Image with random brightness adjustment applied.
    """
    # Convert image to float32
    image = image.astype(np.float32)

    # Generate random brightness adjustment factor
    brightness_factor = np.random.uniform(brightness_range[0], brightness_range[1])

    # Apply brightness adjustment
    image = image * brightness_factor

    # Clip pixel values between 0 and 255
    image = np.clip(image, 0, 255)

    # Convert image back to uint8
    image = image.astype(np.uint8)

    return image

# Add Gaussian noise to image
def add_gaussian_noise(image, mean, std):
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image


def apply_gaussian_blur(image, sigma=0.1):
    # Calculate kernel size based on image shape
    shape = image.shape
    kernel_size = (int(sigma*shape[0]), int(sigma*shape[1]))
    if kernel_size[0] % 2 == 0:
        kernel_size = (kernel_size[0] + 1, kernel_size[1])
    if kernel_size[1] % 2 == 0:
        kernel_size = (kernel_size[0], kernel_size[1] + 1)
    
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
    return blurred_image

# Downsample and upsample image
def down_up_sample(image, scale_factor):
    # Downsample image
    downsampled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    # Upsample image
    upsampled_image = cv2.resize(downsampled_image, image.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)
    return upsampled_image

# Apply sharpening filter to image
def apply_sharpening_filter(image):
    kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    sharpened_image = cv2.filter2D(image,-1,kernel)
    return sharpened_image

def apply_augmentation(image, augmentation_function, result_list, index):
    result_list[index] = augmentation_function(image)

def augment_image(image):
    '''
    Input: one image  ( read by cv2)
    Output List Images (cv2)
    '''
    # Define parameters for augmentations
    shape = image.shape[0]
    level_noise = int(shape/30)
    noise_std_levels = [level_noise, level_noise*2, level_noise*3]
    
    # Define list to hold augmented images
    augmented_images = [None] * 9

    # Define list to hold threads
    threads = []

    # Apply image augmentations in parallel using threads
    threads.append(threading.Thread(target=apply_augmentation, args=(image, apply_sharpening_filter, augmented_images, 0)))
    threads.append(threading.Thread(target=apply_augmentation, args=(image, lambda x:add_gaussian_noise(x, 0, noise_std_levels[0]), augmented_images, 1)))
    threads.append(threading.Thread(target=apply_augmentation, args=(image, lambda x:add_gaussian_noise(x, 0, noise_std_levels[1]), augmented_images, 2)))
    threads.append(threading.Thread(target=apply_augmentation, args=(image, lambda x:add_gaussian_noise(x, 0, noise_std_levels[2]), augmented_images, 3)))
    threads.append(threading.Thread(target=apply_augmentation, args=(image, lambda x:apply_gaussian_blur(x, sigma=0.02), augmented_images, 4)))
    threads.append(threading.Thread(target=apply_augmentation, args=(image, lambda x:apply_gaussian_blur(x, sigma=0.05), augmented_images, 5)))
    threads.append(threading.Thread(target=apply_augmentation, args=(image, lambda x:apply_gaussian_blur(x, sigma=0.07), augmented_images, 6)))
    threads.append(threading.Thread(target=apply_augmentation, args=(image, lambda x:apply_random_brightness(x, brightness_range=(0.3, 1.7)), augmented_images, 7)))
    threads.append(threading.Thread(target=apply_augmentation, args=(image, lambda x:apply_random_contrast(x, contrast_range=(0.5, 1.5)), augmented_images, 8)))

    # Start threads
    for thread in threads:
        thread.start()

    # Wait for threads to finish
    for thread in threads:
        thread.join()

    # Add AugMix augmentation
    augmix = AugMix(severity = 4)
    pil_image = F.to_pil_image(image)
    for _ in range(3):
        augmix_image = np.array(augmix(pil_image))
        augmented_images.append(augmix_image)

    return augmented_images

if __name__ == "__main__":
    folder_input = "/home/sonnt373/Desktop/SoNg/Face_quality/dev/data/data_train_100k_140723/"

    for data_clas in os.listdir(folder_input):
        t0 = time.time()
        data_folder = os.path.join(folder_input, data_clas)
        list_img = random.sample(os.listdir(data_folder), k = int(0.3 * len(os.listdir(data_folder))))

        for path in list_img:
            image_name= path[:-4]
            image_path = os.path.join(data_folder , path)
            input_image = cv2.imread(image_path)
            augmented_images = augment_image(input_image)
            for i, augmented_image in enumerate(augmented_images):
                name = str(image_name)+"_aug" + str(i) +'.jpg'
                if max(augmented_image.shape) > 300:
                    augmented_image = cv2.resize(augmented_image, (300, 300), interpolation=cv2.INTER_AREA)

                # Random quality
                quality_save = random.randint(70,100)
                cv2.imwrite(os.path.join(data_folder, name), augmented_image, [cv2.IMWRITE_JPEG_QUALITY, quality_save])