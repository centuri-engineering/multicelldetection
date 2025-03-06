import numpy as np
from skimage import color, img_as_float, img_as_ubyte


def convert_RGB_to_HSV(image_rgb):
    # Convert the image to float (values between 0 and 1)
    image_float = img_as_float(image_rgb)

    # Convert the RGB image to HSV
    image_hsv = color.rgb2hsv(image_float)

    # conver to 8-bit image and return
    return img_as_ubyte(image_hsv)

def convert_RGB_to_grayscale(image_rgb):
    # Convert the image to float (values between 0 and 1)
    image_float = img_as_float(image_rgb)

    # Convert the RGB image to grayscale
    image_hsv = color.rgb2gray(image_float)

    # conver to 8-bit image and return
    return img_as_ubyte(image_hsv)

def merge_channels(image_rgb, image_hsv):
    # Extract channels:
    r_channel = image_rgb[..., 0]  # Red from RGB
    s_channel = image_hsv[..., 1]  # Saturation from HSV

    # Create a new image by concatenating the channels (R, S, B)
    new_image = np.dstack((s_channel, s_channel, r_channel))

    # Convert the new image to 8-bit image and return
    return img_as_ubyte(new_image)        
