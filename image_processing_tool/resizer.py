## Bulk image resizer

# This script simply resizes all the images in a folder to one-eigth their
# original size. It's useful for shrinking large cell phone pictures down
# to a size that's more manageable for model training.

# Usage: place this script in a folder of images you want to shrink,
# and then run it.

import numpy as np
import cv2
import os

dir_path = os.getcwd()
# dir_path = "D:\\Uni_Stuttgart\\MA\\image5"

for filename in os.listdir(dir_path):
    # If the images are not .JPG images, change the line below to match the image type.
    if filename.endswith(".jpg"):
        print(filename)
        image = cv2.imread(filename)
        height, width, _ = image.shape
        ratio1 = 240 / height
        ratio2 = 320 / width
        ratio = min(ratio1, ratio2)
        print(height, width)
        resized = cv2.resize(image, None,fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
        print(resized.shape)
        cv2.imwrite(filename,resized)