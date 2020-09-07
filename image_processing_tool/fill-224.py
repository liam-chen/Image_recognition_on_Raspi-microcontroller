## Bulk image resizer

# This script simply resizes all the images in a folder to one-eigth their
# original size. It's useful for shrinking large cell phone pictures down
# to a size that's more manageable for model training.

# Usage: place this script in a folder of images you want to shrink,
# and then run it.

import numpy as np
import cv2
import os
from PIL import Image

dir_path = os.getcwd()
# dir_path = "D:\\Uni_Stuttgart\\MA\\image5"

def make_square(im, min_size=224, fill_color=(0, 0, 0, 0)):
    image = im.convert('RGB')
    w, h = image.size
    background = Image.new('RGB', size=(max(w, h), max(w, h)), color=(127, 127, 127))  # 创建背景图，颜色值为127
    length = int(abs(w - h) // 2)  # 一侧需要填充的长度
    box = (length, 0) if w < h else (0, length)  # 粘贴的位置
    background.paste(image, box)
    image_data=background.resize((224,224))#缩放
    # background.show()
    # image_data.show()
    return image_data



for filename in os.listdir(dir_path+'\\resized_picture'):
    # If the images are not .JPG images, change the line below to match the image type.
    if filename.endswith(".jpg"):
        print(filename)
        filenamex = 'resized_picture\\' + filename
        image = Image.open(filenamex)
        newimage = make_square(image)
        # print(newimage.shape)
        newimage.save('new\\'+filename)



