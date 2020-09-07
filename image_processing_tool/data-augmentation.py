from PIL import Image, ImageDraw
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import os
import xml.etree.ElementTree as ET

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def get_random_data(image, box, jitter=.3, hue=.2, sat=1.5, val=1.5, proc_img=True):
    '''random preprocessing for real-time data augmentation'''
    iw, ih = image.size

    # 对图像进行缩放并且进行长和宽的扭曲
    new_ar = rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
    scale = rand(.25,2)
    nh = int(scale*ih)
    nw = int(scale*iw*new_ar)
    image = image.resize((nw,nh), Image.BICUBIC)

    # 翻转图像
    # flip = 1
    # if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # 色域扭曲
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
    val = rand(1, val) if rand()<.5 else 1/rand(1, val)
    x = rgb_to_hsv(np.array(image)/255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0]>1] -= 1
    x[..., 0][x[..., 0]<0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x>1] = 1
    x[x<0] = 0
    image_data = hsv_to_rgb(x) # numpy array, 0 to 1

    # 将box进行调整
    box_data = []
    for values in box:
        values[0] *= scale*new_ar
        values[1] *= scale
        values[2] *= scale*new_ar
        values[3] *= scale
        values = list(map(int, values))
        box_data.append(values)
        
    
    return image_data, box_data


def get_box_from_xml(filename):
    tree = ET.parse('test\\' + filename[:-4] + '.xml')
    root = tree.getroot()
    classes = []
    values = []
    for member in root.findall('object'):
        # root.find('filename').text,
        # int(root.find('size')[0].text),
        # int(root.find('size')[1].text),
        classes.append(member[0].text)
        value = [int(member[4][0].text),
                int(member[4][1].text),
                int(member[4][2].text),
                int(member[4][3].text)]
                 
        values.append(value)
    return classes, values
    
def save_xml(filename, box_data, number):
    tree = ET.parse('test\\' + filename[:-4] + '.xml')
    root = tree.getroot()
    pic = root.findall('filename')
    pic[0].text =  (pic[0].text)[:-4] + '_' + str(number) + '.jpg'
    i = 0
    for member in root.findall('object'):
        member[4][0].text = str(box_data[i][0])
        member[4][1].text = str(box_data[i][1])
        member[4][2].text = str(box_data[i][2])
        member[4][3].text = str(box_data[i][3])
        i = i+1
    tree.write('data-augmentation\\' + filename[:-4] + '_' + str(number) + '.xml')

    return 



dir_path = os.getcwd()




for filename in os.listdir(dir_path+'\\test'):
    if filename.endswith(".jpg"):

        for i in range(4):
            print(filename)
            img = Image.open('test\\' + filename)
            classes, box = get_box_from_xml(filename)
        
            # get new image
            image_data, box_data = get_random_data(img, box)
            new_img = Image.fromarray((image_data*255).astype(np.uint8))
            new_img.save('data-augmentation\\' + filename[:-4] + '_' + str(i) + '.jpg')
            save_xml(filename, box_data, i)
            # for j in range(len(box_data)):
                # thickness = 3
                # left, top, right, bottom  = box_data[j][0:4]
                # draw = ImageDraw.Draw(new_img)
                # for i in range(thickness):
                    # draw.rectangle([left + i, top + i, right - i, bottom - i],outline=(255,255,255))
            # new_img.show()
        

