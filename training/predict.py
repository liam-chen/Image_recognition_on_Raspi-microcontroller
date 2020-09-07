import numpy as np
np.random.seed(111)
import argparse
import os
import json
from yolo.frontend import create_yolo, get_object_labels
import cv2
import glob

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    default="configs/from_scratch.json",
    help='path to configuration file')


if __name__ == '__main__':
    args = argparser.parse_args()
    config_path = args.conf
    weights_path = '2020-05-07_21-40-42.h5'

    with open(config_path) as config_buffer:
        config = json.load(config_buffer)
    if config['train']['is_only_detect']:
        labels = ["object"]
    else:
        if config['model']['labels']:
            labels = config['model']['labels']
        else:
            labels = get_object_labels(config['train']['train_annot_folder'])
    print(labels)

    # 1. Construct the model
    yolo = create_yolo(config['model']['architecture'],
                       labels,
                       config['model']['input_size'],
                       config['model']['anchors'],
                       config['model']['coord_scale'],
                       config['model']['class_scale'],
                       config['model']['object_scale'],
                       config['model']['no_object_scale'])

    yolo.load_weights(weights_path)

    CWD_PATH = os.getcwd()
    IM_DIR = 'val_img'
    PATH_TO_IMAGES = os.path.join(CWD_PATH, IM_DIR)
    txt_path = os.path.join(os.getcwd(), 'detection-results')
    images = glob.glob(PATH_TO_IMAGES + '/*.jpg')

    for image_path in images:
        # print(image_path[40:-4])
        image_id = image_path[40:-4]
        image = cv2.imread(image_path)
        boxs, probs = yolo.predict(image, threshold=0.05)
        print("boxs:", boxs)
        print('probs:', probs)
        if boxs == []:
            print('Error!!!')
            with open(txt_path + "/noresult.txt", "a") as new_f:
                new_f.write("%s \n" % (image_id))
        else:
            for i in range(probs.shape[0]):
                box = boxs[i]
                prob = list(probs[i])
                print(prob)
                left = box[0]  # xmin
                bottom = box[1]  # ymin
                right = box[2]  # xmax
                top = box[3]  # ymax
                prob_max = max(prob)
                prob_index = prob.index(prob_max)
                obj_name = labels[prob_index]
                with open(txt_path + '\\' + image_id + ".txt", "a") as new_f:
                    new_f.write("%s %s %s %s %s %s\n" % (obj_name, prob_max, left, bottom, right, top))

