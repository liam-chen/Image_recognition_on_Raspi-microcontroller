#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import cv2
import os, sys, argparse, json
from tensorflow.lite.python import interpreter as interpreter_wrapper
from common.data_utils import preprocess_image
import numpy as np
from backend.box import to_minmax
from backend.decoder import YoloDecoder
import glob

def validate_yolo_model_tflite(model_path, image, anchors, class_names, loop_count):
    interpreter = interpreter_wrapper.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # check the type of the input tensor
    if input_details[0]['dtype'] == np.float32:
        floating_model = True
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    imH, imW, _ = image.shape
    image_shape = (imH, imW)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    model_image_size = (height, width)
    image_data = preprocess_image(image_rgb, model_image_size)

    # predict once first to bypass the model building time
    interpreter.set_tensor(input_details[0]['index'], image_data)
    interpreter.invoke()

    # start = time.time()
    for i in range(loop_count):
        interpreter.set_tensor(input_details[0]['index'], image_data)
        interpreter.invoke()

    prediction = []
    for output_detail in output_details:
        output_data = interpreter.get_tensor(output_detail['index'])
        prediction.append(output_data)
    # end = time.time()
    # print("Average Inference time: {:.8f}ms".format((end - start) * 1000 / loop_count))

    boxes, classes, scores = handle_prediction(prediction,  image, image_shape, anchors, class_names,
                                               model_image_size)

    return boxes, classes, scores

def _to_original_scale(image_shape, boxes):
    height, width = image_shape
    minmax_boxes = to_minmax(boxes)
    minmax_boxes[:, 0] *= width
    minmax_boxes[:, 2] *= width
    minmax_boxes[:, 1] *= height
    minmax_boxes[:, 3] *= height
    return minmax_boxes.astype(np.int)

def _probs_to_classes_scores(probs):

    classes = []
    scores = []

    for prob in probs:
        scores.append(max(prob))
        classes.append(np.argmax(prob))
    return classes, scores


def handle_prediction(prediction, image, image_shape, anchors, class_names, model_image_size):
    netout = prediction[0].reshape(7,7,5,13)

    yolodecoder = YoloDecoder(anchors=[1.51, 2.43, 1.83, 1.54, 2.26, 3.08, 2.59, 2.07, 3.75, 3.41])
    boxes, probs = yolodecoder.run(netout, 0.3)
    # print('boxes, probs:', boxes, probs)
    if boxes != []:
        boxes = _to_original_scale(image_shape, boxes)
        # print('boxes:',boxes)
    else:
        boxes = []
        print('Error!!!!!')
    classes, scores = _probs_to_classes_scores(probs)
    return boxes, classes, scores




def main():
    parser = argparse.ArgumentParser(description='validate YOLO model (h5/pb/tflite/mnn) with image')
    parser.add_argument('--loop_count', help='loop inference for certain times', type=int, default=1)
    parser.add_argument('--custom_objects', required=False, type=str, help="Custom objects in keras model (swish/tf). Separated with comma if more than one.", default=None)

    args = parser.parse_args()
    CWD_PATH = os.getcwd()
    IM_DIR = 'val_img'
    PATH_TO_IMAGES = os.path.join(CWD_PATH, IM_DIR)
    txt_path = os.path.join(os.getcwd(), 'detection-results')
    images = glob.glob(PATH_TO_IMAGES + '/*.jpg')

    MODEL_NAME = 'test.tflite'
    MODEL_PATH = os.path.join(CWD_PATH, MODEL_NAME)


    # param parse
    anchors = np.array([1.51, 2.43, 1.83, 1.54, 2.26, 3.08, 2.59, 2.07, 3.75, 3.41]).reshape(-1, 2)
    class_names = ["black+black+red", "black+black+silver", "black+white+red", "black+white+silver", "white+black+red", "white+black+silver", "white+white+red", "white+white+silver"]


    for image_path in images:
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imH, imW, _ = image.shape 
        image_id = image_path[38:-4]
        print(image_id)
        
        boxes, classes, scores = validate_yolo_model_tflite(MODEL_PATH, image, anchors, class_names,
                                                            args.loop_count)

        # print('boxes, classes, scores', boxes, classes, scores)
        
        
        
        if boxes == []:
            print('Error!!!')
            with open(txt_path + "/noresult.txt", "a") as new_f:
                new_f.write("%s \n" % (image_id))
        else:
            for i in range(boxes.shape[0]):
                box = boxes[i]
                prob = scores[i]
                # print(prob)
                left = box[0]  # xmin
                bottom = box[1]  # ymin
                right = box[2]  # xmax
                top = box[3]  # ymax
                obj_name = class_names[classes[i]]
                with open(txt_path + '\\' + image_id + ".txt", "a") as new_f:
                    new_f.write("%s %s %s %s %s %s\n" % (obj_name, prob, left, bottom, right, top))





if __name__ == '__main__':
    main()
