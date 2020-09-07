#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import cv2
import os, sys, argparse
from tensorflow.lite.python import interpreter as interpreter_wrapper
from common.data_utils import preprocess_image
import threading as Thread
import numpy as np
from backend.box import to_minmax
from backend.decoder import YoloDecoder


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
    if boxes != []:
        boxes = _to_original_scale(image_shape, boxes)
    classes, scores = _probs_to_classes_scores(probs)
    return boxes, classes, scores



def main():
    class VideoStream:
        """Camera object that controls video streaming from the Picamera"""

        def __init__(self, resolution=(640, 480), framerate=30):
            # Initialize the PiCamera and the camera image stream
            self.stream = cv2.VideoCapture(0)
            ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            ret = self.stream.set(3, resolution[0])
            ret = self.stream.set(4, resolution[1])

            # Read first frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

            # Variable to control when the camera is stopped
            self.stopped = False

        def start(self):
            # Start the thread that reads frames from the video stream
            Thread(target=self.update, args=()).start()
            return self

        def update(self):
            # Keep looping indefinitely until the thread is stopped
            while True:
                # If the camera is stopped, stop the thread
                if self.stopped:
                    # Close camera resources
                    self.stream.release()
                    return

                # Otherwise, grab the next frame from the stream
                (self.grabbed, self.frame) = self.stream.read()

        def read(self):
            # Return the most recent frame
            return self.frame

        def stop(self):
            # Indicate that the camera and thread should be stopped
            self.stopped = True


    parser = argparse.ArgumentParser(description='validate YOLO model (h5/pb/tflite/mnn) with image')
    parser.add_argument('--loop_count', help='loop inference for certain times', type=int, default=1)
    parser.add_argument('--custom_objects', required=False, type=str, help="Custom objects in keras model (swish/tf). Separated with comma if more than one.", default=None)

    args = parser.parse_args()
    CWD_PATH = os.getcwd()
    MODEL_NAME = 'test.tflite'
    MODEL_PATH = os.path.join(CWD_PATH, MODEL_NAME)

    # param parse
    anchors = np.array([1.51, 2.43, 1.83, 1.54, 2.26, 3.08, 2.59, 2.07, 3.75, 3.41]).reshape(-1, 2)
    class_names = ["black+black+red", "black+black+silver", "black+white+red", "black+white+silver", "white+black+red", "white+black+silver", "white+white+red", "white+white+silver"]

    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    # Initialize video stream
    videostream = VideoStream(resolution=(500, 500), framerate=30).start()
    time.sleep(1)

    # for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
    while True:

        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()

        # Grab frame from video stream
        frame1 = videostream.read()

        # Acquire frame and resize to expected shape [1xHxWx3]
        frame = frame1.copy()

        boxes, classes, scores = validate_yolo_model_tflite(MODEL_PATH, frame, anchors, class_names,
                                                            args.loop_count)

        print('boxes, classes, scores', boxes, classes, scores)

        if boxes == []:
            break

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > 0.75) and (scores[i] <= 1.0)):
                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                xmin = int(max(1, boxes[i][0]))
                ymin = int(max(1, boxes[i][1]))
                xmax = int(min(imH, boxes[i][2]))
                ymax = int(min(imW, boxes[i][3]))

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

                # Draw label
                object_name = class_names[int(classes[i])]  # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i] * 100))  # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
                label_ymin = max(ymin, labelSize[1] + 10)  # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                              (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255),
                              cv2.FILLED)  # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),
                            2)  # Draw label text

        # Draw framerate in corner of frame
        cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0),
                    2, cv2.LINE_AA)

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)

        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2 - t1) / freq
        frame_rate_calc = 1 / time1

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    # Clean up
    cv2.destroyAllWindows()
    videostream.stop()

if __name__ == '__main__':
    main()
