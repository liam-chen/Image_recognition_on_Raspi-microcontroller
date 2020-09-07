import numpy as np
import cv2

def draw_boxes(image, boxes, probs, labels):
    for box, score, name in zip(boxes, probs, labels):
        print(box)
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)
        # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

        # cv2.putText(image, 
                    # '{}:  {:.2f}'.format(name, classes), 
                    # (x1, y1 - 13), 
                    # cv2.FONT_HERSHEY_SIMPLEX, 
                    # 1e-3 * image.shape[0], 
                    # (0,255,0), 2)
        # label = '%s: %d%%' % (name, int(score * 100))  # Example: 'person: 72%'
        # labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
        # label_ymin = max(y1, labelSize[1] + 10)  # Make sure not to draw label too close to top of window
        # cv2.rectangle(frame, (x1, label_y1 - labelSize[1] - 10),
                      # (x1 + labelSize[0], label_y1 + baseLine - 10), (255, 255, 255),
                      # cv2.FILLED)  # Draw white box to put label text in
        # cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),
                    # 2)  # Draw label text
        cv2.imshow('ss',image)
    return image 
    
image_path = 'IMG_20200224_141407.jpg'
image = cv2.imread(image_path)
boxes = [[105, 149, 266, 323]]
probs = [0.11440091]
labels = ['black+white+silver']

img = draw_boxes(image, boxes, probs, labels)
cv2.imshow('ss',img)