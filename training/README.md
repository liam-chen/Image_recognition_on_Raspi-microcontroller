# keras-yolo-training
This tutorial can be completed in Win. train a yolo model, and convert it into a tflite format

## Parameters
Modify the network type, label, and other parameters in [configs.json]. Pay attention to the name of the folder where the image (train_img) and the annotation (train_ann) are stored

## Training 
```
python train.py -c configs.json
``` 
When training finished, a folder named by time will appear, the tflite file inside is the trained model.

## Prediciton
```
python predict.py -c configs.json
```
print all prediciton results in folder "detection-results". The variable "threshold" in file "predict.py" can be modified. "detection-results" is used in mAP calculation

## xml to txt
```
python xml_to_txt.py
```
according to the .xml in folder "val_ann", generate a .txt in folder "ground-truth". "ground-truth" is used in mAP calculation. 