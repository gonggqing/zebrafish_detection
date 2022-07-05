# Zebrafish Detection
This AI project is a detection task for identifying zebrafish organs and phenotypes in micrographs, which is based on the Meta AI project, [Detectron2 version 4.1.](https://github.com/facebookresearch/detectron2) It mainly used Mask R-CNN for training and validating. It has 16 detected objects, including 8 specific organs and 8 specific abnormal phenotypes. 

![infer_](https://user-images.githubusercontent.com/57084033/177120642-c2a074d5-0c78-4a35-99f8-85f1ae02a80c.gif)

## How to start
First install detectron2 and other dependent libraries (see [zebrafish_maskrcnn.py](https://github.com/gonggqing/zebrafish_detection/blob/main/zebrafish_maskrcnn.py)).
Put the [pre-trained model weights]() in your specified path.
