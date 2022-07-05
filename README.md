# Zebrafish Detection
This AI project is a detection task for identifying zebrafish organs and phenotypes in micrographs, which is based on the Meta AI project, [Detectron2 version 4.1.](https://github.com/facebookresearch/detectron2) It mainly used Mask R-CNN for training and validating. It has 16 detected objects, including 8 specific organs and 8 specific abnormal phenotypes. 

![infer_](https://user-images.githubusercontent.com/57084033/177120642-c2a074d5-0c78-4a35-99f8-85f1ae02a80c.gif)

## How to start
First install detectron2 and other dependent libraries (see [zebrafish_maskrcnn.py](https://github.com/gonggqing/zebrafish_detection/blob/ddff5e1871fb63bbb34f46db6785534ed34c017a/zebrafish_maskrcnn.py)). 
Then put the [pre-trained model weights](https://drive.google.com/file/d/1yyREJccnKeRDJ4BOnFMt3FNddC_w4fm_/view?usp=sharing) in your specified path.
```python
# Modify some necessary paths
# images path that pending to detect
path = 'your/specified/path'
...
# save your results
df.to_csv(os.path.join('.../results/', 'results.csv'))
```
Run the [zebrafish_maskrcnn.py](https://github.com/gonggqing/zebrafish_detection/blob/ddff5e1871fb63bbb34f46db6785534ed34c017a/zebrafish_maskrcnn.py) file in `inference mode` locally.
If you want to re-train this model locally, please download the train.json, test.json, train images and test images.(see [images_url.txt](https://github.com/gonggqing/zebrafish_detection/blob/fa6b5911c9373ff5d726fe5b4af44394f8cb81f5/images/images_url.txt)) The json file contains annotations of COCO style. After downloading the images and files, register the coco instances in your code, modify the name and path of the register.
```python
# train registry
from detectron2.data.datasets import register_coco_instances
register_coco_instances("zebrafish_train", {}, "/your/path/train.json",
                        "/your/path/train_images/")
register_coco_instances("zebrafish_train", {}, "/your/path/test.json",
                        "/your/path/test_images/")
```
