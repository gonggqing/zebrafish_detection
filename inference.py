''''
This is a Mask R-CNN training workflow for zebrafish images, aiming for identifying multiple  phenotypical features.
Based on the Detectron2 library developed by Meta (Facebook AI Research).
'''
import time

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import common libraries
import numpy as np
import cv2
import random
import os
import torch
import json
import texttable
import glob
import argparse

# import common detectron2 libraries
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.data import transforms as T

# import custom library
from fishutil import *
from fishclass import Zebrafish

##add some input
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_dir", type=str, help="where you store the images")
parser.add_argument("-o", "--output_dir", type=str, help="where you store the inference results")
parser.add_argument("-fc", "--file_csv", type=str, help="the inference results, endwiths .csv")
parser.add_argument("-fj", "--file_json", type=str, help="the inference results raw data, endwiths .json")
parser.add_argument("-t", "--image_type", type=str, help="specify the type of input images, endwiths .png, .jpg, .tiff, etc.")

args = parser.parse_args()
input_path = args.input_dir
output_path = args.output_dir
file_csv = args.file_csv
file_json = args.file_json
image_type = args.image_type

# train registry
from detectron2.data.datasets import register_coco_instances
register_coco_instances("zebrafish_train", {}, "/home/gongching/Downloads/detectron2/zebrafish_archives/zebrafish_train.json",
                        "/home/gongching/Downloads/detectron2/zebrafish_archives/train_images")
register_coco_instances("zebrafish_test", {}, "/home/gongching/Downloads/detectron2/zebrafish_archives/zebrafish_test.json",
                        "/home/gongching/Downloads/detectron2/zebrafish_archives/test_images")

# visualize training data
# my_data_train_metadata = MetadataCatalog.get("zebrafish_train")
# dataset_dicts = DatasetCatalog.get("zebrafish_train")
#
# from detectron2.utils.visualizer import Visualizer
#
# for d in random.sample(dataset_dicts, 3):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=my_data_train_metadata, scale=1)
#     vis = visualizer.draw_dataset_dict(d)
#     cv2.imshow('new', vis.get_image()[:, :, ::-1])
#     cv2.waitKey(0)

torch.backends.cudnn.enable = True
torch.backends.cudnn.benchmark = True

# set detectron2 training configs


cfg = get_cfg()
# # use relative path
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg.OUTPUT_DIR = "zebrafish1128_R_101/OUTPUTS"

#cfg.DATASETS.TRAIN/TEST should be a tuple instead of string, use ("my_dataset_train",) instead of ("my_dataset_train")
cfg.DATASETS.TRAIN = ("zebrafish_train", )
cfg.DATASETS.TEST = ("zebrafish_test", )

# cfg.MODEL.DEVICE='cpu' # use cpu to train

cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
# RESUME FROM LAST CHECKPOINT
# cfg.MODEL.WEIGHTS = "/home/gongching/Downloads/detectron2/toy_data/custom/OUTPUTS/model_final.pth"
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.WARMUP_ITERS = 300 # after 300 iters LR reach 0.001
cfg.SOLVER.MAX_ITER = 30000
cfg.SOLVER.STEPS = []
cfg.SOLVER.GAMMA = 0.05

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 16

cfg.TEST.EVAL_PERIOD = 2000
#
# make sure the model validates against our validation set
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

class CocoTrainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):

        if output_folder is None:
            os.makedirs("coco_eval_zebrafish", exist_ok = True)
            output_folder = "coco_eval_zebrafish"

            return COCOEvaluator(dataset_name, cfg, False, output_folder)
# Train

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = CocoTrainer(cfg)
# trainer.resume_or_load(resume=True)
# trainer.train()
# classes = MetadataCatalog.get("zebrafish_train").thing_classes
# print("Classes: ", classes, len(classes))
# # free gpu memory
# torch.cuda.empty_cache()



# Evaluate
# from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
# from detectron2.evaluation import COCOEvaluator, inference_on_dataset
#
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
# predictor = DefaultPredictor(cfg)
# evaluator = COCOEvaluator("toy_test", cfg, False, output_dir="./output/")
# val_loader = build_detection_test_loader(cfg, "toy_test")
# inference_on_dataset(trainer.model, val_loader, evaluator)
#
# # Inference
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.DATASETS.TEST = ("zebrafish_test", )
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
predictor = DefaultPredictor(cfg)
test_metadata = MetadataCatalog.get("zebrafish_test")

from detectron2.utils.visualizer import ColorMode
#
dataset_test = DatasetCatalog.get('zebrafish_test')
df = create_pd()

try:
    os.makedirs(output_path, exist_ok=True)
    print("%s created successfully."%output_path)
except OSError as error:
    print("directory %s can not be created."%output_path)

# for d in random.sample(dataset_test, 10):
n=0
for d in sorted(glob.glob("%s/*%s" %(input_path,image_type))):
    im = cv2.imread(d)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=test_metadata,
                   scale=0.8,
                   instance_mode=ColorMode.SEGMENTATION)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # cv2.imshow("predict", out.get_image()[:, :, ::-1])
    # # time.sleep(3)
    # cv2.waitKey(0)
    # cv2.destroyWindow("predict")


    cv2.imwrite(os.path.join(output_path, 'fish_%s.png' % str(n)), out.get_image()[:, :, ::-1])
    n += 1
    instances = outputs['instances'].to('cpu')

    zebrafish_instances = split_outputs(instances)

    infos = zebrafish_info(zebrafish_instances) # returns a list, [mask_area, [bounding_box], score]
    print(infos)

    zebrafish = Zebrafish(infos)
    print('------ Establish the zebrafish object succeed, waiting to append to data frame ------')

    zebrafish_endpoints = update_template(zebrafish)

    print("------ Creating the dataframe of zebrafish endpoints ------")
    df = df.append(zebrafish_endpoints, ignore_index=True)

    tb = texttable.Texttable()
    tb.set_cols_align(['c' for i in range(18)])
    tb.header(df.columns)
    tb.add_rows(df.values, header=False)
    tb.set_cols_width([5,5,5,5,5,8,8,8,8,10,10,12,10,11,5,5,5,9])
    print(tb.draw())
    # visualize masks
    # for category in zebrafish_instances:
    #     mask = zebrafish_instances[category].get('mask')
    #     mask = mask*255
    #     cv2.imshow('mask', mask)
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # output to file
    json_infos = json.dumps(infos, indent=4)
    with open("%s/%s" %(output_path, file_json), 'a+') as f:
        f.write(json_infos+'\n')

df.to_csv(os.path.join(output_path, file_csv))

plot(df)
print('Visualization done!')
