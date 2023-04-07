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


# train registry
from detectron2.data.datasets import register_coco_instances
register_coco_instances("zebrafish_train", {}, "/home/gongching/Downloads/detectron2/zebrafish_archives/new_version/zebrafish_train.json",
                        "/home/gongching/Downloads/detectron2/zebrafish_archives/new_version/train_images")
register_coco_instances("zebrafish_val", {}, "/home/gongching/Downloads/detectron2/zebrafish_archives/new_version/zebrafish_test.json",
                        "/home/gongching/Downloads/detectron2/zebrafish_archives/new_version/test_images")

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
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.OUTPUT_DIR = "zebrafish1211/OUTPUTS"

#cfg.DATASETS.TRAIN/TEST should be a tuple instead of string, use ("my_dataset_train",) instead of ("my_dataset_train")
cfg.DATASETS.TRAIN = ("zebrafish_train", )
cfg.DATASETS.TEST = ("zebrafish_val", )

# cfg.MODEL.DEVICE='cpu' # use cpu to train

cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
# RESUME FROM LAST CHECKPOINT
# cfg.MODEL.WEIGHTS = "/home/gongching/Downloads/detectron2/toy_data/custom/OUTPUTS/model_final.pth"
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
# cfg.SOLVER.WARMUP_ITERS = 200 # after 200 iters LR reach 0.001
cfg.SOLVER.MAX_ITER = 2000
# cfg.SOLVER.STEPS = []
# cfg.SOLVER.GAMMA = 0.05

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 16

cfg.TEST.EVAL_PERIOD = 100
#
# make sure the model validates against our validation set
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.events import CommonMetricPrinter
# from TensorboardWriter import CustomTensorboardXWriter
from detectron2.utils.events import JSONWriter
# from ValHookBase import ValLossHook
from detectron2.engine.hooks import PeriodicWriter
from detectron2.utils.marWriter import CustomTensorboardXWriter
from detectron2.utils.mar_ValHook import ValLossHook
from ValBase import ValidationLoss

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

    def build_writers(self):
        """
        Overwrite default writers to contain our custom TensorBoard writer

        Returns
            list[EventWriter]: a list of : class: 'EventWriter objects
        """

        return [
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            CustomTensorboardXWriter(self.cfg.OUTPUT_DIR)
        ]

# Train
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = Trainer(cfg)

trainer.register_hooks(
    [ValidationLoss(cfg)]
)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]

trainer.resume_or_load(resume=False)
trainer.train()