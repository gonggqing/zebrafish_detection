"""
Tensor mask training script

For zebrafish demo

tensormask copyright by facebookresearch

detectron2 version 4.0
"""
import os
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import cv2
import random
import torch

from detectron2 import model_zoo
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog

from tensormask import add_tensormask_config

from detectron2.data.datasets import register_coco_instances
# data registry
register_coco_instances("zebrafish_train", {}, "/home/gongching/Downloads/detectron2/zebrafish_archives/zebrafish_train.json",
                        "/home/gongching/Downloads/detectron2/zebrafish_archives/train_images")
register_coco_instances("zebrafish_test", {}, "/home/gongching/Downloads/detectron2/zebrafish_archives/zebrafish_test.json",
                        "/home/gongching/Downloads/detectron2/zebrafish_archives/test_images")

tensor_train_metadata = MetadataCatalog.get("zebrafish_train")
tensor_dicts = DatasetCatalog.get("zebrafish_train")

# Visualize train data
# import random
# from detectron2.utils.visualizer import Visualizer
# for t in random.sample(tensor_dicts, 2):
#     img = cv2.imread(t["file_name"])
#     visualizer = Visualizer(img[:,:,::-1], metadata=tensor_train_metadata, scale=0.5)
#     vis = visualizer.draw_dataset_dict(t)
#     cv2.imshow('tensor', vis.get_image()[:, :, ::-1])
#     cv2.waitKey(0)

# set tensor mask configs

cfg = get_cfg()
add_tensormask_config(cfg)
cfg.merge_from_file("/home/gongching/Downloads/detectron2/configs/COCO-tensormask/tensormask_R_50_FPN_6x.yaml")
cfg.merge_from_file(model_zoo.get_config_file("COCO-tensormask/tensormask_R_50_FPN_6x.yaml"))

cfg.DATASETS.TRAIN = ("zebrafish_train", )
cfg.DATASETS.TEST = ("zebrafish_test", )

cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.TENSOR_MASK.NUM_CLASSES = 16
cfg.OUTPUT_DIR = "zebrafish1128_tensormask_r_50_6x/OUTPUTS"

cfg.MODEL.WEIGHTS = ("/home/gongching/Downloads/detectron2/projects/TensorMask/model/model_final_R50_6x.pkl")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025

cfg.SOLVER.WARMUP_ITERS = 300
cfg.SOLVER.MAX_ITER = 30000

cfg.TEST.EVAL_PERIOD = 2000

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

# os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# trainer = Trainer(cfg)
# trainer.resume_or_load(resume=True)
# trainer.train()
torch.cuda.empty_cache()

# Inference
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.DATASETS.TEST = ("zebrafish_test", )
cfg.MODEL.TENSOR_MASK.SCORE_THRESH_TEST = 0.5
predictor = DefaultPredictor(cfg)
test_metadata = MetadataCatalog.get("zebrafish_test")

from detectron2.utils.visualizer import ColorMode
import glob
# for m in random.sample(glob.glob("/home/gongching/Downloads/detectron2/zebrafish_archives/test_images/*png"), 5):
test_contents = DatasetCatalog.get('zebrafish_test')
for m in random.sample(test_contents, 6):
    im = cv2.imread(m['file_name'])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=test_metadata,
                   scale=0.8)
    # instances = outputs['instances'].to("cpu")
#     print(type(instances))
#     print(instances)
#     _pred_boxes = instances[0].get('pred_masks')
#     array_mask = _pred_boxes.numpy()
#     array_mask = array_mask + 0
#     shape = array_mask.shape # shape is c,h,w, represent channel, height, weight, respectively
#     mask = np.uint8(array_mask) # using uint8 to map the ndarray, but need to change the value to [0, 255]
    # cv2.imshow('mask', mask[0]*255)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
#     print(type(_pred_boxes))
#     print(len(_pred_boxes))
#     print(type(array_mask))
#     print(array_mask)

    # _pred_classes = instances['pred_classes']
    # _pred_scores = instances['pred_scores']
    # _pred_masks = instances['pred_masks']
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("predict", out.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
print("Visualization finished!")