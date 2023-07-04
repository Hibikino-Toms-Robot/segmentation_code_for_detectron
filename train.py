import os
import cv2
import random
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

# register_coco_instances("train", {}, "./coins/coco-1612779490.2197058.json", "./coins/")
# metadata = MetadataCatalog.get("train")
# dataset_dicts = DatasetCatalog.get("train")

register_coco_instances("tomato", {}, "/home/suke/toms_ws/instanse_seg_tools/detectron_tool/laboro_big/train/train.json", "/home/suke/toms_ws/instanse_seg_tools/detectron_tool/laboro_big/train/")
metadata = MetadataCatalog.get("tomato")
dataset_dicts = DatasetCatalog.get("tomato")

for d in random.sample(dataset_dicts, 1):
  img = cv2.imread(d["file_name"])
  visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
  vis = visualizer.draw_dataset_dict(d)
  cv2.imshow("frame", vis.get_image()[:, :, ::-1])
cv2.waitKey(1000) 
cv2.destroyAllWindows()

cfg = get_cfg()
cfg.OUTPUT_DIR = './output'
cfg.CUDA = 'cuda:0'

cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ('tomato',)
cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 500
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
#cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()