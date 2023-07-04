import os
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode
from detectron2 import model_zoo
import glob
import time

cfg = get_cfg()
cfg.OUTPUT_DIR = './output'
cfg.CUDA = 'cuda:0'


# register_coco_instances("train", {}, "./coins/coco-1612779490.2197058.json", "./coins/")
# metadata = MetadataCatalog.get("train")
# dataset_dicts = DatasetCatalog.get("train")

register_coco_instances("tomato", {}, "/home/suke/toms_ws/instanse_seg_tools/detectron_tool/laboro_big/train/train.json", "/home/suke/toms_ws/instanse_seg_tools/detectron_tool/laboro_big/train/")
metadata = MetadataCatalog.get("tomato")
dataset_dicts = DatasetCatalog.get("tomato")

cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
cfg.MODEL.WEIGHTS = "/home/suke/toms_ws/instanse_seg_tools/detectron_tool/output/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor = DefaultPredictor(cfg)

# inference
#img=cv2.imread("/home/suke/toms_ws/instanse_seg_tools/detectron_tool/coins/IMG_4661.jpg")
#img=cv2.imread("/home/suke/toms_ws/instanse_seg_tools/detectron_tool/laboro_big/train/IMG_0983.jpg")
img=cv2.imread("/home/suke/toms_ws/instanse_seg_tools/detectron_tool/laboro_big/train/IMG_20191215_110730.jpg")
outputs = predictor(img)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
v = Visualizer(img[:, :, ::-1],
                metadata=metadata, 
                scale=1 
                #instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
img = out.get_image()[:, :, ::-1]
height = int(img.shape[0]/3)
width = int(img.shape[1]/3)
img = cv2.resize(img, (width, height))
cv2.imshow("frame", img)
cv2.waitKey(0)
cv2.destroyAllWindows()