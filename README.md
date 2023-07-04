# segmentation_code_for_mmdetection

mmdetection mask-rcnn学習用コード  
train.py 学習用コード  
demo.py 推論用コード  
output　weightやcheckpointファイルのディレクトリ  
laboro_big　データセット  

コード解説
```train.py
register_coco_instances("tomato", {}, "/home/suke/toms_ws/instanse_seg_tools/detectron_tool/laboro_big/train/train.json", "/home/suke/toms_ws/instanse_seg_tools/detectron_tool/laboro_big/train/")
metadata = MetadataCatalog.get("tomato")
dataset_dicts = DatasetCatalog.get("tomato")
```

register_coco_instances　: COCO形式の データセットを Detectron2 のデータ形式に登録
引数は、(name, metadata, json_file, image_root)

MetadataCatalog.get(“train”) : 先ほど登録したデータから、metadata を作成します。
                              　metadata は、評価や可視化においてラベルやカラーの紐付けに使われる
DatasetCatalog.get(“train”) : ラベルの取得
                              0=__background__, 1=finger のように、推論結果とクラスラベルを紐づける辞書を取得


```
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
```
cfg.merge_from_file : 使用するインスタンスセグメンテーションの設定ファイルを読み込み
cfg.MODEL.WEIGHTS : 設定ファイルからネットワークの重みを読み込み,転移学習に使う

```
cfg.DATASETS.TRAIN = ('train',)
cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
```
cfg.DATASETS.TRAIN = (‘train’,) : “train” データセットを学習に使用するために cfg へ読み込まる
cfg.DATASETS.TEST : 検証用のデータセットを指定
cfg.DATALOADER.NUM_WORKERS は、データローダーの数を指定
