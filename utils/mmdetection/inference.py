import sys
import os
import warnings
from copy import deepcopy
import argparse

sys.path.insert(0, '/data/ephemeral/home/mmdetection')
os.chdir("/data/ephemeral/home/mmdetection")


import torch
from torch import nn
import pandas as pd
import numpy as np
from mmengine.config import Config, DictAction
from mmengine import ConfigDict
from mmengine.runner import Runner
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import DATASETS
from tqdm.auto import tqdm as tqdm_auto
from pycocotools.coco import COCO
import cv2

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', 
                        type=str,
                        default="/data/ephemeral/home/june21a/config/cascade-rcnn_convnextV2.py",
                        help="path to config which define tta model and tta pipeline")
    
    parser.add_argument('--checkpoint_path', 
                        type=str,
                        default="/data/ephemeral/home/june21a/cascade-rcnn_convnextv2_baseline/best_coco_bbox_mAP_50_epoch_22.pth",
                        help="path to checkpoint")
    
    parser.add_argument('--work_dir',
                        type=str,
                        default='/data/ephemeral/home/june21a/cascade-rcnn_convnextv2_baseline',
                        help='work_dir path')
    
    parser.add_argument('--use_tta',
                        type=bool,
                        default=True,
                        help='if you want to use tta on inference, set it to True')
    
    parser.add_argument('--batch_size',
                        type=int,
                        default=8)
    
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    config_path = args.config_path
    checkpoint_path = args.checkpoint_path
    work_dir = args.work_dir
    use_tta = args.use_tta
    batch_size = args.batch_size
    
    cfg = Config.fromfile(config_path)
    cfg.work_dir = work_dir
    print("load from : ", checkpoint_path)
    cfg.load_from = checkpoint_path
    
    
    # change data metadata json file path
    cfg.test_dataloader.dataset.ann_file = cfg.test_ann_file_name
    cfg.test_dataloader.batch_size = batch_size
    cfg.test_evaluator.outfile_path = os.path.join(cfg.work_dir, "test_results")
    
    if use_tta:
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
    
    # 모델 구축 및 체크포인트 로드
    runner = Runner.from_cfg(cfg)
    runner.test()
    

if __name__ == "__main__":
    main()