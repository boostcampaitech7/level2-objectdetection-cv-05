import os
import cv2
import torch
import pandas as pd
import argparse
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detic.config import add_detic_config

def setup_cfg(config_file, weights_file, num_classes):
    cfg = get_cfg()
    add_detic_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DETR.NUM_CLASSES = num_classes
    cfg.MODEL.WEIGHTS = weights_file
    return cfg

def register_datasets(train_json, test_json, image_root):
    for name, json_file in [("coco_trash_train", train_json), ("coco_trash_test", test_json)]:
        try:
            DatasetCatalog.remove(name)
            MetadataCatalog.remove(name)
        except KeyError:
            pass
        
        try:
            register_coco_instances(name, {}, json_file, image_root)
            print(f"{name} dataset registered")
        except AssertionError:
            print(f"Error registering {name}")
        
        if name == "coco_trash_train":
            MetadataCatalog.get(name).thing_classes = ["General trash", "Paper", "Paper pack", "Metal", 
                                                       "Glass", "Plastic", "Styroform", "Plastic bag", 
                                                       "Battery", "Clothing"]
        MetadataCatalog.get(name).evaluator_type = 'coco'

def predict_and_save(predictor, image_dir, output_csv):
    results = []
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)
        outputs = predictor(image)
        instances = outputs["instances"].to("cpu")
        
        prediction_str = " ".join([f"{cls.item()} {score.item()} {' '.join(map(str, bbox.tolist()))}" 
                                   for cls, score, bbox in zip(instances.pred_classes, instances.scores, instances.pred_boxes)])
        
        results.append({"Prediction": prediction_str, "image_id": f"test/{image_file}"})

    df = pd.DataFrame(results).sort_values(by="image_id")
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

def main(args):
    cfg = setup_cfg(args.config_file, args.weights_file, args.num_classes)
    register_datasets(args.train_json, args.test_json, args.image_root)
    
    cfg.DATASETS.TRAIN = ('coco_trash_train',)
    cfg.DATASETS.TEST = ('coco_trash_test',)
    
    predictor = DefaultPredictor(cfg)
    predict_and_save(predictor, args.image_dir, args.output_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detic object detection")
    parser.add_argument("--config-file", default="/home/sh/bootcamp/project2/Detic/configs/DeformDETR_swinb.yaml", help="Path to config file")
    parser.add_argument("--weights-file", default="/home/sh/bootcamp/project2/Detic/output/Detic_Swin/DeformDETR_swinb/model_final.pth", help="Path to model weights")
    parser.add_argument("--num-classes", type=int, default=10, help="Number of classes")
    parser.add_argument("--train-json", default="/home/sh/bootcamp/project2/dataset/cv_train1.json", help="Path to train JSON file")
    parser.add_argument("--test-json", default="/home/sh/bootcamp/project2/dataset/test.json", help="Path to test JSON file")
    parser.add_argument("--image-root", default="/home/sh/bootcamp/project2/dataset", help="Path to image root directory")
    parser.add_argument("--image-dir", default="/home/sh/bootcamp/project2/dataset/test/", help="Path to test image directory")
    parser.add_argument("--output-csv", default="/home/sh/bootcamp/project2/dataset/predictions.csv", help="Path to output CSV file")
    
    args = parser.parse_args()
    main(args)