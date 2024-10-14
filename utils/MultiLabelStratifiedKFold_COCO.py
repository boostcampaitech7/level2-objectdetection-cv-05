import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from collections import defaultdict
import os
import argparse

def load_coco_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def save_coco_json(data, output_file):
    with open(output_file, 'w') as f:
        json.dump(data, f)
    

def get_image_class_and_bbox_counts(annotations):
    image_class_bbox_counts = defaultdict(lambda: {'class_counts': defaultdict(int), 'bbox_sizes': []})
    
    
    ids = []
    for ann in annotations:
        image_id = ann['image_id']
        category_id = ann['category_id']
        bbox = ann['bbox']  # [x, y, width, height]
        bbox_area = bbox[2] * bbox[3]  # width * height
        
        image_class_bbox_counts[image_id]['class_counts'][category_id] += 1
        image_class_bbox_counts[image_id]['bbox_sizes'].append(bbox_area)
        ids.append(image_id)
    
    ids = set(ids)
    image_info = np.array([
        [max(image_class_bbox_counts[id]['class_counts']),
         np.mean(image_class_bbox_counts[id]['bbox_sizes']),
         len(image_class_bbox_counts[id]['bbox_sizes'])
         ] for id in ids
        ])
    
    image_info[:, 1] = pd.qcut(
        image_info[:, 1], 
        q=10,
        labels=False,
        duplicates='drop'
        )
    
    image_info[:, 2] = pd.qcut(
        image_info[:, 2], 
        q=5,
        labels=False,
        duplicates='drop'
    )
    
    return image_info


def create_stratified_folds(coco_data, n_splits=5):
    var = [(ann['image_id'], ann['category_id']) for ann in coco_data['annotations']]

    X = np.ones((len(coco_data['images']),1))
    y = get_image_class_and_bbox_counts(coco_data['annotations'])
    
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    folds = []
    for fold, (train_index, val_index) in enumerate(mskf.split(X, y), 1):
        train_ids = train_index
        val_ids = val_index
        
        print(f"k-fold {fold} num of train : {len(train_ids)}, num of val : {len(val_ids)}")
        folds.append((train_ids, val_ids))
    
    return folds


def split_coco_data(coco_data, image_ids):
    new_data = {
        'info': coco_data['info'],
        'licenses': coco_data['licenses'],
        'categories': coco_data['categories'],
        'images': [img for img in coco_data['images'] if img['id'] in image_ids],
        'annotations': [ann for ann in coco_data['annotations'] if ann['image_id'] in image_ids]
    }
    return new_data

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', 
                        type=str,
                        default="/data/ephemeral/home/dataset/train.json",
                        help="path to train.json")
    
    parser.add_argument('--save_dir', 
                        type=str,
                        default="/data/ephemeral/home/dataset/kfold_test",
                        help="path where the results json will be saved")
    
    parser.add_argument('--n_splits', 
                        type=int,
                        default=5,
                        help="kfold n_splits")
    
    args = parser.parse_args()
    return args


def main(input_json, output_dir, n_splits=5):
    coco_data = load_coco_json(input_json)
    folds = create_stratified_folds(coco_data, n_splits)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for fold, (train_ids, val_ids) in enumerate(folds, 1):
        train_data = split_coco_data(coco_data, train_ids)
        val_data = split_coco_data(coco_data, val_ids)
        
        save_coco_json(train_data, os.path.join(output_dir, f'train_fold_{fold}.json'))
        save_coco_json(val_data, os.path.join(output_dir, f'val_fold_{fold}.json'))
    
    print(f"완료: {n_splits}개의 폴드로 데이터를 분할하여 {output_dir}에 저장했습니다.")

if __name__ == "__main__":
    # parsing
    args = parse_arguments()
    input_json = args.json_file
    output_dir = args.save_dir
    n_splits = args.n_splits
    
    ##########################
    main(input_json, output_dir, n_splits)