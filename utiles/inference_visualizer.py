from mmdet.apis import DetInferencer
import json
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--model',type=str,default='/data/ephemeral/home/mmdetection/Custom_configs/dino.py',help='Config .py file for model')
parser.add_argument('--weights',type=str,default='/data/ephemeral/home/mmdetection/work_dirs/dino_new/epoch_19.pth',help='Trained .pth file')
parser.add_argument('--file',type=str,default='/data/ephemeral/home/dataset/split/val_42_fold_1.json',help='Root for inferencing .json file')

args = parser.parse_args()

inf = DetInferencer(model=args.model,weights=args.weights)


def extract(file):
    with open(file,'r') as f:
        coco=json.load(f)

        images= coco['images']
        list = ['/data/ephemeral/home/dataset/'+img['file_name'] for img in images]

        return list

file=args.file
img=extract(file)


inf(img,out_dir='./output/2/',no_save_vis=False)