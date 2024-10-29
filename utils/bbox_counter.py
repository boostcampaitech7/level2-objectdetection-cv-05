import pandas as pd
import numpy as np
import glob
import os
from tqdm import tqdm

def calculate_bbox_stats(df):
    bbox_counts = []
    for pred_str in tqdm(df['PredictionString'], desc="Counting bboxes"):
        # confidence 값의 개수를 세어 bbox 개수 계산
        bbox_count = len(str(pred_str).split(' ')[1::6])
        bbox_counts.append(bbox_count)
    
    return {
        'max_bbox': max(bbox_counts),
        'avg_bbox': np.mean(bbox_counts),
        'min_bbox': min(bbox_counts),
        'median_bbox': np.median(bbox_counts),
        'total_bbox': sum(bbox_counts),
        'bbox_counts': bbox_counts
    }

def main():
    # CSV 파일이 있는 디렉토리 경로
    csv_dir = '/data/ephemeral/home/emsemble_data/result'
    
    # 디렉토리 내의 모든 CSV 파일 찾기
    csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))
    
    stats_list = []
    
    for csv_file in csv_files:
        print(f"\nProcessing: {csv_file}")
        
        # CSV 파일 읽기
        df = pd.read_csv(csv_file)
        
        # bbox 통계 계산
        stats = calculate_bbox_stats(df)
        stats['file_name'] = os.path.basename(csv_file)
        stats['image_count'] = len(df)
        stats_list.append(stats)
        
        print(f"File: {stats['file_name']}")
        print(f"Total images: {stats['image_count']}")
        print(f"Total bounding boxes: {stats['total_bbox']}")
        print(f"Max bbox per image: {stats['max_bbox']}")
        print(f"Avg bbox per image: {stats['avg_bbox']:.2f}")
        print(f"Median bbox per image: {stats['median_bbox']}")
        print(f"Min bbox per image: {stats['min_bbox']}")
        
        # bbox 개수의 분포 출력
        unique, counts = np.unique(stats['bbox_counts'], return_counts=True)
        print("\nBounding box count distribution:")
        for u, c in zip(unique, counts):
            print(f"  {u} bbox: {c} images")

if __name__ == '__main__':
    main()