import os
import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from ensemble_boxes import weighted_boxes_fusion


def process_prediction_string(predict_string, img_width, img_height):
    predict_list = str(predict_string).split()
    if len(predict_list) == 0:
        return [], [], []
    
    predict_list = np.array(predict_list, dtype=float).reshape(-1, 6)
    boxes = [[float(x) / img_width, float(y) / img_height, float(w) / img_width, float(h) / img_height] 
             for x, y, w, h in predict_list[:, 2:6]]
    scores = predict_list[:, 1].tolist()
    labels = predict_list[:, 0].astype(int).tolist()
    
    return boxes, scores, labels

def normalize_manual(scores, min_val=0.5, max_val=1.0):
    """
    수동으로 구현한 min-max 정규화
    
    Args:
        scores: 정규화할 점수 리스트
        min_val: 정규화 후 최소값
        max_val: 정규화 후 최대값
    """
    min_score = min(scores)
    max_score = max(scores)
    normalized = [(score - min_score) / (max_score - min_score) for score in scores]
    
    # min_val ~ max_val 범위로 조정
    adjusted = [min_val + (max_val - min_val) * w for w in normalized]
    return adjusted

def normalize_sklearn(scores, min_val=0.5, max_val=1.0):
    """
    sklearn의 MinMaxScaler를 사용한 정규화
    
    Args:
        scores: 정규화할 점수 리스트
        min_val: 정규화 후 최소값
        max_val: 정규화 후 최대값
    """
    scaler = MinMaxScaler(feature_range=(min_val, max_val))
    return scaler.fit_transform(np.array(scores).reshape(-1, 1)).ravel()


def main():

    parser = argparse.ArgumentParser(description='Weight Normalization for WBF Ensemble')
    
    # 정규화 방법 선택
    parser.add_argument('--method', type=str, choices=['manual', 'sklearn'], default='manual',
                        help='Normalization method to use (manual or sklearn)')
    
    submission_files = [
        '/data/ephemeral/home/mmdetection_3/emsmeble/submission_noTTA.csv',
        '/data/ephemeral/home/mmdetection_3/emsmeble/submission_test.csv',
         '/data/ephemeral/home/mmdetection_3/emsmeble/submission_test.csv',
          '/data/ephemeral/home/mmdetection_3/emsmeble/submission_test.csv',
    ]
    
    ## 가중치 설정하는거 추가하기
    ## min_max = 정규화하는 코드 수정하기
    ## skip_connection 수정하는 부분 추가하기
    
    ## 가중치 수정하는 부분 _ default 값 = 1,1,1,1
    weights = [1,1,1,1]
    ## 박스 스킵 임계값 수정
    skip_box = 0.1
    ## 신뢰도 점수 계산 방식
    conf_type = 'box_and_model_avg'
    ## bbox 오버플로우 허융
    allows_overflow = False



    ## 각 모델의 LB 점수에 따른 가중치 조정
    submissions_score = []
    args = parser.parse_args()

    if args.method == 'manual':
        normalized_weights = normalize_manual(submissions_score, 0.5, 1.0)
    else:
        normalized_weights = normalize_sklearn(submissions_score, 0.5, 1.0)


    submission_df = [pd.read_csv(f) for f in submission_files]
    image_ids = submission_df[0]['image_id'].tolist()
    
    img_width, img_height = 1024, 1024

    prediction_strings = []
    file_names = []

    for image_id in tqdm(image_ids, total=len(image_ids)):
        boxes_list = []
        scores_list = []
        labels_list = []
        
        for df in submission_df:
            predict_string = df.loc[df['image_id'] == image_id, 'PredictionString'].iloc[0]
            boxes, scores, labels = process_prediction_string(predict_string, img_width, img_height)
            
            if boxes:
                boxes_list.append(boxes)
                scores_list.append(scores)
                labels_list.append(labels)
        
        if boxes_list:
            boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, iou_thr=0.5, weights=normalized_weights, skip_box_thr=skip_box, conf_type= conf_type)
            prediction_string = ' '.join([f'{int(l)} {s} {x*img_width} {y*img_height} {w*img_width} {h*img_height}' 
                                          for l, s, (x, y, w, h) in zip(labels, scores, boxes)])
        else:
            prediction_string = ''

        prediction_strings.append(prediction_string)
        file_names.append(image_id)

    submission = pd.DataFrame({'PredictionString': prediction_strings, 'image_id': file_names})
    submission.to_csv('/data/ephemeral/home/submission2.csv', index=False)

if __name__ == '__main__':
    main()