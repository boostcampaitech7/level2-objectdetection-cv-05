import os
import pandas as pd
import numpy as np
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

def main():
    submission_files = [
        '/data/ephemeral/home/mmdetection_3/emsmeble/submission_noTTA.csv',
        '/data/ephemeral/home/mmdetection_3/emsmeble/submission_test.csv',
    ]

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
            
            if boxes:  # Only add non-empty predictions
                boxes_list.append(boxes)
                scores_list.append(scores)
                labels_list.append(labels)
        
        if boxes_list:
            boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, iou_thr=0.5, conf_type='box_and_model_avg')
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