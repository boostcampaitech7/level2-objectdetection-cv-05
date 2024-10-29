# 재활용 품목 분류를 위한 Object Detection

## 🥇 팀 구성원

<div align="center">
<table>
  <tr>
    <td align="center">
      <a href="https://github.com/kimsuckhyun">
        <img src="https://stages.ai/_next/image?url=https%3A%2F%2Faistages-api-public-prod.s3.amazonaws.com%2Fapp%2FUsers%2F00004010%2Fuser_image.png&w=1920&q=75" width="120px" height="120px" alt=""/>
        <hr />
        <sub><b>김석현</b></sub><br />
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/kupulau">
        <img src="https://stages.ai/_next/image?url=https%3A%2F%2Faistages-api-public-prod.s3.amazonaws.com%2Fapp%2FUsers%2F00003808%2Fuser_image.png&w=1920&q=75" width="120px" height="120px" alt=""/>
        <hr />
        <sub><b>황지은</b></sub><br />
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/lexxsh">
        <img src="https://stages.ai/_next/image?url=https%3A%2F%2Faistages-api-public-prod.s3.amazonaws.com%2Fapp%2FUsers%2F00003955%2Fuser_image.png&w=1920&q=75" width="120px" height="120px" alt=""/>
        <hr />
        <sub><b>이상혁</b></sub><br />
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/june21a">
        <img src="https://stages.ai/_next/image?url=https%3A%2F%2Faistages-api-public-prod.s3.amazonaws.com%2Fapp%2FUsers%2F00003793%2Fuser_image.png&w=1920&q=75" width="120px" height="120px" alt=""/>
        <hr />
        <sub><b>박준일</b></sub><br />
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/glasshong">
        <img src="https://stages.ai/_next/image?url=https%3A%2F%2Faistages-api-public-prod.s3.amazonaws.com%2Fapp%2FUsers%2F00004034%2Fuser_image.png&w=1920&q=75" width="120px" height="120px" alt=""/>
        <hr />
        <sub><b>홍유리</b></sub><br />
      </a>
    </td>
  </tr>
</table>
</div>

<br />

## 🗒️ 프로젝트 개요

바야흐로 대량 생산, 대량 소비의 시대. 우리는 많은 물건이 대량으로 생산되고, 소비되는 시대를 살고 있습니다. 하지만 이러한 문화는 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제를 낳고 있습니다.

분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나입니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문입니다.

따라서 우리는 사진에서 쓰레기를 Detection 하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. 문제 해결을 위한 데이터셋으로는 일반 쓰레기, 플라스틱, 종이, 유리 등 10 종류의 쓰레기가 찍힌 사진 데이터셋이 제공됩니다.

여러분에 의해 만들어진 우수한 성능의 모델은 쓰레기장에 설치되어 정확한 분리수거를 돕거나, 어린아이들의 분리수거 교육 등에 사용될 수 있을 것입니다. 부디 지구를 위기로부터 구해주세요! 🌎


<br />

## 📅 프로젝트 일정

프로젝트 전체 일정

- 2024.09.30 (월) 10:00 ~ 2024.10.24 (목) 19:00

![image](https://github.com/user-attachments/assets/e6d03619-fe9b-4b14-8266-e169c765f9a0)


## 💻 개발 환경

```bash
- Language : Python
- Environment
  - CPU : Intel(R) Xeon(R) Gold 5120
  - GPU : Tesla V100-SXM2 32GB × 1
- Framework : PyTorch
- Collaborative Tool : Git, Wandb, Notion
```

## 🏆 프로젝트 결과

- Public 1등, Private 1등

  ![image](https://github.com/user-attachments/assets/4956fa94-51b7-498a-b8c8-4cc7dd8cea33)

## ✏️ Wrap-Up Report

- 프로젝트의 전반적인 내용은 아래 랩업 리포트를 참고 바랍니다.
- [Wrap-Up Report](https://drive.google.com/file/d/13dfWdaCJQfc2CzF-bT4asWYKytZWTk9m/view?usp=sharing)

## 📁 데이터셋 구조

```
📦data
 ┣ 📜train.json
 ┣ 📜test.json
 ┃
 ┣ 📂test
 ┃ ┣ 📜0000.JPG
 ┃ ┣ 📜0001.JPG
 ┃ ┣ 📜0002.JPG
 ┃ ┗ ...
 ┣ 📂train
 ┃ ┣ 📜0000.JPG
 ┃ ┣ 📜0001.JPG
 ┃ ┣ 📜0002.JPG
 ┃ ┗ ...
```

- 학습에 사용할 이미지는 4,883개, 추론에 사용할 이미지는 4,871개로 각각 data/train/, data/test 아래에 저장되어 있습니다.
- 제공되는 이미지 데이터셋은 10개 클래스의 쓰레기가 찍힌 1024 x 1024 크기의 사진으로 구성되어 있습니다.
- train.json과 test.json은 coco format으로 된 각 이미지에 대한 annotation file 입니다.

<br />

## 📁 프로젝트 구조

```
📦level2-objectdetection-cv-05
 ┣ 📂.github
 ┃ ┗ 📜.keep
 ┣ 📂EDA
 ┃ ┣ 📜EDA.ipynb
 ┃ ┣ 📜heatmap.png
 ┣ 📂mmdetection
 ┃ ┣ 📜Co-detr
 ┃ ┃ ┗ 📜co-detrl.py
 ┃ ┣ 📜DiNO
 ┃ ┃ ┗ 📜dino_Swin_L_baseline.py
 ┃ ┣ 📜_base_
 ┃ ┃ ┗ 📜default_dataset.py
 ┃ ┃ ┗ 📜default_multi_dataset.py
 ┃ ┃ ┗ 📜simple_augmentation_dataset.py
 ┃ ┃ ┗ 📜heavy_augmentation_dataset.py
 ┃ ┃ ┗ 📜lsj_mosaic_augmentation_dataset.py
 ┃ ┃ ┗ 📜default_runtime.py
 ┃ ┃ ┗ 📜default_tta.py
 ┃ ┣ 📜cascade-RCNN
 ┃ ┃ ┗ 📜cascade-rcbb_convnextV2.py
 ┃ ┃ ┗ 📜cascade-rcnn_swin_L.py
 ┃ ┣ 📜ddq
 ┃ ┃ ┗ 📜ddq_swinl.py
 ┣ 📂utils
 ┃ ┣ 📜SR_X2_preprocessing
 ┃ ┃ ┗ 📜4_crop.ipynb
 ┃ ┃ ┗ 📜5_random.ipynb
 ┃ ┃ ┗ 📜Update_json.ipynb
 ┃ ┣ 📜mmdetection
 ┃ ┃ ┗ 📜inference.py
 ┃ ┃ ┗ 📜inference_mmdet_v2.py
 ┃ ┃ ┗ 📜train.py
 ┃ ┣ 📜MultiLabelStratifiedKFold_COCO.py
 ┃ ┣ 📜Pascal_to_coco.ipynb
 ┃ ┣ 📜StratifiedKFold_COCO.py
 ┃ ┣ 📜emsemble.py
 ┗ ┗ 📜inference_visualizer.py

```
<br />

## 🧱 Structure

![image](https://github.com/user-attachments/assets/b2e1d2b4-822e-4a39-86b1-97319114f8c8)

- CO-DETR + DDQ-DETR + Cascade R-CNN + YOLO(emsemble) 모델 사용



</details>

<br />
