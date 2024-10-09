# Sketch 이미지 데이터 분류

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

Sketch이미지 분류 경진대회는 주어진 데이터를 활용하여 모델을 제작하고 어떤 객체를 나타내는지 분류하는 대회입니다.

Computer Vision에서는 다양한 형태의 이미지 데이터가 활용되고 있습니다. 이 중, 비정형 데이터의 정확한 인식과 분류는 여전히 해결해야 할 주요 과제로 자리잡고 있습니다. 특히 사진과 같은 일반 이미지 데이터에 기반하여 발전을 이루어나아가고 있습니다.

하지만 일상의 사진과 다르게 스케치는 인간의 상상력과 개념 이해를 반영하는 추상적이고 단순화된 형태의 이미지입니다. 이러한 스케치 데이터는 색상, 질감, 세부적인 형태가 비교적 결여되어 있으며, 대신에 기본적인 형태와 구조에 초점을 맞춥니다. 이는 스케치가 실제 객체의 본질적 특징을 간결하게 표현하는데에 중점을 두고 있다는 점을 보여줍니다.

이러한 스케치 데이터의 특성을 이해하고 스케치 이미지를 통해 모델이 객체의 기본적인 형태와 구조를 학습하고 인식하도록 함으로써, 일반적인 이미지 데이터와의 차이점을 이해하고 또 다른 관점에 대한 모델 개발 역량을 높이는데에 초점을 두었습니다. 이를 통해 실제 세계의 복잡하고 다양한 이미지 데이터에 대한 창의적인 접근방법과 처리 능력을 높일 수 있습니다. 또한, 스케치 데이터를 활용하는 인공지능 모델은 디지털 예술, 게임 개발, 교육 콘텐츠 생성 등 다양한 분야에서 응용될 수 있습니다.
<br />

## 📅 프로젝트 일정

프로젝트 전체 일정

- 2024.09.10 (화) 10:00 ~ 2024.09.26 (목) 17:00

프로젝트 세부 일정
![image](https://github.com/user-attachments/assets/1e019978-bc90-4fd2-821a-2f809b713e8c)

## 💻 개발 환경

```bash
- Language : Python
- Environment
  - CPU : Intel(R) Xeon(R) Gold 5120
  - GPU : Tesla V100-SXM2 32GB × 1
- Framework : PyTorch
- Collaborative Tool : Git, Wandb, Notion
```

## 🏆 프로젝트 결과 (수정)

- Public 1등 예정, Private 1등 예정

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

- **\*\*\*\***수정\***\*\*\*\*\*\*\***
- 학습에 사용할 이미지 데이터는 15,021개로 data/train/ 아래에 각 객체별 폴더로 구분되어 있습니다.
- 제공되는 이미지는 주로 사람의 손으로 그려진 드로잉이나 스케치로 구성되어 있습니다.
- train.csv와 test.csv에는 각 이미지별 폴더명(class_name), 이미지 경로(image_path), 예측해야할 class(target)에 대한 정보가 포함되어 있습니다.

<br />

## 📁 프로젝트 구조

```
📦level1-imageclassification-cv-05
 ┣ 📂.github
 ┃ ┗ 📜.keep
 ┣ 📂data
 ┃ ┣ 📜.DS_Store
 ┃ ┣ 📜._DS_Store
 ┃ ┣ 📜._sample_submission.csv
 ┃ ┣ 📜._test.csv
 ┃ ┣ 📜._train.csv
 ┃ ┣ 📜sample_submission.csv
 ┃ ┣ 📜test.csv
 ┃ ┗ 📜train.csv
 ┣ 📂model_checkpoints
 ┣ 📂training_logs
 ┃ ┗ 📜training_log.txt
 ┣ 📜.gitignore
 ┣ 📜augmentation.py
 ┣ 📜augmentation_list.txt
 ┣ 📜dataset.py
 ┣ 📜inference.py
 ┣ 📜main.py
 ┣ 📜model.py
 ┣ 📜README.md
 ┣ 📜requirements.txt
 ┣ 📜seed.py
 ┣ 📜timm_list.txt
 ┗ 📜train.py
```

#### 1) `train.py`

- 모델 학습을 수행하는 함수로, 학습과 검증 루프를 포함하여 조기 종료와 체크포인트 저장 기능이 구현된 파일
- wandb 로깅, 학습 손실 계산, 검증, 모델 저장, 그리고 최적의 모델 선택 및 조기 종료 로직 포함

#### 2) `seed.py`

- 모든 랜덤 연산에서 동일한 결과를 재현할 수 있도록 시드를 설정하는 파일
- random, numpy, torch 라이브러리와 관련된 시드 설정 및 CUDA 관련 고정 설정

#### 3) `model.py`

- ConvNext 모델을 정의한 파일로, timm 라이브러리를 사용하여 미리 학습된 모델을 로드
- 입력된 데이터를 모델에 전달하여 예측을 수행하는 forward 메서드 포함

#### 4) `main.py`

- 학습과 추론을 위한 메인 스크립트로, argparse를 통해 설정 값을 받아 모델 학습과 추론을 수행
- 데이터셋 로드, 학습/검증 루프, 체크포인트 로드 및 저장, 추론 후 결과 파일 생성

#### 5) `inference.py`

- 모델을 사용해 테스트 데이터를 추론하는 함수와 가장 최근의 체크포인트 파일을 가져오는 함수 정의
- inference 함수는 예측값을 반환하고, get_latest_checkpoint 함수는 체크포인트 디렉토리에서 가장 최근 파일을 선택

#### 6) `dataset.py`

- 학습 및 추론 데이터를 로드하는 CustomDataset 클래스를 정의한 파일
- 이미지 데이터를 로드하고, 주어진 변환(transform)을 적용하여 반환하며, 학습 또는 추론 모드에 따라 라벨과 함께 데이터를 반환

#### 7) `augmentation.py`

- 이미지 데이터에 다양한 데이터 증강 기법을 적용하는 SketchAutoAugment 클래스 정의
- 회전, 포스터화, 전치, 색상 반전 등 여러 Augmentation 정책을 랜덤으로 적용하여 이미지 변환

<br />

## ⚙️ requirements

- pandas==2.1.4
- matplotlib==3.8.4
- seaborn==0.13.2
- Pillow==10.3.0
- numpy==1.26.3
- timm==0.9.16
- albumentations==1.4.4
- tqdm==4.66.1
- scikit-learn==1.4.2
- opencv-python==4.9.0.80
- wandb==0.18.0

`pip install -r requirements.txt`

<br />

## ▶️ 실행 방법

#### dataset

`wget https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000307/data/data.tar.gz`

#### 학습 및 체크포인트 저장

`python main.py --train_dir ../data/train --train_csv ../data/train.csv --test_dir ../data/test --test_csv ../data/test.csv --batch_size 16 --resize_height 448 --resize_width 448 --learning_rate 1e-4 --max_epochs 50`

#### 체크포인트에서 학습 재개

`python main.py --train_dir ../data/train --train_csv ../data/train.csv --test_dir ../data/test --test_csv ../data/test.csv --resume_training --batch_size 16 --resize_height 448 --resize_width 448`

#### `argparse` 인자 설명

<details>
<summary>클릭해서 펼치기/접기</summary>

1. **`--train_dir` (필수 인자)**:
   - **설명**: 학습 데이터가 저장된 디렉토리 경로를 설정합니다.
   - **예시**: `--train_dir ../data/train`
2. **`--train_csv` (필수 인자)**:

   - **설명**: 학습 데이터의 이미지 경로와 레이블이 포함된 CSV 파일 경로를 설정합니다.
   - **예시**: `--train_csv ../data/train.csv`

3. **`--test_dir` (필수 인자)**:

   - **설명**: 테스트 데이터가 저장된 디렉토리 경로를 설정합니다.
   - **예시**: `--test_dir ../data/test`

4. **`--test_csv` (필수 인자)**:

   - **설명**: 테스트 데이터의 이미지 경로와 ID가 포함된 CSV 파일 경로를 설정합니다.
   - **예시**: `--test_csv ../data/test.csv`

5. **`--save_dir` (선택적 인자, 기본값: `./model_checkpoints`)**:

   - **설명**: 학습된 모델 체크포인트를 저장할 디렉토리 경로를 설정합니다.
   - **예시**: `--save_dir ./checkpoints`

6. **`--log_dir` (선택적 인자, 기본값: `./training_logs`)**:

   - **설명**: 학습 로그를 저장할 디렉토리 경로를 설정합니다.
   - **예시**: `--log_dir ./logs`

7. **`--batch_size` (선택적 인자, 기본값: `32`)**:

   - **설명**: 학습과 추론 시 사용할 배치 크기를 설정합니다.
   - **예시**: `--batch_size 16`

8. **`--learning_rate` (선택적 인자, 기본값: `1e-5`)**:

   - **설명**: 학습 시 사용하는 학습률을 설정합니다.
   - **예시**: `--learning_rate 0.001`

9. **`--weight_decay` (선택적 인자, 기본값: `0.01`)**:

   - **설명**: AdamW 옵티마이저에서 사용하는 가중치 감소값을 설정합니다.
   - **예시**: `--weight_decay 0.001`

10. **`--max_epochs` (선택적 인자, 기본값: `50`)**:

    - **설명**: 학습할 최대 에포크 수를 설정합니다.
    - **예시**: `--max_epochs 100`

11. **`--accumulation_steps` (선택적 인자, 기본값: `8`)**:

    - **설명**: 그래디언트 누적을 위한 스텝 수를 설정합니다.
    - **예시**: `--accumulation_steps 4`

12. **`--patience` (선택적 인자, 기본값: `5`)**:

    - **설명**: 학습 중 조기 종료(Early Stopping)를 위한 patience를 설정합니다. 이 값은 검증 손실이 개선되지 않을 때 몇 번의 에포크를 더 실행할지 결정합니다.
    - **예시**: `--patience 10`

13. **`--resume_training` (선택적 인자)**:

    - **설명**: 가장 최근의 체크포인트에서 학습을 재개할지 여부를 설정합니다. 이 플래그를 추가하면, 학습이 중단된 체크포인트에서 이어서 학습이 가능합니다.
    - **예시**: `--resume_training`

14. **`--resize_height` (선택적 인자, 기본값: `448`)**:

    - **설명**: 이미지 변환 시 이미지의 높이를 설정합니다.
    - **예시**: `--resize_height 512`

15. **`--resize_width` (선택적 인자, 기본값: `448`)**:
    - **설명**: 이미지 변환 시 이미지의 너비를 설정합니다.
    - **예시**: `--resize_width 512`

</details>

<br />

## ✏️ Wrap-Up Report

- [Wrap-Up Report](https://drive.google.com/file/d/1QDnYMq0fmI9uFghMYs0ZgODhgBYgLJEq/view?usp=sharing)
