# YOLO Custom Object Detection & Tracking

COCO에 없는 물체를 직접 촬영·라벨링해 YOLO11s로 학습하고, ByteTrack으로 영상 속 물체를 추적한 과제입니다.

## 1. 3클래스 실험 (`3class/`)

- 클래스: film camera, binoculars, bluetooth speaker
- 데이터: 원본 301장, Roboflow 라벨링 및 증강 (Train 630장)
- 모델: YOLO11s, 100 epoch, RTX 4060 8GB Vram 

| Test (30장) | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| 3클래스 | 0.992 | 1.000 | 0.995 | 0.916 |

- 결과 영상: [`3class/result.mp4`](3class/result.mp4)
- 

https://github.com/user-attachments/assets/35ece366-0a1b-46ac-9a90-190b5f7c2f6f


- 모델: [`3class/best.pt`](3class/best.pt)

## 2. 추가 실험: 장난감 총 1클래스 (`toy_gun/`)

손에 쥐고 움직이는 장면, 흔들린 프레임, 비슷하게 생긴 검은 물체(우산, 어두운 옷)가 함께 나오는 어려운 영상에서의 추적을 목표로 했습니다.

- 데이터: 원본 1,042장 (기본 826장 + 우산·옷과 함께 찍은 216장)
- 증강: OpenCV로 반전·회전·밝기·블러·모션 블러 (기본 사진 5배, 우산·옷 사진 8배)
- 모델: YOLO11s, 80 epoch, RTX 5060 Ti 16GB Vram

| Test (83장, 동일) | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| v1 (826장, Train 3,300장) | 0.974 | 0.976 | 0.985 | 0.885 |
| **v2 (1,042장, Train 4,852장)** | **0.975** | **0.988** | 0.980 | **0.906** |

추적 결과 (ByteTrack, `track_buffer` 120, `match_thresh` 0.95)

| | video1 v1 | video1 v2 | video3 v1 | video3 v2 |
|---|---|---|---|---|
| 박스가 2개 이상인 프레임 (오탐지) | 0 | 0 | 84 | 67 |
| 평균 확신도 | 0.84 | 0.91 | 0.64 | 0.77 |
| 사용된 ID 수 | 1 | 3 | 6 | 4 |

- 결과 영상: [`toy_gun/video1_v2.mp4`](toy_gun/video1_v2.mp4), [`toy_gun/video3_v2.mp4`](toy_gun/video3_v2.mp4)
- 

https://github.com/user-attachments/assets/9da43e58-fc94-4d6a-b664-cdf64ac0206b


- 최종 모델: [`toy_gun/toy_gun_v2.pt`](toy_gun/toy_gun_v2.pt)
- 

https://github.com/user-attachments/assets/ca79a112-2304-47bb-abf2-9a31d9cc9f6d


- 한계: 화면 가까이에서 크게 보이는 접힌 우산과 어두운 바지는 아직 총으로 탐지하는 경우가 있습니다.

## 코드

| 파일 | 내용 |
|---|---|
| `3class/train.py`, `track.py`, `make_demo.py` | 학습, 영상 추적, 결과 영상 편집 |
| `toy_gun/convert_heic.py` | iPhone HEIC 사진을 JPG로 변환 |
| `toy_gun/autolabel.py`, `bootstrap.py` | YOLO-World 자동 라벨링, 의사 라벨링 |
| `toy_gun/select_review.py`, `review.py` | 검수할 사진 선택, 로컬 라벨 검수 도구 |
| `toy_gun/make_dataset.py`, `make_dataset_v2.py` | 데이터 분할과 증강 (v1, v2) |
| `toy_gun/train_v1.py`, `train_v2.py` | 학습, Test 평가, 추적 비교 |
| `toy_gun/bytetrack_tuned.yaml` | 조정한 ByteTrack 설정 |
| `lap.py`, `regex.py` | 실행 환경에서 막힌 라이브러리를 대체하는 모듈 |

## 실행

```bash
pip install ultralytics
yolo track model=toy_gun/toy_gun_v2.pt source=영상.mp4 tracker=toy_gun/bytetrack_tuned.yaml conf=0.25 imgsz=960 save=True
```

학습 사진과 데이터셋은 포함하지 않았습니다.
