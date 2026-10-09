# YOLO26 Webcam

Ultralytics YOLO26 모델로 웹캠 영상을 실시간 처리하는 예제입니다.

| 파일 | 기능 | 기본 모델 |
|---|---|---|
| `webcam_detect.py` | 객체 탐지 | `yolo26n.pt` |
| `webcam_segment.py` | 인스턴스 세그멘테이션 | `yolo26n-seg.pt` |
| `webcam_track.py` | 객체 추적 (ByteTrack / BoT-SORT) | `yolo26n.pt` |

## 설치

```bash
pip install -U ultralytics
```

## 실행

```bash
python webcam_detect.py
python webcam_segment.py
python webcam_track.py
```

모델 가중치는 처음 실행할 때 자동으로 다운로드됩니다. 영상 창에서 `q` 또는 `ESC`로 종료합니다.
`--help`로 옵션(`--model`, `--conf`, `--device`, `--source` 등)을 확인할 수 있습니다.
