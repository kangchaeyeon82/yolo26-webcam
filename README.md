# YOLO26으로 배우는 실시간 웹캠 비전 — 강의 노트

> Ultralytics YOLO26 모델로 웹캠 영상에서 **객체 탐지 → 세그멘테이션 → 객체 추적**을 차례로 구현하고,
> 완성한 코드를 GitHub CLI로 저장소에 올리는 과정까지 정리한 노트입니다.

## 목차

1. [학습 목표](#1-학습-목표)
2. [개발 환경 준비](#2-개발-환경-준비)
3. [트러블슈팅: `No module named 'cv2'`](#3-트러블슈팅-no-module-named-cv2)
4. [공통 뼈대: 웹캠 루프](#4-공통-뼈대-웹캠-루프)
5. [실습 1 — 객체 탐지 (`webcam_detect.py`)](#5-실습-1--객체-탐지-webcam_detectpy)
6. [실습 2 — 인스턴스 세그멘테이션 (`webcam_segment.py`)](#6-실습-2--인스턴스-세그멘테이션-webcam_segmentpy)
7. [실습 3 — 객체 추적 (`webcam_track.py`)](#7-실습-3--객체-추적-webcam_trackpy)
8. [세 가지 작업 비교](#8-세-가지-작업-비교)
9. [GitHub CLI로 코드 올리기](#9-github-cli로-코드-올리기)
10. [정리 및 연습 문제](#10-정리-및-연습-문제)

---

## 1. 학습 목표

이 노트를 마치면 다음을 할 수 있습니다.

- Ultralytics 패키지를 설치하고 YOLO26 사전학습 모델을 불러올 수 있다.
- OpenCV로 웹캠 프레임을 읽고, 결과를 화면에 실시간으로 그릴 수 있다.
- **탐지(Detection)**, **세그멘테이션(Segmentation)**, **추적(Tracking)**의 차이를 설명하고 각각 구현할 수 있다.
- `ModuleNotFoundError` 같은 환경 문제를 스스로 진단할 수 있다.
- `git`과 `gh`(GitHub CLI)로 프로젝트를 원격 저장소에 올릴 수 있다.

---

## 2. 개발 환경 준비

### 2-1. 사용한 환경

| 항목 | 버전 |
|---|---|
| OS | Windows 11 |
| Python | 3.12 |
| ultralytics | 8.4.x |
| opencv-python | 5.0 |
| torch | 2.14 (CPU) |

### 2-2. 설치

```bash
python -m pip install -U ultralytics
```

`ultralytics` 하나만 설치하면 **OpenCV(`cv2`)**, **PyTorch**, **NumPy** 등 필요한 패키지가 함께 설치됩니다.

> 💡 **Tip — `pip` 대신 `python -m pip`를 쓰는 이유**
> PC에 Python이 여러 개 있으면 `pip`와 `python`이 서로 다른 Python을 가리킬 수 있습니다.
> `python -m pip`는 "지금 `python` 명령이 실행하는 바로 그 Python"에 설치하므로 이런 혼동을 막아 줍니다.

### 2-3. YOLO26 모델 이름 규칙

| 작업 | 모델 파일 | 크기별 버전 |
|---|---|---|
| 탐지 | `yolo26n.pt` | n / s / m / l / x |
| 세그멘테이션 | `yolo26n-seg.pt` | n / s / m / l / x |

- `n`(nano)이 가장 작고 빠르며, `x`로 갈수록 정확도는 높아지지만 느려집니다.
- 가중치 파일은 `YOLO("yolo26n.pt")`를 처음 실행할 때 **자동으로 다운로드**됩니다.

---

## 3. 트러블슈팅: `No module named 'cv2'`

실습 중 실제로 만난 오류입니다.

```
ModuleNotFoundError: No module named 'cv2'
```

### 원인

`cv2`는 **OpenCV 패키지(`opencv-python`)의 모듈 이름**입니다.
패키지 이름과 `import` 이름이 다르다는 점이 헷갈리기 쉽습니다.

| 설치할 때 (pip) | 코드에서 (import) |
|---|---|
| `opencv-python` | `import cv2` |
| `ultralytics` | `from ultralytics import YOLO` |

이번 경우에는 `ultralytics`를 아직 설치하지 않아서 OpenCV도 없었습니다.

### 진단 순서

```bash
where.exe python                              # 어떤 Python들이 있는지 확인
python --version                              # 지금 쓰는 Python 버전
python -m pip show ultralytics opencv-python  # 설치 여부 확인
```

### 해결

```bash
python -m pip install -U ultralytics
python -c "import cv2, ultralytics; print(cv2.__version__, ultralytics.__version__)"
```

> ⚠️ **설치했는데도 같은 오류가 난다면?**
> VS Code의 ▶ 버튼이 **다른 Python**으로 실행하고 있을 가능성이 큽니다.
> `Ctrl+Shift+P` → **Python: Select Interpreter**에서 패키지를 설치한 Python을 선택하세요.

---

## 4. 공통 뼈대: 웹캠 루프

세 실습 코드는 모두 아래 구조를 공유합니다. **이 뼈대를 먼저 이해하면 나머지는 "가운데 한 줄"만 바뀝니다.**

```python
import cv2
from ultralytics import YOLO

model = YOLO("yolo26n.pt")                    # ① 모델 불러오기

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)      # ② 웹캠 열기 (0번 카메라)

while True:
    ok, frame = cap.read()                    # ③ 프레임 한 장 읽기
    if not ok:
        break

    results = model.predict(frame)            # ④ 추론  ← 실습마다 이 부분이 바뀜
    annotated = results[0].plot()             # ⑤ 결과를 이미지에 그리기

    cv2.imshow("YOLO26", annotated)           # ⑥ 화면에 보여주기
    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):  # ⑦ q 또는 ESC로 종료
        break

cap.release()                                 # ⑧ 자원 정리
cv2.destroyAllWindows()
```

### 핵심 포인트

| 코드 | 설명 |
|---|---|
| `cv2.CAP_DSHOW` | Windows에서 DirectShow 백엔드를 사용해 웹캠을 더 빠르고 안정적으로 엽니다. |
| `results[0]` | `predict`는 결과를 **리스트**로 돌려줍니다. 이미지 1장을 넣었으니 첫 번째 원소만 씁니다. |
| `.plot()` | 박스·라벨·마스크를 그린 **새 이미지(NumPy 배열)**를 돌려줍니다. 원본 `frame`은 바뀌지 않습니다. |
| `cv2.waitKey(1)` | 1ms 동안 키 입력을 기다립니다. 이 호출이 없으면 창이 갱신되지 않습니다. |
| `try / finally` | 실제 코드에서는 오류가 나도 카메라가 반드시 해제되도록 `finally`에서 `release()`합니다. |

### FPS 계산

```python
now = time.perf_counter()
fps = 1.0 / (now - prev)   # 한 프레임 처리에 걸린 시간의 역수
prev = now
```

### 명령줄 옵션 (`argparse`)

세 파일 모두 같은 옵션 체계를 사용합니다. `python 파일명.py --help`로 확인할 수 있습니다.

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--model` | 파일마다 다름 | 사용할 가중치 |
| `--source` | `0` | 웹캠 번호 (외장 웹캠은 보통 `1`) |
| `--conf` | `0.25` | 이 값보다 신뢰도가 낮은 결과는 버림 |
| `--imgsz` | `640` | 추론 시 입력 이미지 크기 |
| `--device` | 자동 | `0`(첫 번째 GPU) 또는 `cpu` |

---

## 5. 실습 1 — 객체 탐지 (`webcam_detect.py`)

### 개념

**객체 탐지(Object Detection)**는 이미지 안의 물체가 **무엇(클래스)**이고 **어디(바운딩 박스)**에 있는지 찾는 작업입니다.

```
입력 이미지 ──▶ [ (person, 0.91, x1,y1,x2,y2), (cup, 0.78, ...), ... ]
```

### 핵심 코드

```python
results = model.predict(frame, conf=args.conf, imgsz=args.imgsz,
                        device=args.device, verbose=False)
result = results[0]
annotated = result.plot()
print(len(result.boxes))   # 탐지된 객체 수
```

### 결과 객체 살펴보기 — `result.boxes`

| 속성 | 모양 | 의미 |
|---|---|---|
| `boxes.xyxy` | (N, 4) | 좌상단·우하단 좌표 |
| `boxes.xywh` | (N, 4) | 중심 좌표 + 너비·높이 |
| `boxes.conf` | (N,) | 신뢰도 |
| `boxes.cls` | (N,) | 클래스 번호 → `result.names[번호]`로 이름 확인 |

### 실행

```bash
python webcam_detect.py
python webcam_detect.py --model yolo26s.pt --conf 0.4
```

> 💡 **응용** — 사람만 탐지하고 싶다면 `model.predict(..., classes=[0])`처럼 COCO 클래스 번호를 지정합니다. (`0` = person)

---

## 6. 실습 2 — 인스턴스 세그멘테이션 (`webcam_segment.py`)

### 개념

**인스턴스 세그멘테이션(Instance Segmentation)**은 박스보다 한 단계 더 나아가 **물체의 픽셀 단위 윤곽(마스크)**을 찾습니다.
같은 클래스라도 사람 A와 사람 B를 **서로 다른 마스크**로 구분하는 것이 "인스턴스"의 의미입니다.

| 탐지 | 세그멘테이션 |
|---|---|
| 사각형 박스 | 물체 모양 그대로의 마스크 |
| 배경이 박스 안에 섞임 | 물체 픽셀만 정확히 분리 |

### 탐지 코드에서 바뀐 부분

```python
model = YOLO("yolo26n-seg.pt")                 # ① -seg 모델 사용

results = model.predict(frame, ...,
                        retina_masks=True)     # ② 고해상도 마스크

annotated = result.plot(boxes=not args.no_boxes)  # ③ 박스 표시 여부 선택
n = 0 if result.masks is None else len(result.masks)  # ④ 아무것도 없으면 masks는 None
```

| 포인트 | 설명 |
|---|---|
| `-seg` 모델 | 탐지 모델에는 마스크를 출력하는 헤드가 없습니다. 반드시 세그멘테이션 모델을 써야 합니다. |
| `retina_masks=True` | 마스크를 **원본 해상도**로 만들어 경계가 매끄럽습니다. 대신 조금 느려집니다. |
| `result.masks is None` | 탐지된 물체가 없으면 `masks`가 `None`이므로, `len()` 전에 꼭 확인해야 합니다. |

### 마스크 데이터 직접 다루기

| 속성 | 모양 | 용도 |
|---|---|---|
| `masks.data` | (N, H, W) | 0/1 픽셀 마스크 — 배경 제거, 면적 계산 |
| `masks.xy` | N개의 (K, 2) 배열 | 외곽선 폴리곤 좌표 — 윤곽 그리기 |

### 실행

```bash
python webcam_segment.py
python webcam_segment.py --no-boxes        # 마스크만 표시
```

---

## 7. 실습 3 — 객체 추적 (`webcam_track.py`)

### 개념

**객체 추적(Object Tracking)**은 탐지 결과에 **ID**를 붙여, 다음 프레임에서도 **같은 물체에 같은 ID**를 유지하는 작업입니다.

```
프레임 1: person(id:1), person(id:2)
프레임 2: person(id:1), person(id:2)   ← 위치가 바뀌어도 ID 유지
프레임 3: person(id:1), person(id:2), cup(id:3)   ← 새 물체는 새 ID
```

탐지는 **매 프레임을 독립적으로** 보지만, 추적은 **프레임 사이의 연결**을 만든다는 점이 핵심 차이입니다.
사람 수 세기, 이동 경로 분석, 구역 침입 감지 같은 응용의 기초가 됩니다.

### 핵심 코드

```python
results = model.track(frame, persist=True, tracker="bytetrack.yaml", ...)
```

| 인자 | 설명 |
|---|---|
| `model.track` | `predict` 대신 사용합니다. 탐지 + ID 부여를 함께 합니다. |
| `persist=True` | ⭐ **가장 중요.** 이전 프레임의 추적 상태를 기억합니다. 빠뜨리면 매 프레임 ID가 새로 매겨집니다. |
| `tracker` | 추적 알고리즘 선택 (아래 표 참고) |

### 추적 알고리즘 비교

| 트래커 | 특징 |
|---|---|
| `bytetrack.yaml` (기본) | 빠르고 가벼움. 신뢰도가 낮은 박스도 활용해 추적이 잘 끊기지 않음 |
| `botsort.yaml` | 카메라 움직임 보정 등 추가. 가려졌다 다시 나타나는 물체에 더 강함 |

> 📦 처음 추적을 실행하면 Ultralytics가 필요한 `lap` 패키지를 **자동으로 설치**합니다.

### 이동 궤적(Trail) 그리기

```python
trails = defaultdict(lambda: deque(maxlen=30))   # ID별로 최근 30개 중심점만 보관

if boxes.id is not None:                         # 추적된 물체가 없으면 id는 None
    for tid, (x, y, _, _) in zip(boxes.id.int().tolist(), boxes.xywh.cpu().numpy()):
        trails[tid].append((int(x), int(y)))
        pts = np.array(trails[tid], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(annotated, [pts], False, id_color(tid), 2)
```

| 기법 | 이유 |
|---|---|
| `defaultdict` | 처음 보는 ID도 따로 초기화할 필요 없이 바로 `append` 가능 |
| `deque(maxlen=30)` | 오래된 점은 자동으로 버려져 메모리가 계속 늘지 않음 |
| `id_color(tid)` | ID를 시드로 난수를 만들어 **같은 ID는 항상 같은 색** |
| `boxes.id is not None` | 추적 대상이 없는 프레임에서 오류를 막는 방어 코드 |

### 실행

```bash
python webcam_track.py
python webcam_track.py --tracker botsort.yaml --trail 60
python webcam_track.py --model yolo26n-seg.pt     # 마스크 + 추적
```

---

## 8. 세 가지 작업 비교

| | 탐지 | 세그멘테이션 | 추적 |
|---|---|---|---|
| 파일 | `webcam_detect.py` | `webcam_segment.py` | `webcam_track.py` |
| 모델 | `yolo26n.pt` | `yolo26n-seg.pt` | `yolo26n.pt` (또는 `-seg`) |
| 호출 | `model.predict()` | `model.predict(retina_masks=True)` | `model.track(persist=True)` |
| 출력 | 박스 + 클래스 | 박스 + 클래스 + **마스크** | 박스 + 클래스 + **ID** |
| 핵심 속성 | `result.boxes` | `result.masks` | `result.boxes.id` |
| 대표 응용 | 물체 존재 여부 확인 | 배경 제거, 면적 측정 | 인원 수 세기, 동선 분석 |

---

## 9. GitHub CLI로 코드 올리기

### 9-1. 사전 확인

```bash
gh auth status                 # GitHub 로그인 상태 확인
git config --global user.name  # 커밋 작성자 이름
git config --global user.email # 커밋 작성자 이메일
```

> ⚠️ **실습 중 발견한 문제** — 전역 설정에 `사용할 이름`, `사용할 이메일` 같은 **예시 문구가 그대로** 들어 있었습니다.
> 이대로 커밋하면 작성자 정보가 엉터리로 기록되므로 커밋 전에 꼭 확인하세요.

### 9-2. `.gitignore`로 큰 파일 제외

```gitignore
*.pt          # 모델 가중치 — 실행 시 자동 다운로드되므로 올릴 필요 없음
__pycache__/
runs/
```

가중치 파일은 수 MB~수백 MB라 저장소를 무겁게 만들고, 어차피 다시 받을 수 있으므로 제외합니다.

### 9-3. 커밋하고 저장소 만들기

```bash
git init -b main
git config user.name  "깃허브아이디"                              # 이 저장소에만 적용
git config user.email "숫자ID+아이디@users.noreply.github.com"   # 실제 이메일 노출 방지
git add .
git commit -m "Add YOLO26 webcam detection, segmentation and tracking scripts"

gh repo create yolo26-webcam --private --source=. --remote=origin --push
```

| 옵션 | 의미 |
|---|---|
| `--private` | 비공개 저장소로 생성 (공개는 `--public`) |
| `--source=.` | 현재 폴더를 저장소로 사용 |
| `--remote=origin` | 원격 저장소 이름을 `origin`으로 등록 |
| `--push` | 생성 직후 바로 push |

> 💡 **noreply 이메일** — GitHub는 `숫자ID+아이디@users.noreply.github.com` 형태의 이메일을 제공합니다.
> 공개 저장소에 개인 이메일이 노출되는 것을 막을 수 있습니다. 숫자 ID는 `gh api user --jq .id`로 확인합니다.

### 9-4. 나중에 공개로 바꾸기

```bash
gh repo edit 아이디/yolo26-webcam --visibility public --accept-visibility-change-consequences
```

---

## 10. 정리 및 연습 문제

### 오늘 배운 것 한눈에 보기

1. `python -m pip install -U ultralytics` 한 줄로 YOLO26 + OpenCV 환경을 갖춘다.
2. `cv2` 오류는 패키지 이름(`opencv-python`)과 import 이름(`cv2`)이 다르기 때문에 헷갈리기 쉽다. **어떤 Python에 설치했는지**부터 확인한다.
3. 웹캠 루프는 **읽기 → 추론 → 그리기 → 표시 → 키 확인**의 반복이다.
4. 탐지 → 세그멘테이션은 **모델만 `-seg`로**, 탐지 → 추적은 **`predict`를 `track(persist=True)`로** 바꾸면 된다.
5. `gh repo create --source=. --push`로 로컬 폴더를 바로 GitHub 저장소로 만들 수 있다.

### 연습 문제

1. **[탐지]** `classes` 인자를 사용해 사람(`0`)과 휴대폰(`67`)만 탐지하도록 바꿔 보세요.
2. **[탐지]** `s` 키를 누르면 현재 화면을 `capture_001.jpg`처럼 번호를 붙여 저장하는 기능을 추가해 보세요.
3. **[세그멘테이션]** `result.masks.data`를 이용해 **사람만 남기고 배경을 검게** 만들어 보세요.
4. **[추적]** 화면 가운데에 세로선을 긋고, 왼쪽에서 오른쪽으로 **선을 넘은 사람 수**를 세어 보세요.
   *(힌트: `trails[tid]`의 마지막 두 점의 x 좌표를 비교)*
5. **[성능]** `yolo26n.pt`와 `yolo26s.pt`의 FPS를 비교하고, `--imgsz 320`일 때 속도와 정확도가 어떻게 달라지는지 기록해 보세요.

---

## 부록: 빠른 실행

```bash
python -m pip install -U ultralytics

python webcam_detect.py     # 객체 탐지
python webcam_segment.py    # 인스턴스 세그멘테이션
python webcam_track.py      # 객체 추적
```

영상 창에서 `q` 또는 `ESC`를 누르면 종료합니다.
