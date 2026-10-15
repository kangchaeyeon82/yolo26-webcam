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
11. [용어 사전 (비전공자용)](#11-용어-사전-비전공자용)
12. [외부 도구 · 연결 서비스 정리](#12-외부-도구--연결-서비스-정리)

> 📖 처음 보는 용어가 나오면 [11장 용어 사전](#11-용어-사전-비전공자용)을 먼저 확인하세요.

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

## 11. 용어 사전 (비전공자용)

> 이 노트에 나오는 용어를 **처음 배우는 분도 이해할 수 있도록** 풀어 썼습니다.
> 각 용어 옆의 *(→ n장)* 은 그 용어가 주로 등장하는 장입니다.

### 11-1. 인공지능 · 컴퓨터 비전 기본 개념

| 용어 | 쉬운 설명 |
|---|---|
| **컴퓨터 비전** (Computer Vision) | 컴퓨터가 사진이나 영상을 보고 그 안에 무엇이 있는지 이해하게 만드는 기술입니다. 사람의 "눈 + 판단" 역할을 컴퓨터가 하는 것입니다. |
| **딥러닝** (Deep Learning) | 사람의 뇌 신경망을 흉내 낸 수학 모델에 수많은 예시를 보여 주며 스스로 규칙을 익히게 하는 인공지능 방법입니다. |
| **모델** (Model) | 학습을 마친 인공지능 "두뇌"입니다. 사진을 넣으면 답(무엇이 어디에 있는지)을 내놓는 계산 장치라고 생각하면 됩니다. |
| **YOLO** | *You Only Look Once*(한 번만 본다)의 약자입니다. 사진을 **한 번에 훑어보고** 모든 물체를 찾아내서 매우 빠른 것이 특징인 모델 계열입니다. **YOLO26**은 그 최신 버전입니다. *(→ 2장)* |
| **Ultralytics** | YOLO 모델을 만들고 배포하는 회사이자, 그 모델을 파이썬에서 쉽게 쓰도록 만든 **패키지 이름**입니다. *(→ 2장)* |
| **사전학습 모델** (Pretrained Model) | 다른 사람이 이미 엄청난 양의 사진으로 학습시켜 둔 모델입니다. 우리는 학습 과정 없이 **바로 가져다 쓰기만** 하면 됩니다. |
| **가중치** (Weights) / `.pt` 파일 | 모델이 학습하며 익힌 "지식"을 숫자로 저장한 파일입니다. `yolo26n.pt`가 바로 이 파일이며, `.pt`는 PyTorch 형식이라는 뜻입니다. *(→ 2장)* |
| **추론** (Inference) | 학습이 끝난 모델에 새 사진을 넣어 **답을 얻는 과정**입니다. 코드의 `model.predict()`가 추론입니다. *(→ 4장)* |
| **학습** (Training) | 모델에게 정답이 달린 사진을 많이 보여 주며 가중치를 조정하는 과정입니다. 이번 실습에서는 하지 않았습니다. |
| **COCO 데이터셋** | 사람, 자동차, 컵 등 **80가지 물체**가 표시된 유명한 사진 모음입니다. YOLO26 기본 모델은 이 80가지를 알아봅니다. *(→ 5장)* |
| **클래스** (Class) | 모델이 구분할 수 있는 물체의 **종류**입니다. 예: `person`(사람), `cup`(컵). 각 클래스에는 번호가 붙어 있습니다(사람 = 0). *(→ 5장)* |

### 11-2. 세 가지 비전 작업

| 용어 | 쉬운 설명 |
|---|---|
| **객체 탐지** (Object Detection) | 사진 속 물체가 **무엇**이고 **어디** 있는지 네모 상자로 표시하는 작업입니다. *(→ 5장)* |
| **바운딩 박스** (Bounding Box) | 물체를 감싸는 **네모 상자**입니다. 왼쪽 위·오른쪽 아래 꼭짓점 좌표로 나타냅니다. *(→ 5장)* |
| **신뢰도** (Confidence) | 모델이 자기 답을 **얼마나 확신하는지**를 0~1 사이 숫자로 나타낸 것입니다. `0.91`이면 "91% 확신한다"는 뜻입니다. *(→ 4장)* |
| **임계값** (Threshold) | "이 값보다 낮으면 버린다"는 **기준선**입니다. `--conf 0.25`는 확신이 25% 미만인 결과는 무시하라는 뜻입니다. *(→ 4장)* |
| **세그멘테이션** (Segmentation) | 네모 상자 대신 물체의 **실제 모양대로 픽셀을 칠하는** 작업입니다. 색칠공부처럼 물체의 윤곽을 따라 칠한다고 생각하면 됩니다. *(→ 6장)* |
| **인스턴스** (Instance) | 같은 종류 안의 **개별 물체 하나하나**입니다. 사람이 세 명이면 "사람" 클래스의 인스턴스가 세 개입니다. *(→ 6장)* |
| **마스크** (Mask) | 세그멘테이션 결과로, 각 픽셀이 물체에 속하면 1, 아니면 0으로 표시한 **흑백 지도**입니다. *(→ 6장)* |
| **픽셀** (Pixel) | 디지털 이미지를 이루는 **가장 작은 점**입니다. 1280×720 화면은 가로 1280개, 세로 720개의 점으로 되어 있습니다. |
| **폴리곤** (Polygon) | 여러 점을 이어 만든 **다각형**입니다. 마스크의 외곽선을 점들의 좌표로 표현할 때 씁니다. *(→ 6장)* |
| **객체 추적** (Object Tracking) | 영상에서 물체에 **번호표(ID)**를 붙이고, 물체가 움직여도 다음 장면에서 **같은 번호를 유지**하는 작업입니다. *(→ 7장)* |
| **트래커** (Tracker) | 추적을 담당하는 알고리즘입니다. 이번 실습에서는 **ByteTrack**과 **BoT-SORT** 두 가지를 썼습니다. *(→ 7장)* |
| **궤적** (Trail) | 물체가 지나온 길을 선으로 그린 것입니다. 물체의 이동 방향을 한눈에 볼 수 있습니다. *(→ 7장)* |

### 11-3. 영상 · 성능 관련

| 용어 | 쉬운 설명 |
|---|---|
| **프레임** (Frame) | 영상을 이루는 **사진 한 장**입니다. 영상은 사진을 빠르게 넘기는 플립북과 같습니다. *(→ 4장)* |
| **FPS** (Frames Per Second) | 1초에 처리하는 프레임 수입니다. 높을수록 화면이 부드럽습니다. 보통 **15 이상**이면 실시간처럼 느껴집니다. *(→ 4장)* |
| **실시간** (Real-time) | 카메라에 찍히는 것과 거의 **동시에** 결과가 나오는 것입니다. |
| **해상도** (Resolution) | 이미지의 가로×세로 픽셀 수입니다. 높을수록 선명하지만 처리량이 많아져 느려집니다. |
| **`imgsz`** | 모델에 넣기 전에 이미지를 줄이는 크기입니다. `640`이면 긴 변을 640픽셀로 맞춥니다. 작을수록 빠르고, 클수록 작은 물체를 잘 찾습니다. *(→ 4장)* |
| **CPU / GPU** | CPU는 컴퓨터의 범용 계산 장치이고, GPU는 그래픽 카드로 **많은 계산을 동시에** 처리하는 데 강합니다. 딥러닝은 GPU에서 훨씬 빠릅니다. *(→ 4장)* |
| **CUDA** | NVIDIA 그래픽 카드를 딥러닝 계산에 쓸 수 있게 해 주는 소프트웨어입니다. `--device 0`으로 GPU를 쓰려면 필요합니다. |
| **웹캠 인덱스** | 컴퓨터에 연결된 카메라의 **번호**입니다. 노트북 내장 카메라는 보통 `0`, 외장 카메라는 `1`입니다. *(→ 4장)* |

### 11-4. 파이썬 · 프로그래밍

| 용어 | 쉬운 설명 |
|---|---|
| **파이썬** (Python) | 문법이 쉬워 인공지능 분야에서 가장 많이 쓰는 프로그래밍 언어입니다. |
| **인터프리터** (Interpreter) | 파이썬 코드를 읽고 실행해 주는 프로그램(`python.exe`)입니다. PC에 여러 개 설치될 수 있어 혼동의 원인이 됩니다. *(→ 3장)* |
| **패키지 / 라이브러리** | 다른 사람이 만들어 둔 **코드 묶음**입니다. 직접 만들지 않고 설치해서 기능을 빌려 씁니다. 예: `ultralytics`, `opencv-python` |
| **pip** | 파이썬 패키지를 **설치해 주는 도구**입니다. 앱스토어처럼 이름만 대면 내려받아 설치합니다. *(→ 2장)* |
| **import** | 설치한 패키지를 코드에서 **불러와 쓰겠다**고 선언하는 명령입니다. *(→ 3장)* |
| **모듈** (Module) | `import`로 불러오는 단위입니다. 설치 이름(`opencv-python`)과 모듈 이름(`cv2`)이 다를 수 있습니다. *(→ 3장)* |
| **`ModuleNotFoundError`** | "불러오려는 모듈을 찾을 수 없다"는 오류입니다. 대부분 **설치를 안 했거나, 다른 파이썬에 설치한 경우**입니다. *(→ 3장)* |
| **OpenCV** (`cv2`) | 카메라 열기, 이미지에 그림 그리기, 창 띄우기 등 **영상 처리 기능을 모아 둔** 대표 라이브러리입니다. *(→ 4장)* |
| **PyTorch** | YOLO 모델이 내부적으로 사용하는 **딥러닝 계산 엔진**입니다. |
| **NumPy** | 숫자 배열을 빠르게 계산하는 라이브러리입니다. 컴퓨터는 이미지를 **숫자가 담긴 거대한 표(배열)**로 다룹니다. |
| **배열 / 텐서** (Array / Tensor) | 숫자를 여러 차원으로 정리한 데이터입니다. 이미지는 `세로 × 가로 × 색(3)` 모양의 배열입니다. |
| **모양** (Shape) | 배열의 크기를 나타낸 표기입니다. `(N, 4)`는 "N개의 물체 각각에 숫자 4개"라는 뜻입니다. *(→ 5장)* |
| **BGR** | OpenCV의 색 순서(파랑-초록-빨강)입니다. 흔히 쓰는 RGB와 순서가 반대라 색이 뒤바뀌어 보이는 실수가 잦습니다. |
| **반복문** (`while True`) | 같은 작업을 **계속 되풀이**하는 코드입니다. 웹캠 루프는 종료 키를 누를 때까지 프레임을 계속 처리합니다. *(→ 4장)* |
| **`try / finally`** | 중간에 오류가 나도 `finally` 안의 코드는 **반드시 실행**되게 하는 문법입니다. 카메라를 확실히 끄는 데 씁니다. *(→ 4장)* |
| **`argparse` / 명령줄 옵션** | 코드를 고치지 않고 실행할 때 `--conf 0.4`처럼 **설정을 바꿀 수 있게** 해 주는 기능입니다. *(→ 4장)* |
| **`defaultdict` / `deque`** | 파이썬의 편리한 자료 구조입니다. `defaultdict`는 없는 키를 자동으로 만들고, `deque(maxlen=30)`은 30개가 넘으면 가장 오래된 것을 자동으로 버립니다. *(→ 7장)* |
| **시드** (Seed) | 난수(무작위 숫자)를 만드는 출발값입니다. 시드가 같으면 항상 같은 난수가 나와서 **같은 ID에 같은 색**을 줄 수 있습니다. *(→ 7장)* |
| **`None`** | 파이썬에서 "값이 없음"을 뜻합니다. 물체가 하나도 없으면 `result.masks`가 `None`이 됩니다. *(→ 6장)* |

### 11-5. Git · GitHub

| 용어 | 쉬운 설명 |
|---|---|
| **Git** | 파일의 **변경 이력을 기록**하는 프로그램입니다. 게임의 세이브 포인트처럼 언제든 이전 상태로 돌아갈 수 있습니다. *(→ 9장)* |
| **GitHub** | Git으로 관리하는 프로젝트를 **인터넷에 올려 보관·공유**하는 웹사이트입니다. *(→ 9장)* |
| **GitHub CLI** (`gh`) | 웹사이트에 들어가지 않고 **명령어로** GitHub 작업(저장소 만들기 등)을 할 수 있게 해 주는 도구입니다. *(→ 9장)* |
| **CLI** (Command Line Interface) | 마우스 대신 **글자로 명령을 입력**해 컴퓨터를 조작하는 방식입니다. 터미널, PowerShell이 여기에 해당합니다. |
| **저장소** (Repository, Repo) | 프로젝트 파일과 그 변경 이력을 담은 **폴더**입니다. *(→ 9장)* |
| **커밋** (Commit) | 현재 파일 상태를 **세이브 포인트로 저장**하는 것입니다. 무엇을 바꿨는지 메시지를 함께 남깁니다. *(→ 9장)* |
| **푸시** (Push) | 내 컴퓨터의 커밋을 GitHub로 **업로드**하는 것입니다. *(→ 9장)* |
| **원격 저장소 / `origin`** | GitHub에 있는 저장소입니다. `origin`은 그 원격 저장소에 붙이는 **기본 별명**입니다. *(→ 9장)* |
| **브랜치 / `main`** | 작업 흐름의 **갈래**입니다. `main`은 기본 갈래의 이름입니다. |
| **`.gitignore`** | Git이 **무시할 파일 목록**입니다. 용량이 큰 모델 파일처럼 올릴 필요 없는 파일을 적어 둡니다. *(→ 9장)* |
| **Public / Private** | 저장소 공개 범위입니다. Public은 누구나 볼 수 있고, Private는 나와 초대한 사람만 볼 수 있습니다. *(→ 9장)* |
| **noreply 이메일** | GitHub가 제공하는 **가짜 이메일 주소**입니다. 커밋 기록에 실제 이메일이 노출되지 않게 해 줍니다. *(→ 9장)* |
| **README.md** | 저장소에 들어가면 가장 먼저 보이는 **설명서 파일**입니다. 지금 읽고 있는 이 문서입니다. |
| **Markdown** (`.md`) | `#`, `|`, `**` 같은 간단한 기호로 제목·표·굵은 글씨를 표현하는 **문서 작성 형식**입니다. |

---

## 12. 외부 도구 · 연결 서비스 정리

이 프로젝트가 **바깥에 연결되거나 기대는 것**을 한곳에 모았습니다. 다른 PC에서 다시 실행하거나 문제를 찾을 때 이 표부터 확인하세요.

### 12-1. 파이썬 라이브러리

| 도구 | 확인한 버전 | 역할 | 설치 방법 |
|---|---|---|---|
| `ultralytics` | 8.4.174 | YOLO26 모델 불러오기·추론·추적 | `python -m pip install -U ultralytics` |
| `torch` (PyTorch) | 2.14.1 (CPU) | 딥러닝 계산 엔진 | ultralytics와 함께 자동 설치 |
| `opencv-python` (`cv2`) | 5.0.0 | 웹캠 열기, 화면 표시, 그리기 | ultralytics와 함께 자동 설치 |
| `numpy` | 2.5.3 | 배열 계산 (추적 궤적 그리기 등) | ultralytics와 함께 자동 설치 |
| `lap` | 0.5.13 | 추적기(ByteTrack/BoT-SORT)의 매칭 계산 | 첫 추적 실행 시 자동 설치 |

### 12-2. 인터넷으로 연결되는 곳

| 연결 대상 | 언제 | 내용 |
|---|---|---|
| **Ultralytics GitHub 릴리스** | `YOLO("yolo26n.pt")`를 처음 실행할 때 | 가중치 파일(`yolo26n.pt`, `yolo26n-seg.pt`)을 내려받습니다. 받은 파일은 폴더에 저장되고 `.gitignore`로 업로드에서 빠집니다. |
| **PyPI** (파이썬 패키지 저장소) | `pip install` 할 때, 첫 추적 실행 때 | 라이브러리를 내려받습니다. `lap`은 Ultralytics가 자동으로 설치합니다. |
| **Ultralytics 사용 통계** | 모델을 실행할 때 | 설정값 `sync`가 켜져(`true`) 있으면 익명 사용 통계를 보냅니다. 끄려면 `yolo settings sync=False` |
| **GitHub** | `git push`, `gh` 명령을 쓸 때 | 코드를 [kangchaeyeon82/yolo26-webcam](https://github.com/kangchaeyeon82/yolo26-webcam) 저장소에 올립니다. |

> 💡 가중치를 한 번 받은 뒤에는 **인터넷 없이도** 탐지·세그멘테이션이 동작합니다. (추적은 `lap`이 설치돼 있어야 합니다.)

### 12-3. 하드웨어

| 장치 | 연결 방식 | 비고 |
|---|---|---|
| **웹캠** | OpenCV `VideoCapture` + Windows **DirectShow**(`cv2.CAP_DSHOW`) | 안 열리면 기본 백엔드로 다시 시도합니다. 번호는 `--source`로 바꿉니다. |
| **GPU** (선택) | PyTorch + CUDA | 현재 환경은 CPU 버전 PyTorch라 CPU로 실행됩니다. |

### 12-4. 개발 도구

| 도구 | 확인한 버전 | 역할 |
|---|---|---|
| Git | 2.55.0 | 변경 이력 기록, 커밋·푸시 |
| GitHub CLI (`gh`) | 2.102.0 | 저장소 만들기. 로그인 정보는 Windows 자격 증명 관리자(keyring)에 저장되고, Git 통신은 HTTPS를 씁니다. |
| Python | 3.12 | 코드 실행 |

### 12-5. 내 환경 직접 확인하기

```bash
python -c "import ultralytics, torch, cv2, numpy; print(ultralytics.__version__, torch.__version__, cv2.__version__, numpy.__version__)"
yolo settings          # Ultralytics 설정 (sync 등)
git remote -v          # 연결된 GitHub 저장소
gh auth status         # GitHub 로그인 상태
```

---

## 부록: 빠른 실행

```bash
python -m pip install -U ultralytics

python webcam_detect.py     # 객체 탐지
python webcam_segment.py    # 인스턴스 세그멘테이션
python webcam_track.py      # 객체 추적
```

영상 창에서 `q` 또는 `ESC`를 누르면 종료합니다.
