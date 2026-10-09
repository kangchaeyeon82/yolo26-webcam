"""YOLO26 웹캠 실시간 객체 추적 (Object Tracking)

설치:  pip install -U ultralytics
실행:  python webcam_track.py
종료:  영상 창에서 'q' 또는 ESC
"""
import argparse
import time
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="YOLO26 webcam object tracking")
    p.add_argument("--model", default="yolo26n.pt",
                   help="yolo26n/s/m/l/x.pt (처음 실행 시 자동 다운로드)")
    p.add_argument("--tracker", default="bytetrack.yaml",
                   choices=["bytetrack.yaml", "botsort.yaml"], help="추적 알고리즘")
    p.add_argument("--source", type=int, default=0, help="웹캠 인덱스")
    p.add_argument("--conf", type=float, default=0.25, help="신뢰도 임계값")
    p.add_argument("--imgsz", type=int, default=640, help="추론 이미지 크기")
    p.add_argument("--device", default=None, help="예: 0 (GPU), cpu")
    p.add_argument("--trail", type=int, default=30, help="이동 궤적 길이(프레임 수), 0이면 끔")
    return p.parse_args()


def id_color(track_id):
    """ID마다 고정된 색을 만든다 (BGR)."""
    rng = np.random.default_rng(track_id)
    return tuple(int(c) for c in rng.integers(64, 256, size=3))


def main():
    args = parse_args()
    model = YOLO(args.model)

    # Windows에서는 CAP_DSHOW가 웹캠을 더 빨리/안정적으로 연다
    cap = cv2.VideoCapture(args.source, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"웹캠({args.source})을 열 수 없습니다.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window = "YOLO26 Webcam Tracking"
    trails = defaultdict(lambda: deque(maxlen=max(args.trail, 1)))  # ID별 중심점 기록
    seen_ids = set()
    prev = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("프레임을 읽지 못했습니다.")
                break

            # persist=True: 프레임 간 추적 상태를 유지해 같은 객체에 같은 ID를 부여
            results = model.track(frame, persist=True, tracker=args.tracker,
                                  conf=args.conf, imgsz=args.imgsz,
                                  device=args.device, verbose=False)
            result = results[0]
            annotated = result.plot()  # 박스 + "id:N 클래스 신뢰도" 라벨

            boxes = result.boxes
            active = 0
            if boxes is not None and boxes.id is not None:
                ids = boxes.id.int().tolist()
                xywh = boxes.xywh.cpu().numpy()
                active = len(ids)
                seen_ids.update(ids)

                if args.trail > 0:
                    for tid, (x, y, _, _) in zip(ids, xywh):
                        trails[tid].append((int(x), int(y)))
                        pts = np.array(trails[tid], dtype=np.int32).reshape(-1, 1, 2)
                        cv2.polylines(annotated, [pts], False, id_color(tid), 2)

            # FPS / 현재 추적 수 / 누적 고유 ID 수 표시
            now = time.perf_counter()
            fps = 1.0 / (now - prev)
            prev = now
            cv2.putText(annotated,
                        f"FPS: {fps:.1f}  Tracking: {active}  Total IDs: {len(seen_ids)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow(window, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
