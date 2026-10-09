"""YOLO26 웹캠 실시간 인스턴스 세그멘테이션

설치:  pip install -U ultralytics
실행:  python webcam_segment.py
종료:  영상 창에서 'q' 또는 ESC
"""
import argparse
import time

import cv2
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="YOLO26 webcam instance segmentation")
    p.add_argument("--model", default="yolo26n-seg.pt",
                   help="yolo26n/s/m/l/x-seg.pt (처음 실행 시 자동 다운로드)")
    p.add_argument("--source", type=int, default=0, help="웹캠 인덱스")
    p.add_argument("--conf", type=float, default=0.25, help="신뢰도 임계값")
    p.add_argument("--imgsz", type=int, default=640, help="추론 이미지 크기")
    p.add_argument("--device", default=None, help="예: 0 (GPU), cpu")
    p.add_argument("--no-boxes", action="store_true", help="박스 없이 마스크만 표시")
    return p.parse_args()


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

    window = "YOLO26 Webcam Segmentation"
    prev = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("프레임을 읽지 못했습니다.")
                break

            # retina_masks=True: 마스크를 원본 해상도로 출력해 경계가 더 매끄럽다
            results = model.predict(frame, conf=args.conf, imgsz=args.imgsz,
                                    device=args.device, retina_masks=True,
                                    verbose=False)
            result = results[0]
            annotated = result.plot(boxes=not args.no_boxes)  # 마스크 + 박스 + 라벨

            # FPS 및 탐지 객체 수 표시
            now = time.perf_counter()
            fps = 1.0 / (now - prev)
            prev = now
            n = 0 if result.masks is None else len(result.masks)
            cv2.putText(annotated, f"FPS: {fps:.1f}  Objects: {n}",
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
