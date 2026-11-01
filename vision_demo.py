"""인식 데모: 카메라 화면에 검출 결과와 로봇이 할 행동을 표시한다. (모터 미연결)

    python vision_demo.py                      # 카메라
    python vision_demo.py --image test.jpg     # 사진 한 장으로 확인
    python vision_demo.py --headless           # 화면 없이 행동만 출력 (SSH 접속 시)

키: q = 종료, m = 화살표 색 마스크 보기
"""
import argparse
import sys
import time

import cv2
import numpy as np

import config
from decision import Decider
from detectors import ArrowDetector, HandGestureDetector, TemplateDetector, load_image, resize_max

COLORS = {"target": (0, 255, 255), "goal": (0, 200, 0), "arrow": (255, 120, 0), "hand": (255, 0, 255)}
ACTION_COLORS = {"STOP": (0, 0, 255), "TURN_LEFT": (255, 120, 0), "TURN_RIGHT": (255, 120, 0),
                 "HOLD": (0, 200, 200), "APPROACH": (0, 200, 0), "SEARCH": (128, 128, 128)}


class ImageSource:
    """사진 한 장을 카메라처럼 반복해서 돌려준다."""

    def __init__(self, path):
        self.img = load_image(path)
        if self.img is None:
            sys.exit(f"이미지를 읽을 수 없습니다: {path}")

    def read(self):
        return True, self.img.copy()

    def release(self):
        pass


def draw(frame, detections, action, fps):
    for d in detections:
        color = COLORS[d.kind]
        if d.kind in ("target", "goal"):
            cv2.polylines(frame, [d.points], True, color, 3)
        elif d.kind == "arrow":
            cv2.drawContours(frame, [d.points], -1, color, 3)
        else:
            for x, y in d.points.astype(int):
                cv2.circle(frame, (x, y), 3, color, -1)
        x, y, _, _ = cv2.boundingRect(d.points.astype(np.int32).reshape(-1, 2))
        text = d.kind if d.kind == d.label else f"{d.kind} {d.label}"
        cv2.putText(frame, text, (x, max(y - 8, 55)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    w = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
    text = f"{action.name}  steer={action.steer:+.2f}  ({action.reason})"
    cv2.putText(frame, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ACTION_COLORS[action.name], 2)
    cv2.putText(frame, f"{fps:.0f} fps", (w - 90, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    if action.name == "APPROACH":  # 목표 방향 표시
        cx = int(w / 2 + action.steer * w / 2)
        cv2.arrowedLine(frame, (w // 2, frame.shape[0] - 20), (cx, frame.shape[0] - 80), (0, 200, 0), 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    parser.add_argument("--image", help="카메라 대신 사진으로 테스트")
    parser.add_argument("--no-gesture", action="store_true", help="손동작 인식 끄기 (속도 향상)")
    parser.add_argument("--headless", action="store_true", help="화면 없이 콘솔 출력만")
    parser.add_argument("--frames", type=int, default=0, help="N프레임 후 종료 (0 = 무한)")
    args = parser.parse_args()

    source = ImageSource(args.image) if args.image else cv2.VideoCapture(args.camera)
    if not args.image and not source.isOpened():
        sys.exit(f"카메라 {args.camera}를 열 수 없습니다.")

    templates = TemplateDetector()
    arrows = ArrowDetector()
    hand = None
    if config.GESTURE_ENABLED and not args.no_gesture:
        try:
            hand = HandGestureDetector()
        except Exception as e:  # mediapipe 미설치/모델 다운로드 실패 시에도 나머지는 동작
            print(f"[손동작] 비활성화: {e}")
    decider = Decider()

    show_mask = False
    last_action = None
    fps = 0.0
    count = 0
    try:
        while not args.frames or count < args.frames:
            t = time.perf_counter()
            ok, frame = source.read()
            if not ok:
                break
            frame = resize_max(frame, config.FRAME_WIDTH)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = templates.detect(gray) + arrows.detect(frame)
            if hand:
                detections += hand.detect(frame)
            action = decider.update(detections, frame.shape[1])
            count += 1

            if action.name != last_action:
                print(f"[행동] {action.name:10s} steer={action.steer:+.2f}  ({action.reason})")
                last_action = action.name
            fps = 0.9 * fps + 0.1 / max(time.perf_counter() - t, 1e-6)

            if args.headless:
                continue
            draw(frame, detections, action, fps)
            cv2.imshow("rasbot vision", frame)
            if show_mask:
                cv2.imshow("arrow mask", arrows.last_mask)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("m"):
                show_mask = not show_mask
                if not show_mask:
                    cv2.destroyWindow("arrow mask")
    finally:
        source.release()
        if hand:
            hand.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
