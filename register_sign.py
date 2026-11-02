"""그림 표지판(GOAL 등)을 매칭용 이미지로 등록한다. 여러 장 등록할수록 인식이 안정적이다.

    python register_sign.py                        # 카메라로 인쇄한 GOAL 촬영해서 추가
    python register_sign.py --image goal_photo.jpg # 파일에서 추가

등록 이미지는 assets/templates/<이름>/ 에 쌓이고, 그중 하나라도 맞으면 인식한다.
과녁과 화살표는 모양으로 인식하므로 등록할 필요 없다.

카메라 모드: SPACE = 촬영, q = 취소. 촬영 후 마우스로 표지판 영역을 드래그하고 ENTER
(드래그 없이 ENTER면 전체 사용).
"""
import argparse
import sys
import time

import cv2

import config
from detectors import load_image, resize_max, save_image

GOOD_KEYPOINTS = 150


def capture(camera_index):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        sys.exit(f"카메라 {camera_index}를 열 수 없습니다.")
    frame = None
    while True:
        ok, img = cap.read()
        if not ok:
            break
        view = img.copy()
        cv2.putText(view, "SPACE: capture  q: quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("register", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            frame = img
            break
        if key == ord("q"):
            break
    cap.release()
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="goal", choices=config.TEMPLATE_NAMES)
    parser.add_argument("--image", help="카메라 대신 사용할 이미지 파일")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    parser.add_argument("--no-crop", action="store_true", help="영역 선택 없이 전체 사용")
    args = parser.parse_args()

    img = load_image(args.image) if args.image else capture(args.camera)
    if img is None:
        sys.exit("이미지가 없어 등록을 취소합니다.")

    if not args.no_crop:
        view = resize_max(img, 900)
        scale = img.shape[1] / view.shape[1]
        x, y, w, h = cv2.selectROI("select area (ENTER)", view, showCrosshair=False)
        if w > 0 and h > 0:
            img = img[int(y * scale):int((y + h) * scale), int(x * scale):int((x + w) * scale)]
    cv2.destroyAllWindows()

    gray = resize_max(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), config.TEMPLATE_MAX_SIZE)
    n = len(cv2.ORB_create(config.ORB_FEATURES).detect(gray, None))
    path = config.TEMPLATES_DIR / args.name / f"{time.strftime('%Y%m%d_%H%M%S')}.png"
    save_image(path, img)
    print(f"등록 완료: {path} (특징점 {n}개)")
    if n < GOOD_KEYPOINTS:
        print("경고: 특징점이 적어 인식이 불안정할 수 있습니다. 선/글자/무늬가 많은 그림을 권장합니다.")


if __name__ == "__main__":
    main()
