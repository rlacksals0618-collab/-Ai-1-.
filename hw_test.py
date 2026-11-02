"""하드웨어 하나씩 확인 (PLAN.md 2단계). 모터 시험은 반드시 바퀴를 바닥에서 띄우고!

    python3 hw_test.py sensor            # 초음파 거리 10초간 출력 + OLED 표시
    python3 hw_test.py oled Left         # OLED에 글자 표시 (다른 글자를 띄울 때까지 남아 있음)
    python3 hw_test.py motors            # 바퀴 하나씩 앞으로 1초 (어느 바퀴가 어느 방향인지 확인)
    python3 hw_test.py drive             # 전진 1초 -> 후진 1초
    python3 hw_test.py turn              # 제자리 왼쪽 90도 (config.TURN_90_SEC 맞추기)
    python3 hw_test.py turn --sec 1.2    # 회전 시간을 바꿔서 시험
    python3 hw_test.py servo 90 30       # 카메라 서보 (좌우, 상하) 각도
    python3 hw_test.py photo             # 카메라 사진 /tmp/rasbot_photo.jpg 저장
    python3 hw_test.py center            # 로봇 정면 중심선에 세운 표지판으로 카메라 정면 각도(CAMERA_PAN) 찾기
    python3 hw_test.py center 84         # 서보 84도에서 다시 확인
"""
import argparse
import time

import config
from hardware import LEFT_MOTORS, RIGHT_MOTORS, Hardware

MOTOR_NAMES = {0: "왼쪽 앞", 1: "왼쪽 뒤", 2: "오른쪽 앞", 3: "오른쪽 뒤"}
MOTOR_LABELS = {0: "L front", 1: "L rear", 2: "R front", 3: "R rear"}  # OLED는 한글이 안 나옴


def run_for(hw, left, right, sec, label):
    print(f"{label}: 왼쪽 {left:+d}, 오른쪽 {right:+d}, {sec}초")
    hw.show(label)
    hw.drive(left, right)
    time.sleep(sec)
    hw.stop()


def center_camera(hw, pan):
    """로봇 정면 중심선 위의 표지판이 화면 가운데에서 얼마나 벗어나 보이는지 재서, 실제 정면 각도를 계산."""
    import cv2

    from vision_demo import Vision

    hw.servo(1, pan)
    hw.servo(2, config.CAMERA_TILT)
    time.sleep(0.8)
    cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_V4L2)
    vision = Vision()
    offsets = []
    for i in range(40):
        ok, frame = cap.read()
        if not ok or i < 10:  # 자동 노출이 맞춰질 때까지 버림
            continue
        frame = cv2.resize(frame, (config.FRAME_WIDTH, frame.shape[0] * config.FRAME_WIDTH // frame.shape[1]))
        signs = [d for d in vision.detect(frame) if d.kind in ("arrow", "target", "goal")]
        if signs:
            d = max(signs, key=lambda s: s.area_ratio)
            offsets.append((d.center[0] - frame.shape[1] / 2) / (frame.shape[1] / 2) * config.CAMERA_HFOV / 2)
    cap.release()
    if len(offsets) < 5:
        print(f"표지판을 찾지 못했습니다 ({len(offsets)}/30 프레임). 표지판이 화면에 다 들어오게 세워 주세요.")
        return ("Center?", "no sign")
    off = sorted(offsets)[len(offsets) // 2]  # 오른쪽으로 몇 도 벗어나 보이는지
    # 표지판은 로봇 정면에 있으니, 화면에서 오른쪽에 보이면 카메라가 그만큼 왼쪽을 보고 있는 것
    new_pan = round(pan - off * config.PAN_RIGHT_SIGN)
    print(f"서보 {pan}도: 표지판이 화면 가운데에서 {'오른쪽' if off > 0 else '왼쪽'}으로 {abs(off):.1f}도 ({len(offsets)}프레임)")
    if abs(off) <= 2:
        print(f"-> 서보 {pan}도가 정면입니다. config.CAMERA_PAN = {pan}")
        return ("Center", f"pan {pan} OK")
    print(f"-> config.CAMERA_PAN 을 {new_pan} 으로 (python3 hw_test.py center {new_pan} 으로 다시 확인)")
    return ("Center", f"try pan {new_pan}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("test", choices=["sensor", "oled", "motors", "drive", "turn", "servo", "photo", "center"])
    parser.add_argument("values", nargs="*")
    parser.add_argument("--speed", type=int, default=config.SPEED_NORMAL)
    parser.add_argument("--sec", type=float, default=None)
    args = parser.parse_args()

    if args.test == "photo":
        import cv2

        cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_V4L2)
        for _ in range(15):  # 자동 노출이 맞춰질 때까지 버림
            ok, frame = cap.read()
        cap.release()
        cv2.imwrite("/tmp/rasbot_photo.jpg", frame)
        print("저장: /tmp/rasbot_photo.jpg")
        return

    hw = Hardware()
    final = ("Off", args.test)  # 시험이 끝난 뒤 화면에 남길 글자
    try:
        if args.test == "sensor":
            end = time.monotonic() + 10
            while time.monotonic() < end:
                d = hw.distance_cm()
                text = f"{d:.1f}cm" if d is not None else "측정 실패"
                print(text)
                hw.show("Distance", text.replace("측정 실패", "--"))
                time.sleep(0.2)
        elif args.test == "oled":
            final = (args.values[0] if args.values else "Hello", " ".join(args.values[1:]))  # 화면에 계속 남김
        elif args.test == "motors":
            for m in LEFT_MOTORS + RIGHT_MOTORS:
                print(f"{MOTOR_NAMES[m]} 바퀴 앞으로 1초")
                hw.show(f"Motor {m}", MOTOR_LABELS[m])
                hw._motor(m, args.speed)
                time.sleep(1)
                hw._motor(m, 0)
                time.sleep(0.5)
        elif args.test == "drive":
            run_for(hw, args.speed, args.speed, args.sec or 1, "Forward")
            time.sleep(0.5)
            run_for(hw, -args.speed, -args.speed, args.sec or 1, "Back")
        elif args.test == "turn":
            sec = args.sec or config.TURN_90_SEC
            run_for(hw, -config.SPEED_TURN, config.SPEED_TURN, sec, "Left")
            print("90도보다 덜 돌았으면 --sec 를 늘리고, 더 돌았으면 줄여서 다시 시험 -> config.TURN_90_SEC 에 적기")
        elif args.test == "center":
            final = center_camera(hw, int(args.values[0]) if args.values else config.CAMERA_PAN)
        elif args.test == "servo":
            pan = int(args.values[0]) if args.values else 90
            tilt = int(args.values[1]) if len(args.values) > 1 else None
            hw.servo(1, pan)
            if tilt is not None:
                hw.servo(2, tilt)
            print(f"서보 좌우 {pan}, 상하 {tilt}")
            time.sleep(1)
    finally:
        hw.close(*final)


if __name__ == "__main__":
    main()
