"""라즈봇 메인: 카메라로 화살표/과녁을 찾아 다가가고, 25cm에서 신호대로 움직인 뒤
다시 처음부터 찾기를 반복한다 (PLAN.md).

    python3 robot.py --dry-run     # 바퀴는 멈춘 채 인식·거리·화면 표시만 (처음엔 이걸로 확인)
    python3 robot.py               # 실제 주행
    python3 robot.py --save 1      # 1초마다 인식 화면과 원본을 /tmp/rasbot/ 에 저장 (원격 확인용)
    python3 robot.py --show        # 모니터/VNC가 있으면 인식 화면 띄우기

Ctrl+C 로 종료하면 바퀴를 멈추고 초음파/화면을 끈다.
"""
import argparse
import os
import sys
import threading
import time
from pathlib import Path

import cv2

import config
from hardware import FakeHardware, open_hardware
from navigator import Navigator
from vision_demo import ImageSource, Vision, describe, draw

STATE_COLORS = {"STOP": (0, 0, 255), "OBSTACLE": (0, 0, 255), "APPROACH": (0, 200, 0),
                "SEARCH": (160, 160, 160)}


class Camera:
    """최신 프레임만 유지하는 카메라. 처리하는 동안 쌓인 옛 프레임 때문에 반응이 늦어지는 걸 막는다."""

    def __init__(self, index):
        backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened():
            sys.exit(f"카메라 {index}를 열 수 없습니다.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.frame, self.count, self.taken = None, 0, 0
        self.cond = threading.Condition()
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                with self.cond:
                    self.frame, self.count = frame, self.count + 1
                    self.cond.notify_all()

    def read(self):
        """아직 처리하지 않은 최신 프레임. 0.5초 안에 새 프레임이 없으면 (False, None)."""
        with self.cond:
            if not self.cond.wait_for(lambda: self.count > self.taken, timeout=0.5):
                return False, None
            self.taken = self.count
            return True, self.frame

    def release(self):
        self.running = False
        self.cap.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="바퀴를 움직이지 않음")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    parser.add_argument("--image", help="카메라 대신 사진 (PC 시험용)")
    parser.add_argument("--distance", type=float, help="가짜 하드웨어일 때 초음파 거리(cm)")
    parser.add_argument("--show", action="store_true", help="인식 화면 띄우기")
    parser.add_argument("--save", type=float, default=0, help="N초마다 인식 화면을 /tmp/rasbot/ 에 저장")
    parser.add_argument("--seconds", type=float, default=0, help="N초 후 자동 종료 (0 = 무한)")
    args = parser.parse_args()

    hw = open_hardware(motors_enabled=not args.dry_run)
    if isinstance(hw, FakeHardware):
        hw.distance = args.distance
    hw.servo(1, config.CAMERA_PAN)
    hw.servo(2, config.CAMERA_TILT)
    camera_angle = (config.CAMERA_PAN, config.CAMERA_TILT)

    camera = ImageSource(args.image) if args.image else Camera(args.camera)
    vision = Vision()
    nav = Navigator()
    nav.on_change = lambda old, new, why: print(f"[상태] {old} -> {new}  {why}")
    save_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "rasbot"
    if args.save:
        save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[시작] {'바퀴 정지(dry-run)' if args.dry_run else '주행'} 모드. Ctrl+C 로 종료")
    start = last_save = time.monotonic()
    stale_since = None
    fps = 0.0
    try:
        while not args.seconds or time.monotonic() - start < args.seconds:
            t = time.monotonic()
            ok, frame = camera.read()
            if not ok:  # 카메라가 멈추면 바퀴도 멈춤
                stale_since = stale_since or t
                hw.stop()
                hw.show("Camera?", "no frame")
                if t - stale_since > 5:
                    print("[오류] 카메라 프레임이 5초간 없음 -> 종료")
                    break
                time.sleep(0.05)
                continue
            stale_since = None

            frame = cv2.resize(frame, (config.FRAME_WIDTH, frame.shape[0] * config.FRAME_WIDTH // frame.shape[1]))
            detections = vision.detect(frame)
            dist = hw.distance_cm()
            cmd = nav.step(t, detections, frame.shape[1], dist, frame.shape[0])
            hw.drive(cmd.left, cmd.right)
            hw.show(cmd.display, cmd.detail)
            if (cmd.pan, cmd.tilt) != camera_angle:  # 카메라 각도는 바뀔 때만 보냄
                if cmd.pan != camera_angle[0]:
                    hw.servo(1, cmd.pan)
                if cmd.tilt != camera_angle[1]:
                    hw.servo(2, cmd.tilt)
                camera_angle = (cmd.pan, cmd.tilt)
            fps = 0.9 * fps + 0.1 / max(time.monotonic() - t, 1e-6)

            if args.show or (args.save and t - last_save >= args.save):
                if args.save and t - last_save >= args.save:  # 원본도 저장 (인식 문제를 나중에 재현하려고)
                    cv2.imwrite(str(save_dir / f"{t - start:07.1f}_raw.jpg"), frame)
                dist_text = f"{dist:.0f}cm" if dist is not None else "--"
                banner = f"{cmd.display} | {dist_text} | L{cmd.left:+d} R{cmd.right:+d} | cam {cmd.pan},{cmd.tilt}"
                draw(frame, detections, banner, fps, STATE_COLORS.get(cmd.state, (255, 120, 0)))
                if args.save and t - last_save >= args.save:
                    cv2.imwrite(str(save_dir / f"{t - start:07.1f}_{cmd.state}.jpg"), frame)
                    last_save = t
                    print(f"[{t - start:6.1f}s] {cmd.display:9s} {dist_text:>6s}  cam {cmd.pan:3d},{cmd.tilt:2d}"
                          f"  검출: {describe(detections)}")
                if args.show:
                    cv2.imshow("rasbot", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    except KeyboardInterrupt:
        pass
    finally:
        hw.close()
        camera.release()
        cv2.destroyAllWindows()
        print("[종료] 바퀴 정지")


if __name__ == "__main__":
    main()
