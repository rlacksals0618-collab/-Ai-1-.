"""라즈봇 V2(Yahboom) 하드웨어: 확장보드(I2C 0x2B)의 모터·서보·초음파 + OLED 화면(0x3C).

라즈베리파이가 아니거나 I2C가 없으면 FakeHardware로 대신해서 PC에서도 robot.py를 돌려볼 수 있다.
"""
from __future__ import annotations

import statistics
import time

import config

BOARD_ADDR = 0x2B
OLED_ADDR = 0x3C

REG_MOTOR = 0x01       # [모터 번호, 방향(0 앞/1 뒤), 속도 0~255]
REG_SERVO = 0x02       # [서보 번호(1 좌우, 2 상하), 각도]
REG_ULTRASONIC = 0x07  # [1 켜기 / 0 끄기]
REG_DIST_LOW = 0x1A    # 거리(mm) 하위 바이트
REG_DIST_HIGH = 0x1B   # 거리(mm) 상위 바이트

LEFT_MOTORS = (0, 1)   # 왼쪽 앞, 왼쪽 뒤
RIGHT_MOTORS = (2, 3)  # 오른쪽 앞, 오른쪽 뒤


class Oled:
    """SSD1306 128x32(또는 64) OLED. 큰 글씨 한 줄 + 작은 글씨 한 줄."""

    WIDTH = 128

    def __init__(self, bus, height=config.OLED_HEIGHT):
        from PIL import ImageFont

        self.bus = bus
        self.height = height
        self.last = None
        try:
            self.big = ImageFont.load_default(size=18)
            self.small = ImageFont.load_default(size=10)
        except TypeError:  # 오래된 Pillow
            self.big = self.small = ImageFont.load_default()
        self._cmd(0xAE, 0xD5, 0x80, 0xA8, height - 1, 0xD3, 0x00, 0x40, 0x8D, 0x14,
                  0x20, 0x00, 0xA1, 0xC8, 0xDA, 0x02 if height == 32 else 0x12,
                  0x81, 0xCF, 0xD9, 0xF1, 0xDB, 0x40, 0xA4, 0xA6, 0x2E, 0xAF)

    def _cmd(self, *cmds):
        for c in cmds:
            self.bus.write_byte_data(OLED_ADDR, 0x00, c)

    def show(self, title, detail=""):
        if (title, detail) == self.last:  # 바뀔 때만 다시 그림
            return
        self.last = (title, detail)
        from PIL import Image, ImageDraw

        img = Image.new("1", (self.WIDTH, self.height))
        draw = ImageDraw.Draw(img)
        draw.text((0, -2), title, fill=1, font=self.big)
        draw.text((0, self.height - 11), detail, fill=1, font=self.small)
        px = img.load()
        data = []
        for page in range(self.height // 8):
            for x in range(self.WIDTH):
                data.append(sum(1 << bit for bit in range(8) if px[x, page * 8 + bit]))
        self._cmd(0x21, 0, self.WIDTH - 1, 0x22, 0, self.height // 8 - 1)
        for i in range(0, len(data), 16):
            self.bus.write_i2c_block_data(OLED_ADDR, 0x40, data[i:i + 16])


class Hardware:
    def __init__(self, motors_enabled=True):
        from smbus2 import SMBus

        self.bus = SMBus(1)
        self.motors_enabled = motors_enabled
        self.oled = None
        try:
            self.oled = Oled(self.bus)
        except OSError as e:
            print(f"[OLED] 사용 안 함: {e}")
        self._write(REG_ULTRASONIC, [1])
        self.dists = []
        self.stop()

    def _write(self, reg, data):
        try:
            self.bus.write_i2c_block_data(BOARD_ADDR, reg, data)
        except OSError:  # I2C가 가끔 한 번씩 실패함 -> 한 번 더
            time.sleep(0.002)
            self.bus.write_i2c_block_data(BOARD_ADDR, reg, data)

    def _motor(self, motor_id, speed):
        speed = int(max(-255, min(255, speed)))
        self._write(REG_MOTOR, [motor_id, 1 if speed < 0 else 0, abs(speed)])

    def drive(self, left, right):
        """왼쪽/오른쪽 바퀴 속도 (-255 ~ 255, +가 앞)."""
        if not self.motors_enabled:
            left = right = 0
        for m in LEFT_MOTORS:
            self._motor(m, left * config.LEFT_MOTOR_SIGN)
        for m in RIGHT_MOTORS:
            self._motor(m, right * config.RIGHT_MOTOR_SIGN)

    def stop(self):
        self.drive(0, 0)

    def servo(self, servo_id, angle):
        self._write(REG_SERVO, [servo_id, int(max(0, min(180, angle)))])

    def distance_cm(self):
        """초음파 거리(cm). 최근 몇 번의 중간값. 측정 실패면 None."""
        try:
            high = self.bus.read_i2c_block_data(BOARD_ADDR, REG_DIST_HIGH, 1)[0]
            low = self.bus.read_i2c_block_data(BOARD_ADDR, REG_DIST_LOW, 1)[0]
        except OSError:
            return None
        cm = (high << 8 | low) / 10
        if not 0 < cm <= config.DIST_MAX_CM:
            self.dists.clear()
            return None
        self.dists = (self.dists + [cm])[-config.DIST_MEDIAN:]
        return statistics.median(self.dists)

    def show(self, title, detail=""):
        if self.oled:
            self.oled.show(title, detail)

    def close(self, title="Off", detail=""):
        """바퀴 정지, 초음파 끄기. 화면은 끄지 않고 마지막 글자(기본 'Off')를 남겨 둔다."""
        self.stop()
        self._write(REG_ULTRASONIC, [0])
        self.show(title, detail)
        self.bus.close()


class FakeHardware:
    """하드웨어 없는 PC용. 명령을 기억만 하고 거리는 지정한 값(기본 측정 실패)을 돌려준다."""

    def __init__(self, distance=None):
        self.distance = distance
        self.left = self.right = 0
        self.screen = ("", "")

    def drive(self, left, right):
        self.left, self.right = left, right

    def stop(self):
        self.drive(0, 0)

    def servo(self, servo_id, angle):
        pass

    def distance_cm(self):
        return self.distance

    def show(self, title, detail=""):
        self.screen = (title, detail)

    def close(self, title="Off", detail=""):
        self.stop()
        self.show(title, detail)


def open_hardware(motors_enabled=True):
    try:
        return Hardware(motors_enabled)
    except (ImportError, FileNotFoundError, OSError) as e:
        print(f"[하드웨어] 라즈봇 보드를 쓸 수 없어 가짜 하드웨어로 실행: {e}")
        return FakeHardware()
