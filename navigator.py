"""표지판 내비게이션 (PLAN.md): 카메라 검출 + 초음파 거리 -> 바퀴 속도, 카메라 각도, 화면 표시.

    SEARCH   : 몸은 멈춘 채 카메라를 좌우로 돌려 둘러본다 (정면 -> 왼쪽 -> 오른쪽 -> ...).
               정면에서 찾으면 APPROACH, 옆에서 찾으면 몸을 그쪽으로 돌림(FACE).
               다 둘러봐도 없으면 제자리에서 다시 둘러본다 (찾을 때까지 반복).
    FACE     : 카메라로 찾은 방향만큼 몸을 돌리고, 정면에서 다시 확인
    APPROACH : 표지판을 화면 가운데에 맞추며 다가가기 (60cm 안에서는 감속).
               카메라 상하 각도는 표지판이 화면 세로 가운데에 오도록 따라간다.
    도착     : 표지판이 정면에 있고 초음파 25cm 이내
    실행     : 화살표 좌/우 -> 90도 회전, 아래 -> 후진, 과녁 -> 몇 초 정지
               실행을 마치면 바로 SEARCH부터 반복 (방금 실행한 표지판은 잠시 무시)
    OBSTACLE : 표지판이 아닌 것이 25cm 안에 있음 -> 정지 후 대기

위쪽(전진) 화살표는 쓰지 않는다 (표지판 앞에서 전진하면 부딪히므로).
하드웨어와 분리돼 있어서 PC에서 가짜 검출/거리로 시험할 수 있다.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import config

# 표지판 -> 도착 후 할 행동
ACTIONS = {
    ("arrow", "LEFT"): "TURN_LEFT",
    ("arrow", "RIGHT"): "TURN_RIGHT",
    ("arrow", "DOWN"): "BACKWARD",
    ("target", "target"): "STOP",
    ("goal", "goal"): "STOP",
}

# 상태 -> OLED 큰 글씨
DISPLAY = {
    "SEARCH": "Search", "FACE": "Turn", "APPROACH": "Approach", "OBSTACLE": "Obstacle",
    "TURN_LEFT": "Left", "TURN_RIGHT": "Right", "BACKWARD": "Back", "STOP": "Stop",
}


@dataclass
class Command:
    left: int = 0       # 왼쪽 바퀴 (-255 ~ 255)
    right: int = 0      # 오른쪽 바퀴
    state: str = ""
    detail: str = ""    # OLED 작은 글씨
    pan: int = 90       # 카메라 좌우 서보 각도
    tilt: int = 0       # 카메라 상하 서보 각도

    @property
    def display(self):
        return DISPLAY[self.state]


def sign_name(sign):
    kind, label = sign
    return kind if kind == label else f"{kind} {label}"


def camera_right_deg(pan):
    """카메라가 정면에서 오른쪽으로 몇 도 돌아가 있는지 (실측: pan이 90보다 작으면 오른쪽)."""
    return (config.CAMERA_PAN - pan) * config.PAN_RIGHT_SIGN


class Navigator:
    def __init__(self):
        self.history = deque(maxlen=config.DEBOUNCE_WINDOW)
        self.cooldown = {}     # 표지판 -> 무시가 끝나는 시각
        self.sign = None       # 다가가는 중인 표지판 (kind, label)
        self.last_det = None   # 그 표지판을 마지막으로 본 검출 결과
        self.last_seen = 0.0
        self.last_seen_dist = None  # 마지막으로 봤을 때의 거리
        self.state = "SEARCH"
        self.since = None      # 현재 상태에 들어온 시각
        self.pan, self.tilt = config.CAMERA_PAN, config.CAMERA_TILT
        self.look_since = None  # 카메라를 마지막으로 움직인 시각
        self.scan_i = 0         # 지금 보고 있는 SCAN_PANS 순번
        self.face_sec, self.face_dir = 0.0, 0
        self.on_change = None  # 상태가 바뀔 때 호출 (로그용)

    # ---- 공통 ----
    def _set(self, state, now, reason=""):
        if state != self.state and self.on_change:
            self.on_change(self.state, state, reason)
        self.state, self.since = state, now

    def _cmd(self, left=0, right=0, detail=""):
        return Command(int(left), int(right), self.state, detail, int(self.pan), int(self.tilt))

    def _look(self, now, pan, tilt=None):
        """카메라를 돌린다. 돌아가며 흔들리는 동안(SCAN_SETTLE_SEC)의 화면은 쓰지 않는다."""
        tilt = self.tilt if tilt is None else tilt
        if (pan, tilt) != (self.pan, self.tilt):
            self.pan, self.tilt = pan, tilt
            self.history.clear()
            self.look_since = now

    def _observe(self, now, detections):
        """이번 프레임에서 가장 가까운(큰) 표지판을 기록하고, 최근 N프레임 중 M번 이상이면 확정."""
        self.cooldown = {s: t for s, t in self.cooldown.items() if t > now}
        self.frame = {}
        if now - self.look_since < config.SCAN_SETTLE_SEC:  # 카메라가 움직이는 중
            return None
        for d in detections:
            key = (d.kind, d.label)
            if key in ACTIONS and key not in self.cooldown:
                if key not in self.frame or d.area_ratio > self.frame[key].area_ratio:
                    self.frame[key] = d
        best = max(self.frame, key=lambda k: self.frame[k].area_ratio, default=None)
        self.history.append(best)
        counts = Counter(k for k in self.history if k)
        confirmed = [k for k, n in counts.items() if n >= config.DEBOUNCE_MIN_HITS and k in self.frame]
        return max(confirmed, key=lambda k: self.frame[k].area_ratio, default=None)

    def step(self, now, detections, frame_width, distance_cm, frame_height=None):
        """한 프레임 처리. distance_cm 이 None 이면 초음파 측정 실패."""
        self.width, self.height = frame_width, frame_height
        if self.since is None:
            self.since = self.look_since = now
        confirmed = self._observe(now, detections)
        handler = getattr(self, "_" + self.state.lower())
        return handler(now, distance_cm, confirmed)

    def _steer(self, det):
        return (det.center[0] - self.width / 2) / (self.width / 2)

    def _start_search(self, now, reason=""):
        self.sign = None
        self.history.clear()
        self._set("SEARCH", now, reason)
        self.scan_i = 0
        self._look(now, config.SCAN_PANS[0], config.CAMERA_TILT)

    def _start_face(self, now, right_deg, reason):
        """몸을 right_deg 만큼 돌린다 (+ 오른쪽 / - 왼쪽). 카메라는 정면으로."""
        self.face_dir = 1 if right_deg > 0 else -1
        self.face_sec = abs(right_deg) / 90 * config.TURN_90_SEC
        self._look(now, config.CAMERA_PAN)
        self._set("FACE", now, reason)

    def _start_approach(self, now, sign):
        self.sign, self.last_det, self.last_seen = sign, self.frame[sign], now
        self.last_seen_dist = None
        self._set("APPROACH", now, sign_name(sign))

    def _finish(self, now):
        """행동을 마치면 방금 실행한 표지판은 잠시 무시하고 1단계(찾기)부터 다시."""
        self.cooldown[self.sign] = now + config.COOLDOWN_SEC
        self._start_search(now, "done")
        return self._cmd(detail="done")

    # ---- 상태별 ----
    def _search(self, now, dist, confirmed):
        if confirmed:
            if self.pan == config.CAMERA_PAN:  # 정면에서 찾음 -> 바로 다가가기
                self._start_approach(now, confirmed)
                return self._approach(now, dist, confirmed)
            # 옆에서 찾음 -> 카메라 방향 + 화면 속 위치만큼 몸을 돌림
            angle = camera_right_deg(self.pan) + self._steer(self.frame[confirmed]) * config.CAMERA_HFOV / 2
            self._start_face(now, angle, f"{sign_name(confirmed)} {angle:+.0f}deg")
            return self._face(now, dist, None)

        if now - self.look_since >= config.SCAN_SETTLE_SEC + config.SCAN_DWELL_SEC:
            # 다음 방향으로. 다 둘러봐도 없으면 처음(정면)부터 다시
            self.scan_i = (self.scan_i + 1) % len(config.SCAN_PANS)
            self._look(now, config.SCAN_PANS[self.scan_i])
        return self._cmd(detail=f"look {camera_right_deg(self.pan):+d}")

    def _face(self, now, dist, confirmed):
        if now - self.since >= self.face_sec:
            self._start_search(now)  # 정면부터 다시 확인
            return self._cmd(detail="look")
        spin = self.face_dir * config.SPEED_TURN
        return self._cmd(spin, -spin, "to sign")

    def _track_tilt(self, det):
        """표지판이 화면 위/아래로 벗어나지 않게 카메라 상하 각도를 조금씩 따라감 (값이 클수록 위)."""
        if not self.height:
            return
        dy = (det.center[1] - self.height / 2) / (self.height / 2)
        if abs(dy) > config.TILT_DEADZONE:
            step = config.TILT_STEP if dy < 0 else -config.TILT_STEP
            self.tilt = max(config.TILT_MIN, min(config.TILT_MAX, self.tilt + step))

    def _approach(self, now, dist, confirmed):
        det = self.frame.get(self.sign)
        if det:
            self.last_det, self.last_seen, self.last_seen_dist = det, now, dist
            self._track_tilt(det)
        lost_for = now - self.last_seen
        was_close = self.last_seen_dist is not None and self.last_seen_dist <= config.SLOW_CM
        lost_limit = config.CLOSE_LOST_SEC if was_close else config.LOST_SEC
        steer = self._steer(self.last_det)
        dist_text = f"{dist:.0f}cm" if dist is not None else "--cm"
        detail = f"{sign_name(self.sign)} {dist_text}"

        if dist is not None and dist <= config.ARRIVE_CM:
            if abs(steer) <= config.CENTER_TOL and lost_for <= lost_limit:
                action = ACTIONS[self.sign]
                self._set(action, now, f"arrived {dist:.0f}cm")
                return getattr(self, "_" + action.lower())(now, dist, None)  # 같은 프레임에 바로 시작
            if not det:  # 앞에 있는 건 표지판이 아님
                self._set("OBSTACLE", now, f"{dist:.0f}cm")
                return self._cmd(detail=dist_text)
            # 표지판은 보이는데 정면이 아님 -> 제자리에서 돌려 맞추기 (아래 회전 코드)

        if lost_for > lost_limit:
            self._start_search(now, "lost sign")
            return self._cmd(detail="lost")

        if not det:  # 잠깐 놓침: 가까우면(화면 밖으로 벗어났을 수 있음) 천천히 직진, 아니면 멈춰서 보기
            if dist is not None and dist <= config.SLOW_CM and abs(steer) <= config.CENTER_TOL:
                return self._cmd(config.SPEED_SLOW, config.SPEED_SLOW, detail)
            return self._cmd(detail=detail)

        if abs(steer) > config.SPIN_TOL or (dist is not None and dist <= config.ARRIVE_CM):
            spin = config.SPEED_SCAN if steer > 0 else -config.SPEED_SCAN
            return self._cmd(spin, -spin, detail)
        speed = config.SPEED_SLOW if dist is None or dist <= config.SLOW_CM else config.SPEED_NORMAL
        return self._cmd(speed * (1 + config.STEER_GAIN * steer), speed * (1 - config.STEER_GAIN * steer), detail)

    def _obstacle(self, now, dist, confirmed):
        if self.sign in self.frame:  # 다시 보이면 이어서 다가가기 (도착 판정은 거기서)
            self._set("APPROACH", now, "sign visible")
            return self._approach(now, dist, confirmed)
        if dist is None or dist > config.ARRIVE_CM + 5:  # 치워짐
            self._start_search(now, "clear")
            return self._cmd(detail="clear")
        return self._cmd(detail=f"{dist:.0f}cm")

    def _turn(self, now, direction):
        if now - self.since >= config.TURN_90_SEC:
            return self._finish(now)
        return self._cmd(direction * config.SPEED_TURN, -direction * config.SPEED_TURN, "90deg")

    def _turn_left(self, now, dist, confirmed):
        return self._turn(now, -1)

    def _turn_right(self, now, dist, confirmed):
        return self._turn(now, 1)

    def _backward(self, now, dist, confirmed):
        if now - self.since >= config.MOVE_SEC:
            return self._finish(now)
        return self._cmd(-config.SPEED_NORMAL, -config.SPEED_NORMAL, "back")

    def _stop(self, now, dist, confirmed):
        if now - self.since >= config.STOP_SEC:
            return self._finish(now)
        return self._cmd(detail=sign_name(self.sign))
