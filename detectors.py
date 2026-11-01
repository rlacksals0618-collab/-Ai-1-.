"""카메라 프레임에서 목표 그림 / GOAL / 화살표 / 손동작을 찾는 검출기들."""
from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np

import config


@dataclass
class Detection:
    kind: str          # "target" | "goal" | "arrow" | "hand"
    label: str         # 세부 정보 (화살표/손 방향 등)
    center: tuple[float, float]
    area_ratio: float  # 화면 대비 면적 (가까울수록 큼)
    points: np.ndarray | None = None  # 그리기용 윤곽/랜드마크
    score: float = 0.0


def load_image(path, flags=cv2.IMREAD_COLOR):
    """cv2.imread는 윈도우에서 한글 경로를 못 읽어서 우회."""
    return cv2.imdecode(np.fromfile(str(path), np.uint8), flags)


def save_image(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(path.suffix, img)[1].tofile(str(path))


def resize_max(img, max_size):
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


class TemplateDetector:
    """등록된 그림(target, goal)을 ORB 특징점 + 호모그래피로 찾는다."""

    def __init__(self, names=config.TEMPLATE_NAMES):
        self.orb = cv2.ORB_create(config.ORB_FEATURES)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.templates = {}
        for name in names:
            path = config.TEMPLATES_DIR / f"{name}.png"
            if not path.exists():
                print(f"[그림] {path.name} 없음 -> '{name}' 인식 건너뜀")
                continue
            img = load_image(path, cv2.IMREAD_GRAYSCALE)
            img = resize_max(img, config.TEMPLATE_MAX_SIZE)
            kp, des = self.orb.detectAndCompute(img, None)
            if des is None or len(kp) < config.MIN_GOOD_MATCHES:
                print(f"[그림] '{name}' 특징점 부족({len(kp)}개) - 무늬/글자가 더 많은 그림을 쓰세요")
                continue
            h, w = img.shape
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            self.templates[name] = (kp, des, corners)
            print(f"[그림] '{name}' 등록 (특징점 {len(kp)}개)")

    def detect(self, gray):
        if not self.templates:
            return []
        kp_f, des_f = self.orb.detectAndCompute(gray, None)
        if des_f is None or len(kp_f) < config.MIN_GOOD_MATCHES:
            return []
        h, w = gray.shape
        found = []
        for name, (kp_t, des_t, corners) in self.templates.items():
            pairs = self.matcher.knnMatch(des_t, des_f, k=2)
            good = [p[0] for p in pairs
                    if len(p) == 2 and p[0].distance < config.MATCH_RATIO * p[1].distance]
            if len(good) < config.MIN_GOOD_MATCHES:
                continue
            src = np.float32([kp_t[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp_f[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if H is None or int(mask.sum()) < config.MIN_INLIERS:
                continue
            poly = cv2.perspectiveTransform(corners, H)
            # 찌그러진(뒤집히거나 꼬인) 사각형은 오검출
            if not cv2.isContourConvex(poly.astype(np.int32)):
                continue
            area_ratio = cv2.contourArea(poly) / (w * h)
            if area_ratio < config.MIN_TEMPLATE_AREA_RATIO:
                continue
            cx, cy = poly.reshape(-1, 2).mean(axis=0)
            found.append(Detection(name, name, (float(cx), float(cy)), area_ratio,
                                   poly.astype(np.int32), float(mask.sum())))
        return found


def arrow_direction(contour):
    """화살표 윤곽이면 LEFT/RIGHT/UP/DOWN, 아니면 None.

    화살표를 다각형으로 근사하면 촉 쪽 끝에는 꼭짓점이 1개(뾰족),
    꼬리 쪽 끝에는 2개(평평)가 있다는 점으로 방향을 정한다.
    """
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    if hull_area <= 0:
        return None
    solidity = cv2.contourArea(contour) / hull_area
    if not 0.45 < solidity < 0.9:  # 사각형/원(≈1)은 제외
        return None
    poly = cv2.approxPolyDP(contour, 0.03 * cv2.arcLength(contour, True), True).reshape(-1, 2)
    if not 6 <= len(poly) <= 9:
        return None
    _, _, bw, bh = cv2.boundingRect(poly)
    for axis, size, neg, pos in ((0, bw, "LEFT", "RIGHT"), (1, bh, "UP", "DOWN")):
        vals = poly[:, axis]
        tol = 0.1 * size
        n_min = int(np.sum(vals <= vals.min() + tol))
        n_max = int(np.sum(vals >= vals.max() - tol))
        if n_min == 1 and n_max >= 2:
            return neg
        if n_max == 1 and n_min >= 2:
            return pos
    return None


class ArrowDetector:
    """파란색 영역 중 화살표 모양을 찾아 방향을 판단한다."""

    def __init__(self):
        self.low = np.array(config.ARROW_HSV_LOW, np.uint8)
        self.high = np.array(config.ARROW_HSV_HIGH, np.uint8)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.last_mask = None  # 색 범위 조정용 디버그 화면

    def detect(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.low, self.high)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        self.last_mask = mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = mask.shape
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
            area = cv2.contourArea(c)
            if area < config.ARROW_MIN_AREA_RATIO * w * h:
                break
            direction = arrow_direction(c)
            if direction is None:
                continue
            m = cv2.moments(c)
            center = (m["m10"] / m["m00"], m["m01"] / m["m00"])
            return [Detection("arrow", direction, center, area / (w * h), c)]
        return []


# 손가락 (끝, 가운데 마디) 랜드마크 번호: 검지, 중지, 약지, 새끼
FINGERS = ((8, 6), (12, 10), (16, 14), (20, 18))


def classify_hand(pts):
    """랜드마크 21개 -> "PALM"(손바닥 펼침) / "LEFT" / "RIGHT"(검지로 가리킴) / None."""
    wrist = pts[0]
    extended = [np.linalg.norm(pts[tip] - wrist) > np.linalg.norm(pts[pip] - wrist)
                for tip, pip in FINGERS]
    if all(extended):
        return "PALM"
    if extended[0] and not any(extended[1:]):
        dx, dy = pts[8] - pts[5]  # 검지 뿌리 -> 끝
        if abs(dx) > 1.2 * abs(dy):
            return "RIGHT" if dx > 0 else "LEFT"
    return None


class HandGestureDetector:
    """MediaPipe 손 랜드마크로 손동작을 판단한다. 방향은 로봇 카메라 화면 기준."""

    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        if not config.HAND_MODEL_PATH.exists():
            print("[손동작] 모델 다운로드 중...")
            config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(config.HAND_MODEL_URL, config.HAND_MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            # 경로 대신 바이트로 전달: 윈도우 한글 경로에서 mediapipe가 파일을 못 엶
            base_options=BaseOptions(model_asset_buffer=config.HAND_MODEL_PATH.read_bytes()),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
        )
        self.mp = mp
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.t0 = time.monotonic()
        self.last_ts = -1

    def detect(self, bgr):
        h, w = bgr.shape[:2]
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        ts = max(int((time.monotonic() - self.t0) * 1000), self.last_ts + 1)
        self.last_ts = ts
        result = self.landmarker.detect_for_video(image, ts)
        if not result.hand_landmarks:
            return []
        pts = np.array([[p.x * w, p.y * h] for p in result.hand_landmarks[0]])
        label = classify_hand(pts)
        if label is None:
            return []
        x, y, bw, bh = cv2.boundingRect(pts.astype(np.int32))
        return [Detection("hand", label, (x + bw / 2, y + bh / 2), bw * bh / (w * h), pts)]

    def close(self):
        self.landmarker.close()
