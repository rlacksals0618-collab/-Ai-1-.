"""카메라 프레임에서 과녁 / GOAL / 화살표 / 손동작을 찾는 검출기들."""
from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np

import config

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


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


def dark_mask(gray):
    """주변보다 어두운 곳(검은 잉크) = 255. 조명이 달라도 종이 대비로 판단한다."""
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                                 config.DARK_BLOCK_SIZE, config.DARK_OFFSET)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


class TemplateDetector:
    """등록된 그림(GOAL 등)을 ORB 특징점 + 호모그래피로 찾는다.

    assets/templates/<이름>/ 폴더 안의 이미지를 모두 쓰고, 그중 하나라도 맞으면 인식한다.
    """

    def __init__(self, names=config.TEMPLATE_NAMES):
        self.orb = cv2.ORB_create(config.ORB_FEATURES)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.templates = []  # (이름, 특징점, 디스크립터, 네 모서리)
        for name in names:
            folder = config.TEMPLATES_DIR / name
            files = sorted(p for p in folder.glob("*") if p.suffix.lower() in IMAGE_EXTS)
            if not files:
                print(f"[그림] {folder} 에 이미지 없음 -> '{name}' 인식 건너뜀")
            for path in files:
                img = resize_max(load_image(path, cv2.IMREAD_GRAYSCALE), config.TEMPLATE_MAX_SIZE)
                kp, des = self.orb.detectAndCompute(img, None)
                if des is None or len(kp) < config.MIN_GOOD_MATCHES:
                    print(f"[그림] {name}/{path.name} 특징점 부족({len(kp)}개) - 무늬/글자가 더 많은 그림을 쓰세요")
                    continue
                h, w = img.shape
                corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                self.templates.append((name, kp, des, corners))
                print(f"[그림] {name}/{path.name} 등록 (특징점 {len(kp)}개)")

    def detect(self, gray):
        if not self.templates:
            return []
        kp_f, des_f = self.orb.detectAndCompute(gray, None)
        if des_f is None or len(kp_f) < config.MIN_GOOD_MATCHES:
            return []
        h, w = gray.shape
        best = {}  # 이름별로 가장 잘 맞은 것 하나
        for name, kp_t, des_t, corners in self.templates:
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
            inliers = float(mask.sum())
            if name in best and best[name].score >= inliers:
                continue
            cx, cy = poly.reshape(-1, 2).mean(axis=0)
            best[name] = Detection(name, name, (float(cx), float(cy)), area_ratio,
                                   poly.astype(np.int32), inliers)
        return list(best.values())


class CrosshairDetector:
    """과녁: 흰 원 -> 검은 고리 -> 흰 띠 -> 검은 고리 구조를 찾는다.

    상하좌우 대칭 무늬라 특징점 매칭이 약해서, 고리 구조를 직접 확인한다.
    """

    # 안쪽 흰 원 반지름을 1로 둔 비율 (assets/originals/target.jpg 에서 측정)
    #   안쪽 검은 고리 1.00~1.26, 흰 띠 1.26~1.75, 바깥 검은 고리 1.75~2.01
    RINGS = ((1.13, True), (1.50, False), (1.88, True))  # (반지름 배율, 검은색이어야 하나)
    OUTER = 2.01
    N_RAYS = 16

    def detect(self, dark):
        contours, hierarchy = cv2.findContours(dark, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if hierarchy is None:
            return []
        h, w = dark.shape
        best = None
        for c, (_, _, _, parent) in zip(contours, hierarchy[0]):
            if parent < 0 or len(c) < 5:  # 검은 영역에 둘러싸인 흰 구멍만
                continue
            area = cv2.contourArea(c)
            if area < np.pi * config.CROSSHAIR_MIN_RADIUS ** 2:
                continue
            (cx, cy), (d1, d2), angle = cv2.fitEllipse(c)
            a, b = d1 / 2, d2 / 2
            if min(a, b) < config.CROSSHAIR_MIN_RADIUS or min(a, b) / max(a, b) < 0.4:
                continue
            if area / (np.pi * a * b) < 0.8:  # 원(타원) 모양이 아님
                continue
            hits = self._ring_hits(dark, cx, cy, a, b, angle)
            if hits < config.CROSSHAIR_MIN_RAYS:
                continue
            if best is None or area > best[0]:
                best = (area, cx, cy, a, b, angle, hits)
        if best is None:
            return []
        _, cx, cy, a, b, angle, hits = best
        ra, rb = a * self.OUTER, b * self.OUTER
        outline = cv2.ellipse2Poly((int(cx), int(cy)), (int(ra), int(rb)), int(angle), 0, 360, 10)
        area_ratio = np.pi * ra * rb / (w * h)
        return [Detection("target", "target", (cx, cy), area_ratio, outline, hits)]

    def _ring_hits(self, dark, cx, cy, a, b, angle):
        """중심에서 16방향으로 고리 패턴(검-흰-검)이 맞는 방향 수. 십자선에 걸린 방향은 빠진다."""
        h, w = dark.shape
        t = np.deg2rad(angle)
        hits = 0
        for k in range(self.N_RAYS):
            th = 2 * np.pi * (k + 0.5) / self.N_RAYS
            ex, ey = a * np.cos(th), b * np.sin(th)
            dx, dy = ex * np.cos(t) - ey * np.sin(t), ex * np.sin(t) + ey * np.cos(t)
            for scale, want_dark in self.RINGS:
                x, y = int(round(cx + scale * dx)), int(round(cy + scale * dy))
                if not (0 <= x < w and 0 <= y < h) or (dark[y, x] > 0) != want_dark:
                    break
            else:
                hits += 1
        return hits


def _side(u, v):
    """2D 외적: 벡터 v가 u의 어느 쪽에 있는지(부호)와 거리."""
    return u[0] * v[1] - u[1] * v[0]


def arrow_direction(contour):
    """화살표 윤곽이면 LEFT/RIGHT/UP/DOWN, 아니면 None.

    화살표는 꼭짓점 7개짜리 다각형이고, 오목한 꼭짓점 2개(촉과 몸통이 만나는 곳) 사이에
    날개-촉-날개, 반대쪽에 꼬리 2개가 있다. 방향 = 두 오목점의 가운데 -> 촉.
    """
    peri = cv2.arcLength(contour, True)
    for eps in (0.02, 0.03, 0.04):
        approx = cv2.approxPolyDP(contour, eps * peri, True)
        if len(approx) == 7:
            break
    else:
        return None
    # 인쇄된 화살표는 테두리가 곧아서 7각형과 거의 겹친다. 들쭉날쭉한 그림자/천 주름은 여기서 걸러짐
    # (실측: 화살표 면적 차 0.02~0.04, 둘레 비 1.04~1.19 / 침구 오검출 0.25~0.29, 1.6)
    area = cv2.contourArea(contour)
    if area <= 0 or abs(area - cv2.contourArea(approx)) / area > 0.12:
        return None
    if peri / cv2.arcLength(approx, True) > 1.35:
        return None
    poly = approx.reshape(-1, 2).astype(float)

    prev, nxt = np.roll(poly, 1, axis=0), np.roll(poly, -1, axis=0)
    turn = np.sign([_side(p - q, n - p) for q, p, n in zip(prev, poly, nxt)])
    concave = np.flatnonzero(turn != np.sign(turn.sum()))  # 다수와 반대로 꺾이는 점
    if len(concave) != 2:
        return None
    i, j = concave
    if (j - i) % 7 == 4:
        head = [(i + 1) % 7, (i + 2) % 7, (i + 3) % 7]
    elif (j - i) % 7 == 3:
        head = [(j + 1) % 7, (j + 2) % 7, (j + 3) % 7]
    else:
        return None
    tails = [k for k in range(7) if k not in head and k not in (i, j)]

    notch_mid = (poly[i] + poly[j]) / 2
    axis = poly[head[1]] - notch_mid
    length = np.linalg.norm(axis)
    if length == 0:
        return None
    u = axis / length
    shaft = max(abs(_side(u, poly[k] - notch_mid)) for k in (i, j))
    wings = [abs(_side(u, poly[k] - notch_mid)) for k in (head[0], head[2])]
    if min(wings) < 1.5 * shaft or min(wings) < 0.5 * max(wings):  # 촉이 몸통보다 넓고 좌우 대칭
        return None
    if any(np.dot(poly[k] - notch_mid, u) > 0 for k in tails):  # 꼬리는 촉 반대쪽
        return None
    # 촉(오목점 -> 끝)은 몸통(오목점 -> 꼬리)보다 짧다. 빛 반사로 촉 한쪽이 지워지면 반대 방향으로
    # 잘못 읽히는데, 그때는 '촉'이 몸통보다 길게 나와서 여기서 걸러진다.
    proj = (poly - notch_mid) @ u
    if proj.max() >= -proj.min():
        return None
    dx, dy = u
    if abs(dx) >= abs(dy):
        return "RIGHT" if dx > 0 else "LEFT"
    return "DOWN" if dy > 0 else "UP"


class ArrowDetector:
    """흰 종이 위의 검은 화살표를 찾아 방향을 판단한다. 화살표 이미지 하나로 좌/우 모두 인식."""

    def detect(self, gray, dark):
        # 종이 바깥 배경이 종이 테두리를 따라 '어둡게' 잡혀 화살표를 감싸므로, 안쪽 윤곽까지 모두 본다
        contours, _ = cv2.findContours(dark, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        h, w = gray.shape
        best = None
        # 역광이나 가구 때문에 어두운 덩어리가 많아도 화살표를 놓치지 않게 넉넉히 본다
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:40]:
            area = cv2.contourArea(c)
            if area < config.ARROW_MIN_AREA_RATIO * w * h:
                break
            x, y, bw, bh = cv2.boundingRect(c)
            if x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:  # 화면 밖으로 잘린 모양
                continue
            direction = arrow_direction(c)
            if direction is None:
                continue
            contrast = self._print_contrast(gray, c)
            if contrast is None or (best and best[0] >= contrast):
                continue
            m = cv2.moments(c)
            center = (m["m10"] / m["m00"], m["m01"] / m["m00"])
            best = (contrast, Detection("arrow", direction, center, area / (w * h), c, contrast))
        return [best[1]] if best else []

    @staticmethod
    def _print_contrast(gray, contour):
        """인쇄된 화살표처럼 안쪽이 고르게 어둡고 바로 바깥(종이)보다 충분히 어두우면 밝기 차이, 아니면 None.

        어두운 물체 여러 개가 우연히 이어져 화살표 모양이 된 경우를 거른다.
        """
        inside = np.zeros_like(gray)
        cv2.drawContours(inside, [contour], -1, 255, -1)
        around = cv2.dilate(inside, np.ones((15, 15), np.uint8)) & ~inside
        core = cv2.erode(inside, np.ones((5, 5), np.uint8))  # 흐린 테두리 제외
        if not around.any() or not core.any():
            return None
        mean_in, std_in = (v[0][0] for v in cv2.meanStdDev(gray, mask=core))
        mean_out = cv2.mean(gray, around)[0]
        contrast = mean_out - mean_in
        if std_in > config.ARROW_MAX_INSIDE_STD or contrast < config.ARROW_MIN_CONTRAST:
            return None
        if mean_in > config.ARROW_MAX_DARK_RATIO * mean_out:  # 검은 잉크는 종이보다 훨씬 어둡다
            return None
        return contrast


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
