"""라즈봇 비전 설정값. 조명/거리/표지판에 맞게 여기서만 조정한다."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSETS_DIR = ROOT / "assets"
SIGNS_DIR = ASSETS_DIR / "signs"          # 인쇄용 표지판 이미지
TEMPLATES_DIR = ASSETS_DIR / "templates"  # 로봇이 매칭할 기준 그림 (<이름>.png)
MODELS_DIR = ROOT / "models"

# 카메라
CAMERA_INDEX = 0
FRAME_WIDTH = 640  # 처리 해상도 (가로). 라즈베리파이에서 느리면 480으로 낮추기

# 그림 매칭 (ORB 특징점)
TEMPLATE_NAMES = ("target", "goal")
TEMPLATE_MAX_SIZE = 480
ORB_FEATURES = 1000
MATCH_RATIO = 0.75       # 낮을수록 엄격
MIN_GOOD_MATCHES = 15
MIN_INLIERS = 12
MIN_TEMPLATE_AREA_RATIO = 0.005  # 화면 대비 너무 작게 잡힌 건 무시

# 화살표 (파란색, OpenCV HSV 기준 H:0~179)
ARROW_HSV_LOW = (95, 120, 50)
ARROW_HSV_HIGH = (130, 255, 255)
ARROW_MIN_AREA_RATIO = 0.003   # 이보다 작으면 무시
ARROW_TURN_AREA_RATIO = 0.015  # 이보다 작으면(멀면) 화살표로 다가가고, 크면(가까우면) 회전
                               # A4 표지판이 화면 가로의 약 27%를 채울 때

# 손동작 (MediaPipe Hand Landmarker)
GESTURE_ENABLED = True
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"

# 행동 판단
DEBOUNCE_WINDOW = 5     # 최근 N프레임 중
DEBOUNCE_MIN_HITS = 3   # 이만큼 같은 판단이 나와야 확정
TARGET_HOLD_AREA_RATIO = 0.25  # 목표 그림이 화면의 25% 이상이면 충분히 가까움 -> 대기
