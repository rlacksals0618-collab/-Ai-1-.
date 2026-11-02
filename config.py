"""라즈봇 설정값 (인식 + 주행). 조명/거리/바닥에 맞게 여기서만 조정한다."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSETS_DIR = ROOT / "assets"
SIGNS_DIR = ASSETS_DIR / "signs"          # 인쇄용 표지판 이미지
TEMPLATES_DIR = ASSETS_DIR / "templates"  # 로봇이 매칭할 기준 그림 (<이름>/ 폴더에 여러 장)
MODELS_DIR = ROOT / "models"

# 카메라
CAMERA_INDEX = 0
FRAME_WIDTH = 640  # 처리 해상도 (가로). 라즈베리파이에서 느리면 480으로 낮추기

# 그림 매칭 (ORB 특징점). 과녁/화살표는 전용 검출기를 쓰므로 여기엔 GOAL만
TEMPLATE_NAMES = ("goal",)
TEMPLATE_MAX_SIZE = 480
ORB_FEATURES = 1000
MATCH_RATIO = 0.75       # 낮을수록 엄격
MIN_GOOD_MATCHES = 15
MIN_INLIERS = 12
MIN_TEMPLATE_AREA_RATIO = 0.005  # 화면 대비 너무 작게 잡힌 건 무시

# 검은 잉크 검출 (화살표, 과녁 공통)
DARK_BLOCK_SIZE = 81   # 주변 밝기를 비교할 범위(px, 홀수)
DARK_OFFSET = 15       # 주변 평균보다 이만큼 어두우면 잉크로 봄

# 과녁
CROSSHAIR_MIN_RADIUS = 6   # 안쪽 흰 원 최소 반지름(px). 이보다 작으면(멀면) 무시
CROSSHAIR_MIN_RAYS = 10    # 16방향 중 고리 패턴이 맞아야 하는 방향 수

# 화살표 (흰 종이 위 검은 화살표)
ARROW_MIN_CONTRAST = 40        # 화살표와 바로 바깥 종이의 밝기 차이 최소값 (실측: 역광에서 종이가 어두우면 73)
ARROW_MAX_INSIDE_STD = 35      # 화살표 안쪽 밝기 편차 최대값 (인쇄면은 고르게 검다)
ARROW_MAX_DARK_RATIO = 0.35    # 화살표 밝기 / 바깥 종이 밝기 최대값 (실측: 인쇄 화살표 0.18~0.20, 그림자·천 0.55)
ARROW_MIN_AREA_RATIO = 0.003   # 이보다 작으면 무시

# 손동작 (MediaPipe Hand Landmarker). vision_demo.py --gesture 에서만 사용
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"

# 표지판 확정
DEBOUNCE_WINDOW = 5     # 최근 N프레임 중
DEBOUNCE_MIN_HITS = 3   # 이만큼 같은 표지판이 보여야 확정

# ---- 주행 (PLAN.md 3~5단계). 바퀴를 띄운 채 hw_test.py로 확인하며 조정 ----
# 모터 값: -255 ~ 255
SPEED_NORMAL = 70
SPEED_SLOW = 40
SPEED_TURN = 60          # 90도 회전
SPEED_SCAN = 45          # 다가가다 표지판이 옆으로 벗어났을 때 제자리 회전
LEFT_MOTOR_SIGN = 1      # 앞으로 가라고 했는데 왼쪽 바퀴가 뒤로 돌면 -1
RIGHT_MOTOR_SIGN = 1     # 오른쪽도 마찬가지

STEER_GAIN = 0.6         # 표지판이 옆에 있을 때 꺾는 정도
CENTER_TOL = 0.35        # 표지판이 화면 중앙 ±35% 안이면 '정면' (초음파 값을 믿는 조건)
SPIN_TOL = 0.6           # 이보다 옆이면 전진하지 않고 제자리에서 돌려 맞춤

# 거리 (초음파, cm)
ARRIVE_CM = 25           # 이 안이면 도착
SLOW_CM = 60             # 이 안이면 감속
DIST_MEDIAN = 3          # 최근 N번 측정값의 중간값 사용
DIST_MAX_CM = 400        # 이보다 크거나 0이면 측정 실패로 처리

# 시간 (초)
TURN_90_SEC = 0.54       # 90도 회전 시간 (SPEED_TURN=60, 실측: 0.9초 = 약 150도). 바닥이 바뀌면 hw_test.py turn 으로 다시 맞추기
MOVE_SEC = 1.5           # 아래 화살표에서 후진하는 시간
LOST_SEC = 2.0           # 다가가다 표지판을 이 시간 넘게 놓치면 다시 찾기
CLOSE_LOST_SEC = 4.0     # 60cm 안에서 놓쳤으면(가까워서 화면을 벗어난 것) 이만큼 더 기억
STOP_SEC = 3.0           # 과녁 앞에서 멈춰 있는 시간. 그 뒤 다시 1단계(찾기)부터
COOLDOWN_SEC = 8.0       # 다시 찾기 시작한 뒤, 방금 실행한 표지판은 이 시간 동안 무시
                         # (카메라로 옆까지 둘러보므로 방금 지나온 화살표가 다시 보일 수 있음)

# ---- 카메라 서보 (2DOF) ----
CAMERA_PAN = 90          # 좌우: 실제로 로봇 정면을 보는 서보 각도. 거치대가 삐뚤면 hw_test.py center 로 맞추기
PAN_RIGHT_SIGN = 1       # 실측: pan이 90보다 작으면 오른쪽, 크면 왼쪽을 봄. 반대면 -1
CAMERA_TILT = 20         # 상하 기본 각도. 실측: 90 = 거의 천장, 30 = 약간 위, 10 = 약간 아래
CAMERA_HFOV = 65         # 카메라 가로 시야각(도). 실측 약 60~76

# 찾기: 몸은 멈추고 카메라만 이 순서로 돌려 본다 (정면 -> 왼쪽 -> 오른쪽 -> 더 왼쪽 -> 더 오른쪽)
# 정면(CAMERA_PAN) 기준 각도. 서보는 0~180까지만 가므로 그 안으로 자름
SCAN_PANS = tuple(max(0, min(180, CAMERA_PAN + d * PAN_RIGHT_SIGN)) for d in (0, 35, -35, 70, -70))
SCAN_SETTLE_SEC = 0.35   # 카메라를 돌린 직후 흔들리는 동안의 화면은 버림
SCAN_DWELL_SEC = 0.8     # 한 방향을 보는 시간

# 다가갈 때 카메라 상하 각도가 표지판을 따라감
TILT_MIN, TILT_MAX = 0, 60
TILT_DEADZONE = 0.25     # 표지판이 화면 세로 가운데 ±25% 안이면 그대로
TILT_STEP = 1            # 벗어나면 프레임마다 이만큼(도) 움직임

# OLED (확장보드 화면)
OLED_HEIGHT = 32         # 128x32. 글자가 위아래로 늘어나 보이면 64
