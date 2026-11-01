# -Ai-1-.
피지컬 Ai 기초반 1팀 라즈봇제작

목표 그림 / 표지판 / 손동작에 반응해서 움직이는 4륜 라즈봇.

| 보이는 것 | 행동 |
|---|---|
| **GOAL 표지판** | `STOP` 정지 |
| 손바닥 펼침 | `STOP` 정지 |
| 파란 화살표 (가까움) / 검지로 좌·우 가리키기 | `TURN_LEFT` / `TURN_RIGHT` |
| 파란 화살표 (멀리 있음) | `APPROACH` 표지판 앞까지 다가감 |
| 목표 그림 | `APPROACH` 다가감 (`steer`로 좌우 방향), 충분히 가까우면 `HOLD` |
| 아무것도 없음 | `SEARCH` |

여러 개가 동시에 보이면 위쪽이 우선. 순간적인 오검출을 막기 위해 최근 5프레임 중 3번 이상 나온 판단만 확정한다.

> 현재 단계: **인식 + 행동 판단까지**. 모터 제어는 다음 단계에서 `Action`(name, steer)을 받아 연결한다.

## 설치

```bash
pip install -r requirements.txt
```

손동작 모델(`models/hand_landmarker.task`)은 처음 실행할 때 자동으로 다운로드된다.

## 사용 순서

### 1. 표지판 만들고 인쇄

```bash
python make_signs.py
```

`assets/signs/`에 생성됨 → A4 가로로 인쇄.

- `goal.png` — GOAL 표지판 (그림 매칭으로 인식)
- `arrow_left.png`, `arrow_right.png` — 파란 화살표 (색 + 모양으로 인식)

### 2. 목표 그림 등록

목표로 쓸 그림을 카메라로 찍어서 등록한다. 선·글자·무늬가 많은 그림일수록 잘 인식된다 (단색 도형만 있는 그림은 X).

```bash
python register_target.py                      # 카메라: SPACE 촬영 → 마우스로 그림 영역 드래그 → ENTER
python register_target.py --image drawing.jpg  # 사진 파일로 등록
python register_target.py --name goal          # (선택) 인쇄한 GOAL을 실제 조명에서 다시 등록
```

`assets/templates/target.png`로 저장된다.

### 3. 인식 데모

```bash
python vision_demo.py                 # 카메라 화면에 인식 결과 + 행동 표시 (q 종료, m 화살표 색 마스크)
python vision_demo.py --image a.jpg   # 사진으로 테스트
python vision_demo.py --headless      # 모니터 없이 (SSH) 행동만 출력
python vision_demo.py --no-gesture    # 손동작 끄기 (라즈베리파이에서 속도 향상)
```

## 튜닝 (`config.py`)

| 증상 | 조정 |
|---|---|
| 화살표를 못 찾음 | `m` 키로 마스크 확인 → `ARROW_HSV_LOW/HIGH` 범위 조정 |
| 너무 멀리서/가까이서 회전함 | `ARROW_TURN_AREA_RATIO` |
| 목표 그림 앞에서 너무 일찍/늦게 멈춤 | `TARGET_HOLD_AREA_RATIO` |
| 그림 오검출이 잦음 | `MIN_INLIERS` 올리기 / `MATCH_RATIO` 낮추기 |
| 그림을 잘 못 찾음 | `MIN_INLIERS` 내리기, 실제 조명에서 다시 등록 |
| 느림 | `FRAME_WIDTH`를 480으로, `--no-gesture` |

참고: GOAL 표지판은 화면 가로의 약 20% 이상 크기로 보여야 인식된다.

## 파일 구조

```
config.py           설정값
detectors.py        그림(ORB) / 화살표(색+모양) / 손동작(MediaPipe) 검출
decision.py         검출 결과 → 행동 판단
make_signs.py       인쇄용 표지판 생성
register_target.py  목표 그림 등록
vision_demo.py      인식 데모
assets/signs/       인쇄용 표지판
assets/templates/   매칭 기준 그림 (target.png, goal.png)
```
