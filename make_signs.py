"""인쇄용 표지판 생성: GOAL, 왼쪽/오른쪽 화살표 (A4 가로, 150dpi).

    python make_signs.py           # assets/signs/ 에 생성, GOAL은 매칭용 템플릿에도 등록
    python make_signs.py --force   # 이미 등록된 goal 템플릿도 덮어쓰기
"""
import argparse

import cv2
import numpy as np

import config
from detectors import load_image, save_image

W, H = 1754, 1240
GREEN = (60, 170, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


def shape_band(img, cy, order):
    """서로 다른 도형 띠. 체크무늬처럼 반복되는 무늬는 특징점 매칭을 헷갈리게 해서 피한다."""
    xs = np.linspace(170, W - 170, len(order)).astype(int)
    r = 55
    for x, shape in zip(xs, order):
        if shape == "circle":
            cv2.circle(img, (x, cy), r, BLACK, -1)
        elif shape == "ring":
            cv2.circle(img, (x, cy), r, BLACK, 14)
        elif shape == "square":
            cv2.rectangle(img, (x - r, cy - r), (x + r, cy + r), BLACK, -1)
        elif shape == "triangle":
            cv2.fillPoly(img, [np.array([[x, cy - r], [x + r, cy + r], [x - r, cy + r]], np.int32)], BLACK)
        elif shape == "cross":
            cv2.line(img, (x - r, cy - r), (x + r, cy + r), BLACK, 22)
            cv2.line(img, (x - r, cy + r), (x + r, cy - r), BLACK, 22)
        elif shape == "diamond":
            cv2.fillPoly(img, [np.array([[x, cy - r], [x + r, cy], [x, cy + r], [x - r, cy]], np.int32)], GREEN)


def centered_text(img, text, cy, scale, thickness, color=BLACK):
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(img, text, ((W - tw) // 2, cy + th // 2), font, scale, color, thickness, cv2.LINE_AA)


def make_goal():
    img = np.full((H, W, 3), 255, np.uint8)
    cv2.rectangle(img, (25, 25), (W - 25, H - 25), GREEN, 50)
    shape_band(img, 160, ("circle", "triangle", "cross", "square", "diamond", "ring", "triangle"))
    shape_band(img, H - 160, ("square", "ring", "diamond", "triangle", "circle", "cross", "square"))
    centered_text(img, "GOAL", 560, 9, 40)
    centered_text(img, "STOP HERE", 860, 3, 10, GREEN)
    # 깃발 아이콘: 좌우 비대칭 무늬로 특징점 매칭을 안정적으로
    cv2.line(img, (170, 330), (170, 930), BLACK, 18)
    pts = np.array([[180, 340], [400, 420], [180, 500]], np.int32)
    cv2.fillPoly(img, [pts], GREEN)
    cv2.circle(img, (W - 220, 620), 90, BLACK, 16)
    cv2.circle(img, (W - 220, 620), 35, GREEN, -1)
    return img


def make_arrow(direction):
    """오른쪽 화살표를 그리고 왼쪽은 좌우 반전. 촉이 크고 뾰족해야 방향 판정이 잘 된다."""
    img = np.full((H, W, 3), 255, np.uint8)
    x0, x1 = 180, W - 180
    head_x = x0 + int((x1 - x0) * 0.55)
    cy, half_head, half_shaft = H // 2, 450, 135
    pts = np.array([
        [x0, cy - half_shaft], [head_x, cy - half_shaft], [head_x, cy - half_head],
        [x1, cy], [head_x, cy + half_head], [head_x, cy + half_shaft], [x0, cy + half_shaft],
    ], np.int32)
    cv2.fillPoly(img, [pts], BLACK)
    return cv2.flip(img, 1) if direction == "left" else img


def pad_original_arrow(margin=0.2):
    """assets/originals/arrow.jpg 를 A4 가로 가운데에 넉넉한 흰 여백(가로세로 각 margin)으로 배치.

    원본은 화살촉 끝이 종이 가장자리에 너무 가까워서, 멀리서 보면 종이 밖 어두운 배경과
    붙어 버려 인식이 안 된다.
    """
    src = config.ASSETS_DIR / "originals" / "arrow.jpg"
    if not src.exists():
        return None
    arrow = load_image(src)
    gray = cv2.cvtColor(arrow, cv2.COLOR_BGR2GRAY)
    ys, xs = np.nonzero(gray < 128)  # 화살표(검은 부분)만 잘라냄
    arrow = arrow[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = arrow.shape[:2]
    scale = min(W * (1 - 2 * margin) / w, H * (1 - 2 * margin) / h)
    arrow = cv2.resize(arrow, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    img = np.full((H, W, 3), 255, np.uint8)
    y0, x0 = (H - arrow.shape[0]) // 2, (W - arrow.shape[1]) // 2
    img[y0:y0 + arrow.shape[0], x0:x0 + arrow.shape[1]] = arrow
    return img


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="goal 템플릿 덮어쓰기")
    args = parser.parse_args()

    goal = make_goal()
    signs = {"goal.png": goal, "arrow_left.png": make_arrow("left"), "arrow_right.png": make_arrow("right")}
    padded = pad_original_arrow()
    if padded is not None:
        signs["arrow_print.png"] = padded  # 우리 화살표를 여백 넉넉하게 (인쇄용)
    for name, img in signs.items():
        save_image(config.SIGNS_DIR / name, img)
        print(f"생성: {config.SIGNS_DIR / name}")

    template = config.TEMPLATES_DIR / "goal" / "printed.png"
    if args.force or not template.exists():
        save_image(template, goal)
        print(f"GOAL 매칭 템플릿 등록: {template}")
    else:
        print(f"GOAL 템플릿이 이미 있어 유지: {template} (--force 로 덮어쓰기)")


if __name__ == "__main__":
    main()
