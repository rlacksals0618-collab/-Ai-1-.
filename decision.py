"""검출 결과 -> 로봇 행동 판단. (모터 제어는 아직 없고, 행동 이름만 정한다)

우선순위: GOAL 정지 > 손바닥 정지 > 손가락/가까운 화살표 회전 > 먼 화살표 접근 > 목표 그림 접근 > 탐색
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import config

PRIORITY = {"STOP": 0, "TURN_LEFT": 1, "TURN_RIGHT": 1, "HOLD": 2, "APPROACH": 3, "SEARCH": 4}


@dataclass
class Action:
    name: str           # STOP / TURN_LEFT / TURN_RIGHT / HOLD / APPROACH / SEARCH
    steer: float = 0.0  # -1(왼쪽) ~ +1(오른쪽). APPROACH 때 목표가 화면 어디 있는지
    reason: str = ""


def _steer(detection, frame_width):
    """화면 중앙 기준 좌우 위치 (-1 ~ +1)."""
    return (detection.center[0] - frame_width / 2) / (frame_width / 2)


def decide_frame(detections, frame_width):
    """한 프레임만 보고 내리는 판단."""
    by_kind = {}
    for d in detections:
        by_kind.setdefault(d.kind, []).append(d)

    if "goal" in by_kind:
        return Action("STOP", reason="GOAL sign")
    hands = by_kind.get("hand", [])
    if any(h.label == "PALM" for h in hands):
        return Action("STOP", reason="open palm")

    arrows = [d for d in by_kind.get("arrow", []) if d.label in ("LEFT", "RIGHT")]
    near_arrows = [d for d in arrows if d.area_ratio >= config.ARROW_TURN_AREA_RATIO]
    turns = [d for d in hands if d.label in ("LEFT", "RIGHT")] + near_arrows
    if turns:
        d = max(turns, key=lambda t: t.area_ratio)  # 가장 가까운 신호
        return Action(f"TURN_{d.label}", -1.0 if d.label == "LEFT" else 1.0, f"{d.kind} {d.label}")
    if arrows:  # 아직 먼 화살표: 표지판 앞까지 다가간 뒤 회전
        d = max(arrows, key=lambda t: t.area_ratio)
        return Action("APPROACH", _steer(d, frame_width), f"arrow {d.label} far")

    if "target" in by_kind:
        t = max(by_kind["target"], key=lambda t: t.area_ratio)
        steer = _steer(t, frame_width)
        if t.area_ratio >= config.TARGET_HOLD_AREA_RATIO:
            return Action("HOLD", steer, "target close")
        return Action("APPROACH", steer, f"target {t.area_ratio:.0%}")

    return Action("SEARCH", reason="nothing")


class Decider:
    """최근 몇 프레임에서 반복된 판단만 확정해 순간적인 오검출을 거른다."""

    def __init__(self, window=config.DEBOUNCE_WINDOW, min_hits=config.DEBOUNCE_MIN_HITS):
        self.history = deque(maxlen=window)
        self.min_hits = min_hits

    def update(self, detections, frame_width):
        self.history.append(decide_frame(detections, frame_width))
        counts = Counter(a.name for a in self.history)
        confirmed = [n for n, c in counts.items() if c >= self.min_hits and n != "SEARCH"]
        if not confirmed:
            return Action("SEARCH", reason="nothing confirmed")
        best = min(confirmed, key=PRIORITY.__getitem__)
        return next(a for a in reversed(self.history) if a.name == best)
