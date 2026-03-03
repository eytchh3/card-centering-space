from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


PSA_MIN_RATIO = 45.0 / 55.0
UNCERTAIN_MSG = "UNCERTAIN – insufficient photo quality."


@dataclass
class DetectionResult:
    status: str
    confidence: float
    lr_ratio: Optional[float]
    tb_ratio: Optional[float]
    details: Dict[str, Any]
    warped: Optional[np.ndarray] = None
    overlay: Optional[np.ndarray] = None


def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]  # tl
    ordered[2] = pts[np.argmax(s)]  # br
    ordered[1] = pts[np.argmin(d)]  # tr
    ordered[3] = pts[np.argmax(d)]  # bl
    return ordered


def _find_card_quad(image: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 160)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0

    h, w = gray.shape
    img_area = float(h * w)
    best = None
    best_score = -1.0

    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.1 * img_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) < 4:
            continue

        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box_area = cv2.contourArea(box)
        if box_area <= 0:
            continue

        rectangularity = float(area / box_area)
        fill_ratio = area / img_area
        score = (0.55 * rectangularity) + (0.45 * min(1.0, fill_ratio / 0.85))

        if score > best_score:
            best_score = score
            best = box

    if best is None:
        return None, 0.0
    return _order_quad_points(best), float(np.clip(best_score, 0.0, 1.0))


def _warp_card(image: np.ndarray, quad: np.ndarray, h_out: int = 1000) -> np.ndarray:
    aspect = 2.5 / 3.5  # standard card width/height
    w_out = int(h_out * aspect)
    dst = np.array([[0, 0], [w_out - 1, 0], [w_out - 1, h_out - 1], [0, h_out - 1]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    warped = cv2.warpPerspective(image, m, (w_out, h_out), flags=cv2.INTER_LINEAR)
    return warped


def _smooth_1d(arr: np.ndarray, k: int = 21) -> np.ndarray:
    k = max(3, int(k) | 1)
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(arr.astype(np.float32), kernel, mode="same")


def _peak_in_range(profile: np.ndarray, lo: int, hi: int) -> Tuple[int, float, float]:
    lo = int(np.clip(lo, 0, len(profile) - 1))
    hi = int(np.clip(hi, lo + 1, len(profile)))
    seg = profile[lo:hi]
    idx_local = int(np.argmax(seg))
    idx = lo + idx_local
    peak = float(seg[idx_local])
    baseline = float(np.median(seg)) if len(seg) else 0.0
    prominence = max(0.0, peak - baseline)
    return idx, peak, prominence


def _detect_inner_rails(warped: np.ndarray) -> Tuple[Optional[Dict[str, Any]], float]:
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    core_y0, core_y1 = int(0.18 * h), int(0.82 * h)
    core = gray[core_y0:core_y1, :]
    gx = cv2.Sobel(core, cv2.CV_32F, 1, 0, ksize=3)
    x_profile = np.mean(np.abs(gx), axis=0)
    x_profile = _smooth_1d(x_profile, k=max(15, w // 40))

    lx, lpk, lprom = _peak_in_range(x_profile, int(0.05 * w), int(0.45 * w))
    rx, rpk, rprom = _peak_in_range(x_profile, int(0.55 * w), int(0.95 * w))
    if rx <= lx + int(0.15 * w):
        return None, 0.0

    core_x0, core_x1 = int(lx + 0.08 * (rx - lx)), int(rx - 0.08 * (rx - lx))
    core_x0 = max(0, min(core_x0, w - 2))
    core_x1 = max(core_x0 + 1, min(core_x1, w - 1))

    mid = gray[:, core_x0:core_x1]
    gy = cv2.Sobel(mid, cv2.CV_32F, 0, 1, ksize=3)
    y_profile = np.mean(np.abs(gy), axis=1)
    y_profile = _smooth_1d(y_profile, k=max(15, h // 40))

    ty, tpk, tprom = _peak_in_range(y_profile, int(0.05 * h), int(0.45 * h))
    by, bpk, bprom = _peak_in_range(y_profile, int(0.55 * h), int(0.95 * h))
    if by <= ty + int(0.15 * h):
        return None, 0.0

    prom = np.array([lprom, rprom, tprom, bprom], dtype=np.float32)
    scale = float(np.percentile(np.concatenate([x_profile, y_profile]), 90) + 1e-6)
    norm = np.clip(prom / scale, 0.0, 3.0)
    conf = float(np.clip(np.mean(norm) / 1.5, 0.0, 1.0))

    return {
        "left": lx,
        "right": rx,
        "top": ty,
        "bottom": by,
        "x_profile": x_profile,
        "y_profile": y_profile,
        "peaks": {"l": lpk, "r": rpk, "t": tpk, "b": bpk},
    }, conf


def analyze_centering(image_bgr: np.ndarray, conf_threshold: float = 0.45) -> DetectionResult:
    quad, card_conf = _find_card_quad(image_bgr)
    if quad is None:
        return DetectionResult(
            status=UNCERTAIN_MSG,
            confidence=0.0,
            lr_ratio=None,
            tb_ratio=None,
            details={"reason": "outer_card_not_found"},
        )

    warped = _warp_card(image_bgr, quad)
    rails, rail_conf = _detect_inner_rails(warped)
    if rails is None:
        return DetectionResult(
            status=UNCERTAIN_MSG,
            confidence=card_conf * 0.5,
            lr_ratio=None,
            tb_ratio=None,
            details={"reason": "inner_frame_not_found", "card_conf": card_conf},
            warped=warped,
        )

    h, w = warped.shape[:2]
    left_gap = float(rails["left"])
    right_gap = float((w - 1) - rails["right"])
    top_gap = float(rails["top"])
    bottom_gap = float((h - 1) - rails["bottom"])

    lr_ratio = min(left_gap, right_gap) / max(left_gap, right_gap, 1e-6)
    tb_ratio = min(top_gap, bottom_gap) / max(top_gap, bottom_gap, 1e-6)

    confidence = float(np.clip(0.55 * card_conf + 0.45 * rail_conf, 0.0, 1.0))

    overlay = warped.copy()
    cv2.line(overlay, (int(rails["left"]), 0), (int(rails["left"]), h - 1), (0, 255, 0), 2)
    cv2.line(overlay, (int(rails["right"]), 0), (int(rails["right"]), h - 1), (0, 255, 0), 2)
    cv2.line(overlay, (0, int(rails["top"])), (w - 1, int(rails["top"])), (255, 0, 0), 2)
    cv2.line(overlay, (0, int(rails["bottom"])), (w - 1, int(rails["bottom"])), (255, 0, 0), 2)

    if confidence < conf_threshold:
        status = UNCERTAIN_MSG
    else:
        passed = (lr_ratio >= PSA_MIN_RATIO) and (tb_ratio >= PSA_MIN_RATIO)
        status = "PASS" if passed else "FAIL"

    return DetectionResult(
        status=status,
        confidence=confidence,
        lr_ratio=lr_ratio,
        tb_ratio=tb_ratio,
        details={
            "left_gap": left_gap,
            "right_gap": right_gap,
            "top_gap": top_gap,
            "bottom_gap": bottom_gap,
            "card_conf": card_conf,
            "rail_conf": rail_conf,
            "threshold_ratio": PSA_MIN_RATIO,
        },
        warped=warped,
        overlay=overlay,
    )
