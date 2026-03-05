from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

WARP_WIDTH = 700
WARP_HEIGHT = 1000
CARD_ASPECT_RATIO = WARP_WIDTH / WARP_HEIGHT


def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[1] = pts[np.argmin(d)]
    ordered[2] = pts[np.argmax(s)]
    ordered[3] = pts[np.argmax(d)]
    return ordered


def _line_from_points(points: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if points.shape[0] < 8:
        return None
    vx, vy, x0, y0 = cv2.fitLine(points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
    direction = np.array([float(vx.item()), float(vy.item())], dtype=np.float32)
    origin = np.array([float(x0.item()), float(y0.item())], dtype=np.float32)
    if np.linalg.norm(direction) < 1e-6:
        return None
    return origin, direction


def _line_intersection(
    line_a: Tuple[np.ndarray, np.ndarray], line_b: Tuple[np.ndarray, np.ndarray]
) -> Optional[np.ndarray]:
    p, r = line_a
    q, s = line_b
    matrix = np.array([[r[0], -s[0]], [r[1], -s[1]]], dtype=np.float32)
    det = float(np.linalg.det(matrix))
    if abs(det) < 1e-6:
        return None
    rhs = q - p
    t_u = np.linalg.solve(matrix, rhs)
    intersection = p + (t_u[0] * r)
    return intersection.astype(np.float32)


def _quad_by_contour(gray: np.ndarray) -> Optional[np.ndarray]:
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 45, 145)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    img_area = float(h * w)
    best_area = -1.0
    best_quad: Optional[np.ndarray] = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 0.12 * img_area:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) != 4:
            continue

        quad = _order_quad_points(approx.reshape(4, 2).astype(np.float32))
        if area > best_area:
            best_area = area
            best_quad = quad

    return best_quad


def _quad_by_hough(gray: np.ndarray) -> Optional[np.ndarray]:
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=int(min(h, w) * 0.2),
        maxLineGap=20,
    )
    if lines is None or len(lines) < 8:
        return None

    vertical_segments = []
    horizontal_segments = []
    vertical_lengths = []
    horizontal_lengths = []

    for raw in lines[:, 0, :]:
        x1, y1, x2, y2 = map(float, raw)
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0
        seg = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
        length = float(np.hypot(x2 - x1, y2 - y1))
        if angle < 25 or angle > 155:
            horizontal_segments.append(seg)
            horizontal_lengths.append(length)
        elif 65 < angle < 115:
            vertical_segments.append(seg)
            vertical_lengths.append(length)

    if len(vertical_segments) < 2 or len(horizontal_segments) < 2:
        return None

    v_idx = np.argsort(vertical_lengths)[-max(2, len(vertical_lengths) // 2) :]
    h_idx = np.argsort(horizontal_lengths)[-max(2, len(horizontal_lengths) // 2) :]

    v_pts = np.vstack([vertical_segments[i] for i in v_idx])
    h_pts = np.vstack([horizontal_segments[i] for i in h_idx])

    x_q1, x_q3 = np.quantile(v_pts[:, 0], [0.25, 0.75])
    y_q1, y_q3 = np.quantile(h_pts[:, 1], [0.25, 0.75])

    left_line = _line_from_points(v_pts[v_pts[:, 0] <= x_q1])
    right_line = _line_from_points(v_pts[v_pts[:, 0] >= x_q3])
    top_line = _line_from_points(h_pts[h_pts[:, 1] <= y_q1])
    bottom_line = _line_from_points(h_pts[h_pts[:, 1] >= y_q3])

    if any(v is None for v in [left_line, right_line, top_line, bottom_line]):
        return None

    tl = _line_intersection(left_line, top_line)
    tr = _line_intersection(right_line, top_line)
    br = _line_intersection(right_line, bottom_line)
    bl = _line_intersection(left_line, bottom_line)
    if any(v is None for v in [tl, tr, br, bl]):
        return None

    quad = _order_quad_points(np.array([tl, tr, br, bl], dtype=np.float32))
    return quad


def _angle_score(quad: np.ndarray) -> float:
    pts = quad.astype(np.float32)
    score = 0.0
    for i in range(4):
        a = pts[(i - 1) % 4] - pts[i]
        b = pts[(i + 1) % 4] - pts[i]
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-6 or nb < 1e-6:
            return 0.0
        cosang = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
        angle = np.degrees(np.arccos(cosang))
        score += max(0.0, 1.0 - abs(angle - 90.0) / 45.0)
    return score / 4.0


def _edge_gradient_score(gray: np.ndarray, quad: np.ndarray) -> float:
    grad = cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=3)
    vals = []
    for i in range(4):
        p0 = quad[i].astype(np.int32)
        p1 = quad[(i + 1) % 4].astype(np.int32)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.line(mask, tuple(p0), tuple(p1), 255, 5)
        edge_vals = grad[mask > 0]
        if edge_vals.size == 0:
            vals.append(0.0)
        else:
            vals.append(float(np.mean(np.abs(edge_vals))))
    m = float(np.mean(vals))
    return float(np.clip(m / 40.0, 0.0, 1.0))


def _score_quad(gray: np.ndarray, quad: np.ndarray) -> float:
    h, w = gray.shape
    area = max(0.0, cv2.contourArea(quad))
    area_score = float(np.clip(area / (h * w), 0.0, 1.0))
    angle_score = _angle_score(quad)
    grad_score = _edge_gradient_score(gray, quad)

    x_ok = np.all((quad[:, 0] >= -0.1 * w) & (quad[:, 0] <= 1.1 * w))
    y_ok = np.all((quad[:, 1] >= -0.1 * h) & (quad[:, 1] <= 1.1 * h))
    if not (x_ok and y_ok):
        return 0.0

    return (0.45 * area_score) + (0.30 * angle_score) + (0.25 * grad_score)


def _detect_card_outer_quad(gray: np.ndarray) -> Optional[np.ndarray]:
    cand_a = _quad_by_contour(gray)
    cand_b = _quad_by_hough(gray)

    candidates = []
    if cand_a is not None:
        candidates.append((cand_a, _score_quad(gray, cand_a), "contour"))
    if cand_b is not None:
        candidates.append((cand_b, _score_quad(gray, cand_b), "hough"))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[1], reverse=True)
    best_quad, best_score, _ = candidates[0]

    if best_score < 0.22:
        return None

    return _order_quad_points(best_quad)


def _expand_quad(quad: np.ndarray, scale: float, shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    center = np.mean(quad, axis=0)
    expanded = center + (quad - center) * scale
    expanded[:, 0] = np.clip(expanded[:, 0], 0, w - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, h - 1)
    return expanded.astype(np.float32)


def _warp_card(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    dst = np.array(
        [[0, 0], [WARP_WIDTH - 1, 0], [WARP_WIDTH - 1, WARP_HEIGHT - 1], [0, WARP_HEIGHT - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(image, matrix, (WARP_WIDTH, WARP_HEIGHT), flags=cv2.INTER_LINEAR)


def _detect_frame_by_hsv_mask(warped: np.ndarray) -> Optional[np.ndarray]:
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]

    lower_blue = np.array([85, 35, 20], dtype=np.uint8)
    upper_blue = np.array([140, 255, 165], dtype=np.uint8)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Include dark low-saturation frame lines often seen in scans/synthetic cards.
    lower_dark = np.array([0, 0, 15], dtype=np.uint8)
    upper_dark = np.array([180, 120, 110], dtype=np.uint8)
    mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)

    mask = cv2.bitwise_or(mask_blue, mask_dark)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    card_area = float(h * w)
    best_quad: Optional[np.ndarray] = None
    best_score = -1.0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 0.05 * card_area or area > 0.90 * card_area:
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            quad = _order_quad_points(approx.reshape(4, 2).astype(np.float32))
        else:
            rect = cv2.minAreaRect(contour)
            quad = _order_quad_points(cv2.boxPoints(rect).astype(np.float32))

        x_min, y_min = np.min(quad, axis=0)
        x_max, y_max = np.max(quad, axis=0)
        width = float(max(1.0, x_max - x_min))
        height = float(max(1.0, y_max - y_min))
        ratio = width / height

        if abs(ratio - CARD_ASPECT_RATIO) > 0.25:
            continue
        if x_min < 1 or y_min < 1 or x_max > (w - 2) or y_max > (h - 2):
            continue

        box_area = cv2.contourArea(quad)
        if box_area <= 0:
            continue

        rectangularity = float(area / max(box_area, 1.0))
        center_bias = abs((x_min + x_max) * 0.5 - (w * 0.5)) / max(w, 1.0)
        score = (0.65 * (area / card_area)) + (0.25 * rectangularity) + (0.10 * (1.0 - min(center_bias, 1.0)))

        if score > best_score:
            best_score = score
            best_quad = quad

    return best_quad


def _estimate_frame_by_border_color(warped: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    mid_band = gray[int(0.2 * h) : int(0.8 * h), :]
    vert_grad = np.mean(np.abs(np.diff(mid_band.astype(np.float32), axis=1)), axis=0)

    side_band = gray[:, int(0.2 * w) : int(0.8 * w)]
    horiz_grad = np.mean(np.abs(np.diff(side_band.astype(np.float32), axis=0)), axis=1)

    smooth_kx = max(9, (w // 40) | 1)
    smooth_ky = max(9, (h // 40) | 1)
    vert_grad = cv2.GaussianBlur(vert_grad.reshape(1, -1), (smooth_kx, 1), 0).reshape(-1)
    horiz_grad = cv2.GaussianBlur(horiz_grad.reshape(-1, 1), (1, smooth_ky), 0).reshape(-1)

    left = int(np.argmax(vert_grad[: int(0.45 * w)]))
    right = int(np.argmax(vert_grad[int(0.55 * w) :]) + int(0.55 * w))
    top = int(np.argmax(horiz_grad[: int(0.45 * h)]))
    bottom = int(np.argmax(horiz_grad[int(0.55 * h) :]) + int(0.55 * h))

    frame_quad = np.array([[left, top], [right, top], [right, bottom], [left, bottom]], dtype=np.float32)
    return frame_quad


def _ratio_text(a: float, b: float) -> str:
    total = max(a + b, 1e-6)
    left = int(round((a / total) * 100))
    right = 100 - left
    return f"{left}/{right}"


def _draw_outer_debug(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    dbg = image.copy()
    pts = quad.astype(int)
    cv2.polylines(dbg, [pts], True, (255, 255, 0), 2)  # cyan edges
    for x, y in pts:
        cv2.circle(dbg, (int(x), int(y)), 7, (0, 255, 255), -1)  # yellow corners
    return dbg


def analyze_centering(image_bgr: np.ndarray) -> Dict[str, Any]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    card_quad = _detect_card_outer_quad(gray)
    if card_quad is None:
        inner_quad = _quad_by_hough(gray)
        if inner_quad is None:
            return {"error": "Card could not be detected"}
        card_quad = _expand_quad(inner_quad, scale=1.15, shape=gray.shape)
        card_source = "inner_hough_fallback"
    else:
        card_source = "outer_dual_method"

    warped = _warp_card(image_bgr, card_quad)

    frame_quad = _detect_frame_by_hsv_mask(warped)
    used_fallback_frame = False
    if frame_quad is None:
        frame_quad = _estimate_frame_by_border_color(warped)
        used_fallback_frame = True

    h, w = warped.shape[:2]
    frame = _order_quad_points(frame_quad)

    left_border = float(frame[0][0])
    right_border = float((w - 1) - frame[1][0])
    top_border = float(frame[0][1])
    bottom_border = float((h - 1) - frame[2][1])

    result = {
        "left_border_px": round(left_border, 2),
        "right_border_px": round(right_border, 2),
        "top_border_px": round(top_border, 2),
        "bottom_border_px": round(bottom_border, 2),
        "centering_lr": _ratio_text(left_border, right_border),
        "centering_tb": _ratio_text(top_border, bottom_border),
        "card_detection": card_source,
        "frame_detection": "border_color_fallback" if used_fallback_frame else "hsv_mask",
        "warped_image": warped,
        "debug_image": _draw_outer_debug(image_bgr, card_quad),
    }
    return result
