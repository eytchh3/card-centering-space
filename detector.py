from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

WARP_WIDTH = 700
WARP_HEIGHT = 1000


def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.astype(np.float32)
    y_sorted = pts[np.argsort(pts[:, 1])]
    top = y_sorted[:2]
    bottom = y_sorted[2:]

    top = top[np.argsort(top[:, 0])]
    bottom = bottom[np.argsort(bottom[:, 0])]

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = top[0]  # tl
    ordered[1] = top[1]  # tr
    ordered[2] = bottom[1]  # br
    ordered[3] = bottom[0]  # bl
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


def _quad_from_hough(gray: np.ndarray, min_area_ratio: float) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
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
        return None, {"reason": "insufficient_hough_lines"}

    vertical_segments = []
    horizontal_segments = []
    for raw in lines[:, 0, :]:
        x1, y1, x2, y2 = map(float, raw)
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0
        segment = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
        if angle < 25 or angle > 155:
            horizontal_segments.append(segment)
        elif 65 < angle < 115:
            vertical_segments.append(segment)

    if len(vertical_segments) < 2 or len(horizontal_segments) < 2:
        return None, {"reason": "missing_line_orientations"}

    v_pts = np.vstack(vertical_segments)
    h_pts = np.vstack(horizontal_segments)

    x_q1, x_q3 = np.quantile(v_pts[:, 0], [0.25, 0.75])
    y_q1, y_q3 = np.quantile(h_pts[:, 1], [0.25, 0.75])

    left_line = _line_from_points(v_pts[v_pts[:, 0] <= x_q1])
    right_line = _line_from_points(v_pts[v_pts[:, 0] >= x_q3])
    top_line = _line_from_points(h_pts[h_pts[:, 1] <= y_q1])
    bottom_line = _line_from_points(h_pts[h_pts[:, 1] >= y_q3])

    if any(line is None for line in [left_line, right_line, top_line, bottom_line]):
        return None, {"reason": "line_fitting_failed"}

    tl = _line_intersection(left_line, top_line)
    tr = _line_intersection(right_line, top_line)
    br = _line_intersection(right_line, bottom_line)
    bl = _line_intersection(left_line, bottom_line)

    if any(corner is None for corner in [tl, tr, br, bl]):
        return None, {"reason": "line_intersection_failed"}

    quad = _order_quad_points(np.array([tl, tr, br, bl], dtype=np.float32))

    area = cv2.contourArea(quad)
    min_area = min_area_ratio * float(h * w)
    if area < min_area:
        return None, {"reason": "quad_too_small", "area": float(area)}

    in_bounds = np.all((quad[:, 0] >= -0.1 * w) & (quad[:, 0] <= 1.1 * w) & (quad[:, 1] >= -0.1 * h) & (quad[:, 1] <= 1.1 * h))
    if not in_bounds:
        return None, {"reason": "quad_out_of_bounds"}

    return quad, {
        "reason": "ok",
        "hough_lines": int(len(lines)),
        "vertical_segments": int(len(vertical_segments)),
        "horizontal_segments": int(len(horizontal_segments)),
        "area": float(area),
    }


def _expand_quad(quad: np.ndarray, scale: float, shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    center = np.mean(quad, axis=0)
    expanded = center + (quad - center) * scale
    expanded[:, 0] = np.clip(expanded[:, 0], 0, w - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, h - 1)
    return expanded.astype(np.float32)


def _sample_gradient(grad_mag: np.ndarray, point: np.ndarray) -> float:
    h, w = grad_mag.shape
    x = int(np.clip(round(float(point[0])), 0, w - 1))
    y = int(np.clip(round(float(point[1])), 0, h - 1))
    return float(grad_mag[y, x])


def _find_valley_after_peak(profile: np.ndarray) -> Tuple[int, int]:
    if profile.size == 0:
        return 0, 0

    max_val = float(np.max(profile))
    strong_threshold = max(5.0, max_val * 0.6)
    peak_idx = int(np.argmax(profile))

    for idx in range(1, len(profile) - 1):
        is_local_peak = profile[idx] >= profile[idx - 1] and profile[idx] > profile[idx + 1]
        if is_local_peak and profile[idx] >= strong_threshold:
            peak_idx = idx
            break

    valley_idx = peak_idx
    valley_threshold = float(profile[peak_idx] * 0.45)
    for idx in range(peak_idx + 1, len(profile) - 1):
        is_local_valley = profile[idx] <= profile[idx - 1] and profile[idx] < profile[idx + 1]
        if is_local_valley and profile[idx] <= valley_threshold:
            valley_idx = idx
            break

    if valley_idx == peak_idx and peak_idx + 1 < len(profile):
        valley_idx = int(peak_idx + 1 + np.argmin(profile[peak_idx + 1 :]))

    return peak_idx, valley_idx


def _expand_quad_to_card_boundary(gray: np.ndarray, quad: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    h, w = gray.shape
    quad = _order_quad_points(quad)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)

    max_steps = max(12, int(0.12 * min(h, w)))
    side_points: list[np.ndarray] = []
    debug_samples: list[Dict[str, tuple[float, float]]] = []

    for idx in range(4):
        p0 = quad[idx]
        p1 = quad[(idx + 1) % 4]
        edge = p1 - p0
        length = float(np.linalg.norm(edge))
        if length < 1e-6:
            side_points.append(np.array([p0], dtype=np.float32))
            continue

        tangent = edge / length
        normal = np.array([tangent[1], -tangent[0]], dtype=np.float32)

        num_samples = max(16, int(length / 20))
        ts = np.linspace(0.05, 0.95, num_samples)
        moved_points: list[np.ndarray] = []

        for t in ts:
            base = p0 + (edge * float(t))
            profile = np.zeros(max_steps + 1, dtype=np.float32)

            for step in range(max_steps + 1):
                profile[step] = _sample_gradient(grad_mag, base + normal * step)

            profile = cv2.GaussianBlur(profile.reshape(1, -1), (1, 5), 0).reshape(-1)
            peak_idx, valley_idx = _find_valley_after_peak(profile)

            peak_point = base + normal * float(peak_idx)
            valley_point = base + normal * float(valley_idx)
            moved_points.append(valley_point)
            debug_samples.append(
                {
                    "start": (float(base[0]), float(base[1])),
                    "end": (float((base + normal * max_steps)[0]), float((base + normal * max_steps)[1])),
                    "peak": (float(peak_point[0]), float(peak_point[1])),
                    "valley": (float(valley_point[0]), float(valley_point[1])),
                }
            )

        if moved_points:
            side_points.append(np.array(moved_points, dtype=np.float32))
        else:
            side_points.append(np.array([p0, p1], dtype=np.float32))

    lines = [_line_from_points(points) for points in side_points]
    if any(line is None for line in lines):
        return quad, {"samples": debug_samples}

    tl = _line_intersection(lines[0], lines[3])
    tr = _line_intersection(lines[1], lines[0])
    br = _line_intersection(lines[2], lines[1])
    bl = _line_intersection(lines[3], lines[2])
    if any(corner is None for corner in [tl, tr, br, bl]):
        return quad, {"samples": debug_samples}

    expanded = _order_quad_points(np.array([tl, tr, br, bl], dtype=np.float32))
    expanded[:, 0] = np.clip(expanded[:, 0], 0, w - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, h - 1)

    if cv2.contourArea(expanded) < cv2.contourArea(quad) * 0.9:
        return quad, {"samples": debug_samples}
    return expanded, {"samples": debug_samples}


def _warp_card(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    dst = np.array(
        [[0, 0], [WARP_WIDTH - 1, 0], [WARP_WIDTH - 1, WARP_HEIGHT - 1], [0, WARP_HEIGHT - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(image, matrix, (WARP_WIDTH, WARP_HEIGHT), flags=cv2.INTER_LINEAR)


def _draw_outer_quad_labels(
    image: np.ndarray,
    initial_quad: np.ndarray,
    expanded_quad: np.ndarray,
    expansion_debug: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    debug = image.copy()
    labels = ["TL", "TR", "BR", "BL"]
    colors = [(0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0)]

    initial_points = initial_quad.astype(int)
    expanded_points = expanded_quad.astype(int)

    cv2.polylines(debug, [initial_points], True, (0, 255, 255), 2)
    cv2.polylines(debug, [expanded_points], True, (0, 255, 0), 2)

    for (x, y), label, color in zip(expanded_points, labels, colors):
        cv2.circle(debug, (x, y), 8, color, -1)
        cv2.putText(debug, label, (x + 10, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(debug, "initial", tuple(initial_points[0] + np.array([10, 20])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(debug, "expanded", tuple(expanded_points[0] + np.array([10, -10])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    for sample in (expansion_debug or {}).get("samples", []):
        start = tuple(np.round(sample["start"]).astype(int))
        end = tuple(np.round(sample["end"]).astype(int))
        peak = tuple(np.round(sample["peak"]).astype(int))
        valley = tuple(np.round(sample["valley"]).astype(int))
        cv2.line(debug, start, end, (255, 128, 0), 1)
        cv2.circle(debug, peak, 3, (0, 165, 255), -1)
        cv2.circle(debug, valley, 4, (0, 255, 0), -1)

    return debug


def _draw_warp_preview_border(warped: np.ndarray) -> np.ndarray:
    preview = warped.copy()
    h, w = preview.shape[:2]
    cv2.rectangle(preview, (0, 0), (w - 1, h - 1), (0, 255, 0), 3)
    return preview


def _detect_frame_by_contour(warped: np.ndarray) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 130)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = gray.shape
    card_area = float(h * w)
    candidates: list[tuple[float, np.ndarray]] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 0.12 * card_area or area > 0.92 * card_area:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) != 4:
            continue

        quad = _order_quad_points(approx.reshape(4, 2).astype(np.float32))
        x_min, y_min = np.min(quad, axis=0)
        x_max, y_max = np.max(quad, axis=0)
        if x_min < 4 or y_min < 4 or x_max > w - 5 or y_max > h - 5:
            continue

        box = cv2.minAreaRect(contour)
        box_area = cv2.contourArea(cv2.boxPoints(box))
        if box_area <= 0:
            continue
        rectangularity = float(area / box_area)
        score = area * rectangularity
        candidates.append((score, quad))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


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


def _draw_debug(
    warped: np.ndarray,
    card_quad: np.ndarray,
    frame_quad: np.ndarray,
    used_fallback_frame: bool,
) -> np.ndarray:
    debug = warped.copy()
    h, w = debug.shape[:2]

    for x, y in card_quad.astype(int):
        cv2.circle(debug, (x, y), 8, (0, 255, 255), -1)

    frame = frame_quad.astype(int)
    cv2.polylines(debug, [frame], True, (0, 255, 0), 2)

    left_x, right_x = int(frame[0][0]), int(frame[1][0])
    top_y, bottom_y = int(frame[0][1]), int(frame[3][1])

    mid_y = h // 2
    mid_x = w // 2
    cv2.line(debug, (0, mid_y), (left_x, mid_y), (255, 0, 0), 2)
    cv2.line(debug, (right_x, mid_y), (w - 1, mid_y), (255, 0, 0), 2)
    cv2.line(debug, (mid_x, 0), (mid_x, top_y), (0, 0, 255), 2)
    cv2.line(debug, (mid_x, bottom_y), (mid_x, h - 1), (0, 0, 255), 2)

    label = "frame: border-color fallback" if used_fallback_frame else "frame: contour"
    cv2.putText(debug, label, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    return debug


def analyze_centering(image_bgr: np.ndarray) -> Dict[str, Any]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    card_quad, card_meta = _quad_from_hough(gray, min_area_ratio=0.2)
    card_source = "outer_edge"

    if card_quad is None:
        inner_quad, inner_meta = _quad_from_hough(gray, min_area_ratio=0.08)
        if inner_quad is None:
            return {"error": "Card could not be detected", "details": {"outer": card_meta, "inner": inner_meta}}
        card_quad = _expand_quad(inner_quad, scale=1.18, shape=gray.shape)
        card_source = "inner_frame_fallback"

    initial_quad = _order_quad_points(card_quad)
    expanded_quad, expansion_debug = _expand_quad_to_card_boundary(gray, initial_quad)
    warped = _warp_card(image_bgr, expanded_quad)
    warped_preview = _draw_warp_preview_border(warped)

    frame_quad = _detect_frame_by_contour(warped)
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
        "outer_quad": [[round(float(x), 2), round(float(y), 2)] for x, y in expanded_quad],
        "initial_outer_quad": [[round(float(x), 2), round(float(y), 2)] for x, y in initial_quad],
        "frame_detection": "border_color_fallback" if used_fallback_frame else "contour",
        "warped_image": warped_preview,
        "warped_card": warped_preview,
        "debug_image": _draw_outer_quad_labels(image_bgr, initial_quad, expanded_quad, expansion_debug),
        "warped_debug_image": _draw_debug(
            warped=warped_preview,
            card_quad=np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32),
            frame_quad=frame,
            used_fallback_frame=used_fallback_frame,
        ),
    }
    return result
