from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

CARD_ASPECT_RATIO = 1.4


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
    ordered_quad = _order_quad_points(quad)
    width = int(round(float(np.linalg.norm(ordered_quad[1] - ordered_quad[0]))))
    width = max(width, 1)
    height = int(round(width * CARD_ASPECT_RATIO))
    height = max(height, 1)

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered_quad.astype(np.float32), dst)
    return cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_LINEAR)


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


def _line_length(line: np.ndarray) -> float:
    x1, y1, x2, y2 = map(float, line)
    return float(np.hypot(x2 - x1, y2 - y1))


def _line_angle(line: np.ndarray) -> float:
    x1, y1, x2, y2 = map(float, line)
    return float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0)


def _select_edge_horizontal_line(
    horizontal_lines: list[np.ndarray],
    card_height: int,
    edge: str,
    min_inset_px: int = 8,
) -> Optional[float]:
    if not horizontal_lines:
        return None

    y_candidates = np.array([(line[1] + line[3]) * 0.5 for line in horizontal_lines], dtype=np.float32)
    if y_candidates.size == 0:
        return None

    min_y = float(min_inset_px)
    max_y = float((card_height - 1) - min_inset_px)
    if edge == "top":
        valid = y_candidates[(y_candidates >= min_y) & (y_candidates <= 0.4 * card_height)]
        if valid.size == 0:
            return None
        return float(np.min(valid))

    if edge == "bottom":
        valid = y_candidates[(y_candidates <= max_y) & (y_candidates >= 0.6 * card_height)]
        if valid.size == 0:
            return None
        return float(np.max(valid))

    return None


def _detect_frame_by_contour(warped: np.ndarray) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 35, 120)

    h, w = gray.shape
    min_line_length = max(20, int(0.18 * w))
    hough_threshold = max(25, int(0.07 * min(h, w)))
    raw_lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max(18, int(0.04 * w)),
    )

    all_lines = [] if raw_lines is None else [line[0].astype(np.float32) for line in raw_lines]
    if len(all_lines) < 8:
        reduced_threshold = max(15, int(hough_threshold * 0.7))
        raw_lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=reduced_threshold,
            minLineLength=min_line_length,
            maxLineGap=max(18, int(0.04 * w)),
        )
        all_lines = [] if raw_lines is None else [line[0].astype(np.float32) for line in raw_lines]

    angle_tolerance = 10.0
    vertical_lines: list[np.ndarray] = []
    horizontal_lines: list[np.ndarray] = []
    line_angles: list[float] = []
    for line in all_lines:
        angle = _line_angle(line)
        line_angles.append(angle)
        if min(abs(angle - 90.0), abs(angle - 270.0)) <= angle_tolerance:
            vertical_lines.append(line)
        elif min(abs(angle - 0.0), abs(angle - 180.0)) <= angle_tolerance:
            horizontal_lines.append(line)

    lengths = [_line_length(line) for line in all_lines]
    avg_length = float(np.mean(lengths)) if lengths else 0.0

    rejected_rectangles: list[np.ndarray] = []
    frame_quad: Optional[np.ndarray] = None
    confidence_score = 0.0

    if len(vertical_lines) >= 2 and len(horizontal_lines) >= 2:
        x_mids = np.array([(line[0] + line[2]) * 0.5 for line in vertical_lines], dtype=np.float32)
        left_x = float(np.quantile(x_mids, 0.2))
        right_x = float(np.quantile(x_mids, 0.8))
        top_y = _select_edge_horizontal_line(horizontal_lines, h, edge="top", min_inset_px=8)
        bottom_y = _select_edge_horizontal_line(horizontal_lines, h, edge="bottom", min_inset_px=8)

        if top_y is None or bottom_y is None or top_y >= bottom_y:
            top_y = None
            bottom_y = None

        if top_y is not None and bottom_y is not None:
            candidate = _order_quad_points(
                np.array([[left_x, top_y], [right_x, top_y], [right_x, bottom_y], [left_x, bottom_y]], dtype=np.float32)
            )
            candidate_area = float(cv2.contourArea(candidate))
            min_area = 0.06 * float(h * w)
            in_bounds = np.all(
                (candidate[:, 0] >= -0.12 * w)
                & (candidate[:, 0] <= 1.12 * w)
                & (candidate[:, 1] >= -0.12 * h)
                & (candidate[:, 1] <= 1.12 * h)
            )
            candidate_width = max(1e-6, right_x - left_x)
            candidate_height = max(1e-6, bottom_y - top_y)
            aspect = candidate_height / candidate_width
            aspect_score = 1.0 - min(0.4, abs(aspect - CARD_ASPECT_RATIO) / CARD_ASPECT_RATIO)
            coverage_score = min(1.0, candidate_area / max(min_area, 1.0))
            confidence_score = float(max(0.0, min(1.0, 0.55 * coverage_score + 0.45 * aspect_score)))

            if candidate_area >= min_area and in_bounds and confidence_score >= 0.45:
                frame_quad = candidate
            else:
                rejected_rectangles.append(candidate)

    print("Frame detection debug:")
    print("vertical lines:", len(vertical_lines))
    print("horizontal lines:", len(horizontal_lines))
    print("avg line length:", round(avg_length, 2))
    print("line angles:", [round(angle, 1) for angle in line_angles[:20]])
    print("confidence:", round(confidence_score, 3))

    if frame_quad is not None:
        return frame_quad, {
            "method": "frame_lines",
            "vertical_lines": vertical_lines,
            "horizontal_lines": horizontal_lines,
            "rejected_rectangles": rejected_rectangles,
            "confidence": confidence_score,
        }

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, {
            "method": "none",
            "vertical_lines": vertical_lines,
            "horizontal_lines": horizontal_lines,
            "rejected_rectangles": rejected_rectangles,
            "confidence": confidence_score,
        }

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
        return None, {
            "method": "none",
            "vertical_lines": vertical_lines,
            "horizontal_lines": horizontal_lines,
            "rejected_rectangles": rejected_rectangles,
            "confidence": confidence_score,
        }
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1], {
        "method": "contour",
        "vertical_lines": vertical_lines,
        "horizontal_lines": horizontal_lines,
        "rejected_rectangles": rejected_rectangles,
        "confidence": confidence_score,
    }


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
    frame_debug: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    debug = warped.copy()
    h, w = debug.shape[:2]

    for x, y in card_quad.astype(int):
        cv2.circle(debug, (x, y), 8, (0, 255, 255), -1)

    frame = frame_quad.astype(int)
    cv2.polylines(debug, [frame], True, (0, 255, 0), 2)

    for line in (frame_debug or {}).get("vertical_lines", []):
        x1, y1, x2, y2 = map(int, np.round(line))
        cv2.line(debug, (x1, y1), (x2, y2), (255, 0, 0), 2)

    for line in (frame_debug or {}).get("horizontal_lines", []):
        x1, y1, x2, y2 = map(int, np.round(line))
        cv2.line(debug, (x1, y1), (x2, y2), (255, 255, 0), 2)

    for rejected in (frame_debug or {}).get("rejected_rectangles", []):
        cv2.polylines(debug, [rejected.astype(int)], True, (0, 0, 255), 1)

    left_x, right_x = int(frame[0][0]), int(frame[1][0])
    top_y, bottom_y = int(frame[0][1]), int(frame[3][1])

    mid_y = h // 2
    mid_x = w // 2
    cv2.line(debug, (0, mid_y), (left_x, mid_y), (255, 0, 0), 2)
    cv2.line(debug, (right_x, mid_y), (w - 1, mid_y), (255, 0, 0), 2)
    cv2.line(debug, (mid_x, 0), (mid_x, top_y), (0, 0, 255), 2)
    cv2.line(debug, (mid_x, bottom_y), (mid_x, h - 1), (0, 0, 255), 2)

    method = (frame_debug or {}).get("method", "frame_lines")
    label = "frame: border-color fallback" if used_fallback_frame else f"frame: {method}"
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
    warped_output = cv2.resize(warped_preview, (700, 1000), interpolation=cv2.INTER_LINEAR)

    frame_quad, frame_debug = _detect_frame_by_contour(warped)
    used_fallback_frame = False
    if frame_quad is None:
        frame_quad = _estimate_frame_by_border_color(warped)
        used_fallback_frame = True
        frame_debug = {"method": "border_color_fallback", **(frame_debug or {})}

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
        "frame_detection": "border_color_fallback" if used_fallback_frame else frame_debug.get("method", "frame_lines"),
        "warped_image": warped_output,
        "warped_card": warped_output,
        "debug_image": _draw_outer_quad_labels(image_bgr, initial_quad, expanded_quad, expansion_debug),
        "warped_debug_image": _draw_debug(
            warped=warped_preview,
            card_quad=np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32),
            frame_quad=frame,
            used_fallback_frame=used_fallback_frame,
            frame_debug=frame_debug,
        ),
    }
    return result
