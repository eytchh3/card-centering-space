import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from detector import _select_edge_horizontal_line, analyze_centering


def make_card_image(offset_x=0, offset_y=0, perspective=False):
    img = np.full((900, 900, 3), 35, dtype=np.uint8)

    card = np.full((720, 520, 3), 215, dtype=np.uint8)
    left = 65 + offset_x
    right = 520 - 65 + offset_x
    top = 80 + offset_y
    bottom = 720 - 80 + offset_y
    cv2.rectangle(card, (left, top), (right, bottom), (80, 80, 80), 3)

    src = np.array([[0, 0], [519, 0], [519, 719], [0, 719]], dtype=np.float32)
    dst = np.array([[180, 90], [720, 110], [710, 820], [170, 800]], dtype=np.float32)

    if perspective:
        dst = np.array([[210, 120], [700, 80], [730, 810], [145, 840]], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(src, dst)
    warped_card = cv2.warpPerspective(card, matrix, (900, 900))

    mask = np.any(warped_card > 0, axis=2)
    img[mask] = warped_card[mask]
    return img


def test_returns_expected_measurement_keys():
    img = make_card_image(offset_x=0, offset_y=0)
    result = analyze_centering(img)

    assert "error" not in result
    for key in [
        "left_border_px",
        "right_border_px",
        "top_border_px",
        "bottom_border_px",
        "centering_lr",
        "centering_tb",
        "debug_image",
        "warped_card",
    ]:
        assert key in result


def test_offcenter_card_detects_lr_imbalance():
    img = make_card_image(offset_x=30)
    result = analyze_centering(img)

    assert "error" not in result
    assert abs(result["left_border_px"] - result["right_border_px"]) > 10


def test_perspective_photo_supported():
    img = make_card_image(offset_x=10, offset_y=-8, perspective=True)
    result = analyze_centering(img)

    assert "error" not in result
    assert isinstance(result["centering_lr"], str)
    assert isinstance(result["centering_tb"], str)



def test_warped_card_has_fixed_size():
    img = make_card_image()
    result = analyze_centering(img)

    assert "error" not in result
    assert result["warped_card"].shape[:2] == (1000, 700)

def test_blank_image_returns_error():
    img = np.zeros((600, 600, 3), dtype=np.uint8)
    result = analyze_centering(img)
    assert result["error"] == "Card could not be detected"


def test_select_edge_horizontal_line_prefers_edge_nearby_candidates():
    lines = [
        np.array([10, 35, 690, 35], dtype=np.float32),
        np.array([12, 120, 688, 120], dtype=np.float32),
        np.array([8, 955, 692, 955], dtype=np.float32),
        np.array([15, 860, 685, 860], dtype=np.float32),
    ]

    top = _select_edge_horizontal_line(lines, card_height=1000, edge="top", min_inset_px=8)
    bottom = _select_edge_horizontal_line(lines, card_height=1000, edge="bottom", min_inset_px=8)

    assert top == 35
    assert bottom == 955


def test_select_edge_horizontal_line_rejects_too_deep_lines():
    lines = [
        np.array([10, 420, 690, 420], dtype=np.float32),
        np.array([10, 580, 690, 580], dtype=np.float32),
    ]

    top = _select_edge_horizontal_line(lines, card_height=1000, edge="top", min_inset_px=8)
    bottom = _select_edge_horizontal_line(lines, card_height=1000, edge="bottom", min_inset_px=8)

    assert top is None
    assert bottom is None
