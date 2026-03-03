import cv2
import numpy as np

from detector import UNCERTAIN_MSG, analyze_centering


def make_card_image(offset_x=0, offset_y=0):
    img = np.full((900, 900, 3), 30, dtype=np.uint8)

    # Outer card
    cv2.rectangle(img, (170, 90), (730, 810), (210, 210, 210), -1)

    # Inner frame rails (chrome style)
    left = 170 + 70 + offset_x
    right = 730 - 70 + offset_x
    top = 90 + 85 + offset_y
    bottom = 810 - 85 + offset_y

    cv2.rectangle(img, (left, top), (right, bottom), (110, 110, 110), 2)

    # subtle texture/noise
    noise = np.random.default_rng(0).integers(0, 15, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    return img


def test_centered_card_passes_or_high_ratio():
    img = make_card_image(0, 0)
    result = analyze_centering(img)
    assert result.status in {"PASS", "FAIL", UNCERTAIN_MSG}
    assert result.lr_ratio is None or result.lr_ratio > 0.80
    assert result.tb_ratio is None or result.tb_ratio > 0.80


def test_offcenter_card_fails_on_lr_ratio():
    img = make_card_image(offset_x=35)
    result = analyze_centering(img, conf_threshold=0.1)
    assert result.lr_ratio is not None
    assert result.lr_ratio < 0.8182


def test_blank_image_uncertain():
    img = np.zeros((600, 600, 3), dtype=np.uint8)
    result = analyze_centering(img)
    assert result.status == UNCERTAIN_MSG
