from __future__ import annotations

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger(__name__)

import cv2
import gradio as gr
import numpy as np

from detector import analyze_centering


def run_detector(image: np.ndarray):
    if image is None:
        return "Error", "Card not detected", None

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    result = analyze_centering(bgr)

    if "error" in result:
        return "Error", "Card not detected", None

    debug = result.get("debug_image")
    debug_rgb = cv2.cvtColor(debug, cv2.COLOR_BGR2RGB) if debug is not None else None
    return result.get("centering_lr", "Error"), result.get("centering_tb", "Card not detected"), debug_rgb


with gr.Blocks(title="Trading Card Centering Detector") as demo:
    gr.Markdown(
        """
# Trading Card Centering Detector
Classical CV centering pipeline for listing photos and scans:
1. Detects card corners with edges + Hough lines.
2. Applies perspective correction.
3. Finds the printed frame line.
4. Measures edge-to-frame borders and reports centering ratios.
5. Returns a debug overlay showing corners, frame, and measurements.

If card detection fails, outputs are: `Error`, `Card not detected`, and no debug image.
"""
    )

    with gr.Row():
        inp = gr.Image(type="numpy", label="Card Photo")

    with gr.Row():
        out_lr = gr.Textbox(label="Left/Right Centering")
        out_tb = gr.Textbox(label="Top/Bottom Centering")

    out_img = gr.Image(type="numpy", label="Debug Overlay")

    btn = gr.Button("Analyze")
    btn.click(fn=run_detector, inputs=inp, outputs=[out_lr, out_tb, out_img], api_name="/analyze")

demo.queue()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
