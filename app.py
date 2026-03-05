from __future__ import annotations

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger(__name__)

import cv2
import gradio as gr

from detector import analyze_centering


def run_detector(image):
    if image is None:
        return "Error", "Card not detected", None, None

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    result = analyze_centering(bgr)

    if not isinstance(result, dict) or "error" in result:
        return "Error", "Card not detected", None, None

    centering_lr = str(result.get("centering_lr", "Error"))
    centering_tb = str(result.get("centering_tb", "Card not detected"))

    debug_image = result.get("debug_image")
    warped_image = result.get("warped_image")

    if debug_image is not None:
        debug_image = cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)
    if warped_image is not None:
        warped_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB)

    return centering_lr, centering_tb, debug_image, warped_image


with gr.Blocks(title="Trading Card Centering Detector") as demo:
    gr.Markdown(
        """
# Trading Card Centering Detector
Upload a card photo to estimate centering.
Returns explicit Left/Right and Top/Bottom centering strings, an outer-detection debug overlay, and warped card preview.
"""
    )

    inp = gr.Image(type="numpy", label="Card Photo")

    outputs = [
        gr.Textbox(label="Left/Right Centering"),
        gr.Textbox(label="Top/Bottom Centering"),
        gr.Image(label="Debug Overlay"),
        gr.Image(label="Warped Card Preview"),
    ]

    btn = gr.Button("Analyze")
    btn.click(fn=run_detector, inputs=inp, outputs=outputs, api_name="/analyze")

demo.queue()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
