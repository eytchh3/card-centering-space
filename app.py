from __future__ import annotations

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger(__name__)

import cv2
import gradio as gr

from detector import analyze_centering


def _format_result(result: dict) -> str:
    if "error" in result:
        return result["error"]

    return "\n".join(
        [
            "Centering analysis complete",
            f"Left border: {result['left_border_px']}px",
            f"Right border: {result['right_border_px']}px",
            f"Top border: {result['top_border_px']}px",
            f"Bottom border: {result['bottom_border_px']}px",
            f"Centering L/R: {result['centering_lr']}",
            f"Centering T/B: {result['centering_tb']}",
            f"Card detection: {result['card_detection']}",
            f"Frame detection: {result['frame_detection']}",
        ]
    )


def run_detector(image):
    if image is None:
        return "Error: No image", None, None

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    result = analyze_centering(bgr)

    if "error" in result:
        return "Error: Card not detected", None, None

    text = _format_result(result)
    debug = result.get("debug_image")
    warped_card = result.get("warped_card")

    debug_rgb = cv2.cvtColor(debug, cv2.COLOR_BGR2RGB) if debug is not None else None
    warped_rgb = cv2.cvtColor(warped_card, cv2.COLOR_BGR2RGB) if warped_card is not None else None

    return text, debug_rgb, warped_rgb


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

If card detection fails, the API returns: `{"error": "Card could not be detected"}`.
"""
    )

    with gr.Row():
        inp = gr.Image(type="numpy", label="Card Photo")
        out_img = gr.Image(type="numpy", label="Debug Overlay")
        out_warp = gr.Image(type="numpy", label="Warped Card Preview")
    out_text = gr.Textbox(label="Centering Result", lines=10)

    btn = gr.Button("Analyze")
    btn.click(fn=run_detector, inputs=inp, outputs=[out_text, out_img, out_warp])

demo.queue()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
