from __future__ import annotations

import cv2
import gradio as gr
import numpy as np

from detector import PSA_MIN_RATIO, UNCERTAIN_MSG, analyze_centering


def _format_result(status: str, confidence: float, lr_ratio: float | None, tb_ratio: float | None) -> str:
    lines = [f"Result: {status}", f"Confidence: {confidence:.2f}"]
    if lr_ratio is not None and tb_ratio is not None:
        lines.append(f"L/R ratio (smaller/larger): {lr_ratio:.3f}")
        lines.append(f"T/B ratio (smaller/larger): {tb_ratio:.3f}")
        lines.append(f"PSA 55/45 minimum ratio: {PSA_MIN_RATIO:.3f}")
    if status == UNCERTAIN_MSG:
        lines.append("Tip: use a sharper, evenly-lit photo with the full card visible and minimal glare.")
    return "\n".join(lines)


def run_detector(image: np.ndarray):
    if image is None:
        return "Please upload an image.", None

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    result = analyze_centering(bgr)

    text = _format_result(result.status, result.confidence, result.lr_ratio, result.tb_ratio)

    if result.overlay is None:
        return text, None

    overlay_rgb = cv2.cvtColor(result.overlay, cv2.COLOR_BGR2RGB)
    return text, overlay_rgb


with gr.Blocks(title="Trading Card Centering Detector") as demo:
    gr.Markdown(
        """
# Trading Card Centering Detector
Upload a photo of a modern chrome-style trading card. The detector will:
1. Find the outer card and perspective-flatten it.
2. Detect inner frame rails without relying on text/nameplate alignment.
3. Compute left/right and top/bottom gap ratios.
4. Return PASS only when both axes meet PSA 55/45 (ratio ≥ 0.818).

If confidence is low, the app returns: **UNCERTAIN – insufficient photo quality.**
"""
    )

    with gr.Row():
        inp = gr.Image(type="numpy", label="Card Photo")
        out_img = gr.Image(type="numpy", label="Detected Rails Overlay")
    out_text = gr.Textbox(label="Centering Result", lines=8)

    btn = gr.Button("Analyze")
    btn.click(
        fn=run_detector,
        inputs=inp,
        outputs=[out_text, out_img],
        api_name="analyze",
    )

demo.queue()


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
