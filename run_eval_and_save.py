from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


LOG_DIR = Path("artifacts") / "eval_logs"
EVAL_SCRIPT = Path("eval_centering.py")

FOUND_RE = re.compile(r"^Found (?P<count>\d+) fixture image\(s\)")
COMPLETED_RE = re.compile(
    r"^Completed evaluation\. Successful detections: (?P<success>\d+)/(?P<total>\d+)"
)
STATUS_RE = re.compile(r"^\[(?P<status>OK|FAIL|SKIP)\] (?P<image>.*?): (?P<detail>.*)$")


def parse_summary(log_text: str, return_code: int, command: list[str]) -> dict:
    images: list[dict[str, str]] = []
    found_total: int | None = None
    completed_total: int | None = None
    successful_detections: int | None = None

    for raw_line in log_text.splitlines():
        line = raw_line.strip()

        found_match = FOUND_RE.match(line)
        if found_match:
            found_total = int(found_match.group("count"))

        completed_match = COMPLETED_RE.match(line)
        if completed_match:
            successful_detections = int(completed_match.group("success"))
            completed_total = int(completed_match.group("total"))

        status_match = STATUS_RE.match(line)
        if status_match:
            images.append(
                {
                    "image": status_match.group("image"),
                    "status": status_match.group("status"),
                    "detail": status_match.group("detail"),
                }
            )

    status_counts = {
        "ok": sum(1 for item in images if item["status"] == "OK"),
        "fail": sum(1 for item in images if item["status"] == "FAIL"),
        "skip": sum(1 for item in images if item["status"] == "SKIP"),
    }

    inferred_total = found_total if found_total is not None else len(images)

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "command": command,
        "return_code": return_code,
        "totals": {
            "fixtures_found": found_total,
            "images_reported": len(images),
            "successful_detections": successful_detections,
            "completed_total": completed_total,
            "ok": status_counts["ok"],
            "fail": status_counts["fail"],
            "skip": status_counts["skip"],
            "inferred_total": inferred_total,
        },
        "per_image": images,
    }


def build_summary_lines(summary: dict, txt_path: Path, json_path: Path) -> list[str]:
    totals = summary["totals"]
    total_images = totals["fixtures_found"]
    if total_images is None:
        total_images = totals["completed_total"]
    if total_images is None:
        total_images = totals["images_reported"]

    fail_images = [
        item["image"] for item in summary["per_image"] if item["status"] == "FAIL"
    ]

    lines = [
        "=== Evaluation summary ===",
        f"Total images: {total_images} | Pass: {totals['ok']} | Fail: {totals['fail']}",
    ]

    if fail_images:
        lines.append("Failing files (top 10):")
        lines.extend([f"- {image}" for image in fail_images[:10]])
    else:
        lines.append("Failing files (top 10): none")

    if total_images == 0:
        lines.append(
            "No fixtures found. Place evaluation images under "
            "'fixtures/centering/' (or pass --fixtures-dir to eval_centering.py)."
        )

    lines.append(f"Saved log: {txt_path.as_posix()}")
    lines.append(f"Saved json: {json_path.as_posix()}")

    return lines


def run_eval_and_save() -> dict:
    if not EVAL_SCRIPT.exists():
        raise FileNotFoundError(f"expected '{EVAL_SCRIPT}' at repository root")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = LOG_DIR / f"eval_centering_{timestamp}.txt"
    json_path = LOG_DIR / f"eval_centering_{timestamp}.json"

    cmd = [sys.executable, str(EVAL_SCRIPT)]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    captured_lines: list[str] = []
    assert process.stdout is not None
    with txt_path.open("w", encoding="utf-8") as log_file:
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            captured_lines.append(line)

    return_code = process.wait()

    combined_output = "".join(captured_lines)
    summary = parse_summary(combined_output, return_code=return_code, command=cmd)
    with json_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)
        summary_file.write("\n")

    summary_lines = build_summary_lines(summary=summary, txt_path=txt_path, json_path=json_path)
    for line in summary_lines:
        print(line)

    return {
        "return_code": return_code,
        "summary": summary,
        "txt_path": txt_path,
        "json_path": json_path,
        "summary_lines": summary_lines,
        "raw_output": combined_output,
    }


def main() -> int:
    try:
        result = run_eval_and_save()
    except FileNotFoundError as exc:
        print(f"Error: {exc}.", file=sys.stderr)
        return 1

    return result["return_code"]


if __name__ == "__main__":
    raise SystemExit(main())
