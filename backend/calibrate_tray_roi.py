import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to an empty-tray image.")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).parent / "tray_roi.json"),
        help="Output ROI json path.",
    )
    parser.add_argument("--output-width", type=int, default=0, help="Optional warp output width.")
    parser.add_argument("--output-height", type=int, default=0, help="Optional warp output height.")
    return parser.parse_args()


def order_points(points: np.ndarray) -> np.ndarray:
    s = points.sum(axis=1)
    diff = np.diff(points, axis=1).reshape(-1)
    tl = points[np.argmin(s)]
    br = points[np.argmax(s)]
    tr = points[np.argmin(diff)]
    bl = points[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def main():
    args = parse_args()
    img_path = Path(args.image).resolve()
    if not img_path.exists():
        print(f"Image not found: {img_path}")
        raise SystemExit(1)

    image = cv2.imread(str(img_path))
    if image is None:
        print(f"Failed to read image: {img_path}")
        raise SystemExit(1)

    h, w = image.shape[:2]
    window = "Tray ROI Calibration (Left click: add, Right click: undo, R: reset, Enter: save, Esc: cancel)"
    points: list[list[int]] = []

    def redraw():
        canvas = image.copy()
        for i, (x, y) in enumerate(points):
            cv2.circle(canvas, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(
                canvas,
                str(i + 1),
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        if len(points) >= 2:
            pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], False, (255, 255, 0), 2)
        cv2.putText(
            canvas,
            f"Points: {len(points)}/4",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window, canvas)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([int(x), int(y)])
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()
            redraw()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    redraw()

    saved = False
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            break
        if key in (ord("r"),):
            points.clear()
            redraw()
        if key in (13, 10, ord("s"), ord(" ")):
            if len(points) != 4:
                print("Please click exactly 4 points before saving.")
                continue
            ordered = order_points(np.array(points, dtype=np.float32))
            data = {
                "points": [[float(p[0]), float(p[1])] for p in ordered],
                "source_size": [int(w), int(h)],
            }
            if args.output_width > 0 and args.output_height > 0:
                data["output_size"] = [int(args.output_width), int(args.output_height)]
            out_path = Path(args.out).resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Saved ROI config: {out_path}")
            print(json.dumps(data, ensure_ascii=False))
            saved = True
            break

    cv2.destroyAllWindows()
    if not saved:
        print("Calibration canceled.")


if __name__ == "__main__":
    main()
