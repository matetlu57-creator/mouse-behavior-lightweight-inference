"""Interactive picker for the cage-floor quadrilateral used by scale calibration.

The colour-band proposal plus gradient refinement gets within a few tens of
pixels on most edges, but on real footage two of the four floor/wall junctions
can be confused with shading steps on the wall, which is a 4-10% scale error.
Scale is the one input every centimetre threshold inherits, so the corners are
confirmed by a human instead of trusted to the detector.

Usage
-----
    python scripts/pick_arena_corners.py --video clip.mp4 --frame 150 \
        --out outputs/scale/rig_a.json --width-cm 60 --height-cm 45

Drag the four handles onto the floor/wall junction, press Enter to save and
Esc to quit without writing.  ``a`` re-runs the automatic proposal, ``r``
resets to the last saved quad when ``--out`` already exists.  Without
``--width-cm``/``--height-cm`` only the quad and diagnostics are stored; add
the tape-measured inner size later with::

    python scripts/pick_arena_corners.py --fill-size outputs/scale/rig_a.json \
        --width-cm 60 --height-cm 45
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mouse_behavior.preprocessing import scale_calibration as sc  # noqa: E402

WINDOW = "pick cage floor (drag handles, Enter=save, Esc=quit, a=auto)"
HANDLE_RADIUS = 14


def _read_frame(video: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SystemExit(f"cannot open video: {video}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise SystemExit(f"cannot read frame {frame_index} from {video}")
    return frame


def _auto_quad(frame: np.ndarray) -> np.ndarray | None:
    coarse = sc.propose_floor_quad(frame)
    if coarse is None:
        return None
    return sc.refine_quad_to_edges(frame, coarse)


def _draw(frame: np.ndarray, quad: np.ndarray, box_cm: tuple[float, float] | None) -> np.ndarray:
    overlay = frame.copy()
    ordered = sc.order_quad_corners(quad).astype(int)
    cv2.polylines(overlay, [ordered.reshape(-1, 1, 2)], True, (0, 255, 0), 3)
    edges = sc.quad_edge_lengths(quad)
    for index, name in enumerate(sc.CORNER_ORDER):
        x, y = ordered[index]
        cv2.circle(overlay, (int(x), int(y)), HANDLE_RADIUS, (0, 0, 255), -1)
        cv2.putText(overlay, name, (int(x) + 16, int(y) + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    text = (
        f"top={edges['top_px']:.0f} bottom={edges['bottom_px']:.0f} "
        f"left={edges['left_px']:.0f} right={edges['right_px']:.0f} px"
    )
    if box_cm:
        scale = sc.cm_per_pixel_from_quad(quad, box_cm[0], box_cm[1])
        text += (
            f" | cm/px={scale['cm_per_pixel']:.5f}"
            f" axis_ratio={scale['axis_consistency_ratio']:.4f}"
        )
    cv2.putText(overlay, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return overlay


def run_picker(frame: np.ndarray, initial: np.ndarray, box_cm: tuple[float, float] | None) -> np.ndarray | None:
    state = {"quad": sc.order_quad_corners(initial), "drag": -1}

    def on_mouse(event: int, x: int, y: int, _flags: int, _user: object) -> None:
        quad = state["quad"]
        if event == cv2.EVENT_LBUTTONDOWN:
            distances = np.linalg.norm(quad - np.asarray([x, y], dtype=float), axis=1)
            state["drag"] = int(np.argmin(distances)) if distances.min() <= 30 else -1
        elif event == cv2.EVENT_MOUSEMOVE and state["drag"] >= 0:
            quad[state["drag"]] = (float(x), float(y))
        elif event == cv2.EVENT_LBUTTONUP:
            state["drag"] = -1

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)
    while True:
        cv2.imshow(WINDOW, _draw(frame, state["quad"], box_cm))
        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # Esc
            cv2.destroyAllWindows()
            return None
        if key in (13, 10):  # Enter
            cv2.destroyAllWindows()
            return sc.order_quad_corners(state["quad"])
        if key == ord("a"):
            auto = _auto_quad(frame)
            if auto is not None:
                state["quad"] = auto


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width-cm", type=float, default=None)
    parser.add_argument("--height-cm", type=float, default=None)
    parser.add_argument("--fill-size", type=Path, default=None,
                        help="update an existing calibration file with the inner box size")
    args = parser.parse_args()

    if args.fill_size is not None:
        payload = sc.load_calibration(args.fill_size)
        if args.width_cm is None or args.height_cm is None:
            raise SystemExit("--fill-size requires --width-cm and --height-cm")
        quad = np.asarray(payload["quad_px"], dtype=float)
        filled = sc.build_calibration(
            quad,
            width_cm=args.width_cm,
            height_cm=args.height_cm,
            source_video=payload.get("source_video"),
            method=str(payload.get("method", "manual")),
        )
        sc.save_calibration(filled, args.fill_size)
        print(f"updated {args.fill_size}")
        print(f"cm_per_pixel={filled['scale']['cm_per_pixel']:.6f}")
        print(f"axis_consistency_ratio={filled['scale']['axis_consistency_ratio']:.4f}")
        return 0

    if args.video is None:
        raise SystemExit("--video is required unless --fill-size is used")
    frame = _read_frame(args.video, args.frame)
    box_cm = None
    if args.width_cm is not None and args.height_cm is not None:
        box_cm = (args.width_cm, args.height_cm)

    initial = None
    if args.out.exists():
        initial = np.asarray(sc.load_calibration(args.out)["quad_px"], dtype=float)
    if initial is None:
        initial = _auto_quad(frame)
    if initial is None:
        height, width = frame.shape[:2]
        initial = np.asarray(
            [
                [width * 0.25, height * 0.25],
                [width * 0.75, height * 0.25],
                [width * 0.75, height * 0.75],
                [width * 0.25, height * 0.75],
            ],
            dtype=float,
        )
        print("auto proposal failed; starting from a centred rectangle")

    confirmed = run_picker(frame, initial, box_cm)
    if confirmed is None:
        print("cancelled; nothing written")
        return 1
    payload = sc.build_calibration(
        confirmed,
        width_cm=args.width_cm if args.width_cm is not None else 1.0,
        height_cm=args.height_cm if args.height_cm is not None else 1.0,
        source_video=args.video,
        method="manual_confirmed",
    )
    if box_cm is None:
        payload["box_inner_cm"] = None
        payload["scale"] = None
        payload["homography"] = None
        payload["homography_px_per_cm"] = None
    sc.save_calibration(payload, args.out)
    print(f"wrote {args.out}")
    diagnostics = payload["diagnostics"]
    print("quad_px:", payload["quad_px"])
    print(f"max_relative_edge_deviation={diagnostics['max_relative_edge_deviation']:.4f}")
    if box_cm is not None:
        print(f"cm_per_pixel={payload['scale']['cm_per_pixel']:.6f}")
    else:
        print("box size not given; run --fill-size to add the tape-measured inner size")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())