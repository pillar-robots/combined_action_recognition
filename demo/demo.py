#!/usr/bin/env python3
"""
Standalone demo for the Combined Action pipeline.

The script mirrors the behaviour of ``Final_action/inference2.py`` by running the combined
perception stack (person detection, pose classification, optional gaze estimation and
object interaction reasoning) on a video stream.

Usage example:
    python demo.py --video ../demo/samples/input.mp4 --output ../demo/samples/output.mp4

When ``--output`` is provided the annotated frames are written to disk. With ``--display``
set the demo also shows a live preview window.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2

from combined_action_pillar_core import (
    CombinedActionPipeline,
    CONF_THRESH,
    DEFAULT_IMGSZ,
    FACE_YOLO_WEIGHTS,
    IOU_THRESH,
    MODEL_PIPELINE_PATH,
    OBJECT_YOLO_WEIGHTS,
    POSE_TASK_PATH,
    YOLO_WEIGHTS,
)

def _apply_visibility_and_flatten(X_flat):
    """
    X_flat: (N, 33*4). Reshape to (N,33,4),
    multiply last channel (visibility) into xyz, keep xyz only (33,3),
    flatten to (N, 99).
    """
    X = X_flat.reshape(X_flat.shape[0], 33, 4)
    X_xyz = X[..., :3] * X[..., 3:4]   # multiply xyz by visibility
    return X_xyz.reshape(X_xyz.shape[0], -1)


def resolve_asset(args: argparse.Namespace, candidate: Optional[str], default_path: Optional[str]) -> Optional[str]:
    if candidate:
        return candidate
    if default_path is None:
        return None
    if args.models_root:
        return str(Path(args.models_root) / Path(default_path).name)
    return default_path


def format_person_summary(person: Dict[str, object]) -> str:
    gaze = person.get("gaze_target") or "none"
    interactions = ", ".join(person.get("interactions", [])) if person.get("interactions") else "none"
    conf = person.get("pose_confidence")
    confidence = f"{conf:.2f}" if conf is not None else "n/a"
    return f"P{person['person_id']} pose={person['pose_label']} (conf={confidence}) gaze={gaze} interactions={interactions}"


def summarise_aggregate(aggregate: Dict[int, Dict[str, List]]) -> Dict[str, Dict[str, List]]:
    summary: Dict[str, Dict[str, List]] = {}
    for pid, data in aggregate.items():
        summary[f"P{pid}"] = {
            "posture": data.get("posture", []),
            "gaze": data.get("gaze", []),
            "interactions": data.get("interactions", []),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Combined Action pillar demo")
    parser.add_argument("--video", type=Path, required=True, help="Input video file")
    parser.add_argument("--output", type=Path, help="Optional output path for annotated video")
    parser.add_argument("--display", action="store_true", help="Show annotated frames in a window")
    parser.add_argument("--models-root", type=Path, help="Directory containing default assets (pipeline, weights, tasks)")
    parser.add_argument("--pipeline-path", help="Override path for the pose classification pipeline")
    parser.add_argument("--person-model-path", help="Override path for the person detection model")
    parser.add_argument("--object-model-path", help="Override path for the object detection model")
    parser.add_argument("--face-model-path", help="Override path for the optional face detector")
    parser.add_argument("--pose-task-path", help="Override path for the MediaPipe pose task")
    parser.add_argument("--device", default="cuda", help="Torch device to run the YOLO models on")
    parser.add_argument("--detect-interval", type=int, default=1,
                        help="Number of frames between full detections")
    parser.add_argument("--aggregate-detections", type=int, default=1,
                        help="Number of detection runs to aggregate before publishing a summary")
    parser.add_argument("--pose-downscale", type=int, default=0,
                        help="If >0, resize crops so their max side matches this value before pose detection")
    parser.add_argument("--confidence-threshold", type=float, default=CONF_THRESH,
                        help="YOLO person confidence threshold")
    parser.add_argument("--iou-threshold", type=float, default=IOU_THRESH,
                        help="YOLO person IoU threshold")
    parser.add_argument("--max-persons", type=int, default=3,
                        help="Maximum number of tracked persons per frame")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ,
                        help="Input image size for YOLO person detection")
    parser.add_argument("--enable-gaze", action="store_true", default=True,
                        help="Enable gaze estimation (requires face detector and gazelle)")
    parser.add_argument("--disable-gaze", dest="enable_gaze", action="store_false")
    parser.add_argument("--enable-object-interactions", action="store_true", default=True,
                        help="Enable hand/object interaction reasoning")
    parser.add_argument("--disable-object-interactions", dest="enable_object_interactions", action="store_false")
    parser.add_argument("--collect-timings", action="store_true", help="Print per-stage timings")
    parser.add_argument("--yolo-export-format", choices=["onnx", "tensorrt", "trt", "engine"],
                        help="Optionally export YOLO weights to the requested format before running inference")
    parser.add_argument("--yolo-export-device", help="Device to use while exporting YOLO weights (defaults to --device)")
    parser.add_argument("--yolo-force-export", action="store_true",
                        help="Force re-export even if converted weights already exist")

    args = parser.parse_args()

    if not args.video.exists():
        raise FileNotFoundError(f"Video file not found: {args.video}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)

    config_kwargs = {
        "visualize": True,
        "debug_timings": args.collect_timings,
        "detect_interval": max(1, args.detect_interval),
        "aggregate_detections": max(1, args.aggregate_detections),
        "pose_downscale": args.pose_downscale if args.pose_downscale and args.pose_downscale > 0 else None,
        "device": args.device,
        "confidence_threshold": args.confidence_threshold,
        "iou_threshold": args.iou_threshold,
        "max_persons": max(1, args.max_persons),
        "imgsz": max(1, args.imgsz),
        "enable_gaze": args.enable_gaze,
        "enable_object_interactions": args.enable_object_interactions,
    }

    config_kwargs["pipeline_path"] = resolve_asset(args, args.pipeline_path, MODEL_PIPELINE_PATH)
    config_kwargs["person_model_path"] = resolve_asset(args, args.person_model_path, YOLO_WEIGHTS)
    config_kwargs["object_model_path"] = resolve_asset(args, args.object_model_path, OBJECT_YOLO_WEIGHTS)
    config_kwargs["pose_task_path"] = resolve_asset(args, args.pose_task_path, POSE_TASK_PATH)
    face_path = resolve_asset(args, args.face_model_path, FACE_YOLO_WEIGHTS)
    if face_path:
        config_kwargs["face_model_path"] = face_path
    else:
        config_kwargs["face_model_path"] = None

    if args.yolo_export_format:
        config_kwargs["yolo_export_format"] = args.yolo_export_format
    if args.yolo_export_device:
        config_kwargs["yolo_export_device"] = args.yolo_export_device
    if args.yolo_force_export:
        config_kwargs["yolo_force_export"] = True

    pipeline = CombinedActionPipeline(**config_kwargs)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(args.output), fourcc, fps, (width, height))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_result, aggregate = pipeline.process_frame(frame)

            persons = frame_result.get("persons", [])
            if persons:
                summaries = [format_person_summary(person) for person in persons]
                print(f"Frame {frame_result['frame_index']}:\n  " + "\n  ".join(summaries))

            if args.collect_timings:
                timings = frame_result.get("timings")
                if timings:
                    pretty = json.dumps({k: round(v, 4) for k, v in timings.items()}, indent=2)
                    print(f"Timings (s):\n{pretty}")

            if aggregate is not None:
                json_summary = json.dumps(
                    {
                        "detection_count": aggregate["detection_count"],
                        "data": summarise_aggregate(aggregate.get("data", {})),
                    },
                    indent=2,
                )
                print(f"Aggregate after {aggregate['detection_count']} detections:\n{json_summary}")

            visualization = frame_result.get("visualization")
            output_frame = visualization if visualization is not None else frame

            if writer is not None:
                writer.write(output_frame)

            if args.display:
                cv2.imshow("Combined Action", output_frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
