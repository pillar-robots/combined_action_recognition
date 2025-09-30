"""Combined Action pipeline closely mirroring Final_action/inference2.py."""

from __future__ import annotations

import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import joblib
import mediapipe as mp
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import torch
from torch import nn
from ultralytics.nn.modules import conv as ultraconv
from ultralytics import YOLO

class ConcatHead(nn.Module):
    """Concatenaion layer for Detect heads."""

    def __init__(self, nc1=80, nc2=1, ch=()):
        """Initializes the ConcatHead."""
        super().__init__()
        self.nc1 = nc1  # number of classes of head 1
        self.nc2 = nc2  # number of classes of head 2

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""

        # x is a list of length 2
        # Each element is either a tuple or just the decoded features
        # depending whether it's being exported.
        # First element of tuple are the decoded preds,
        # second element are feature maps for heatmap visualization

        if isinstance(x[0], tuple):
            preds1 = x[0][0]   
            preds2 = x[1][0]
        elif isinstance(x[0], list): # when returned raw outputs
            # The shape is used for stride creation in tasks.py.
            # Feature maps will have to be decoded individually if used as they can't be merged.
            return [torch.cat((x0, x1), dim=1) for x0, x1 in zip(x[0], x[1])]
        else:
            preds1 = x[0]
            preds2 = x[1]

        # Concatenate the new head outputs as extra outputs

        # 1. Concatenate bbox outputs
        # Shape changes from [N, 4, 6300] to [N, 4, 12600]
        preds = torch.cat((preds1[:, :4, :], preds2[:, :4, :]), dim=2)

        # 2. Concatenate class outputs
        # Append preds 1 with empty outputs of size 6300
        shape = list(preds1.shape)
        shape[-1] = preds1.shape[-1] + preds2.shape[-1]

        preds1_extended = torch.zeros(shape, device=preds1.device,
                                      dtype=preds1.dtype)
        preds1_extended[..., : preds1.shape[-1]] = preds1

        # Prepend preds 2 with empty outputs of size 6300
        shape = list(preds2.shape)
        shape[-1] = preds1.shape[-1] + preds2.shape[-1]

        preds2_extended = torch.zeros(shape, device=preds2.device,
                                      dtype=preds2.dtype)
        preds2_extended[..., preds2.shape[-1] :] = preds2

        # Arrange the class probabilities in order preds1, preds2. The
        # class indices of preds2 will therefore start after preds1
        preds = torch.cat((preds, preds1_extended[:, 4:, :]), dim=1)
        preds = torch.cat((preds, preds2_extended[:, 4:, :]), dim=1)

        if isinstance(x[0], tuple):
            return (preds, x[0][1])
        else:
            return preds

ultraconv.ConcatHead = ConcatHead

# if not hasattr(ultraconv, "ConcatHead"):
#     class ConcatHead(nn.Module):
#         def __init__(self, head: nn.Module, dimension: int = 1) -> None:
#             super().__init__()
#             self.head = head
#             self.dimension = dimension

#         def forward(self, x):
#             if isinstance(x, (list, tuple)):
#                 x = torch.cat(x, dim=self.dimension)
#             return self.head(x)

#     ultraconv.ConcatHead = ConcatHead

from gazelle import gaze_prediction

PACKAGE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = PACKAGE_DIR / "assets"

MODEL_PIPELINE_PATH = str(ASSETS_DIR / "models" / "sitstand_pose_pipeline.joblib")
YOLO_WEIGHTS = str(ASSETS_DIR / "models" / "yolov8n.pt")
OBJECT_YOLO_WEIGHTS = str(ASSETS_DIR / "models" / "merged.pt")
FACE_YOLO_WEIGHTS = str(ASSETS_DIR / "models" / "face_yolov8n.pt")
POSE_TASK_PATH = str(ASSETS_DIR / "models" / "pose_landmarker_full.task")
YOLO_YAML_PATH = str(ASSETS_DIR / "models" / "yolov8n-2xhead.yaml")

CONF_THRESH = 0.3
IOU_THRESH = 0.45
DEFAULT_IMGSZ = 640


LOGGER = logging.getLogger(__name__)


EXPORT_SUFFIXES = {
    "onnx": ".onnx",
    "engine": ".engine",
}

EXPORT_ALIASES = {
    "onnx": "onnx",
    "tensorrt": "engine",
    "trt": "engine",
    "engine": "engine",
}


def _maybe_export_yolo(weights_path: str,
                       export_format: Optional[str],
                       force: bool = False,
                       device: Optional[str] = None) -> str:
    """Export YOLO weights to the requested format when needed."""

    if not export_format:
        return weights_path

    fmt_key = export_format.lower().strip()
    if fmt_key not in EXPORT_ALIASES:
        LOGGER.warning("Unknown YOLO export format '%s'; falling back to original weights", export_format)
        return weights_path

    resolved_format = EXPORT_ALIASES[fmt_key]
    target_suffix = EXPORT_SUFFIXES.get(resolved_format)
    weights_file = Path(weights_path)

    if not weights_file.exists():
        LOGGER.warning("YOLO weights '%s' not found; skipping export", weights_path)
        return weights_path

    target_path = weights_file.with_suffix(target_suffix) if target_suffix else weights_file

    if not force and target_path.exists():
        return str(target_path)

    export_kwargs: Dict[str, Any] = {"format": resolved_format, "verbose": False}
    if device:
        export_kwargs["device"] = device

    try:
        model = YOLO(str(weights_file))
        exported_path = model.export(**export_kwargs)
        # ultralytics returns the exported file path on success
        if exported_path:
            return str(exported_path)
    except Exception as exc:  # pragma: no cover - hard to simulate export failures
        LOGGER.warning("Failed to export YOLO weights '%s' to %s: %s", weights_path, resolved_format, exc)
        return weights_path

    if target_path.exists():
        return str(target_path)

    LOGGER.warning("YOLO export for '%s' reported success but output not found; using original weights", weights_path)
    return weights_path


def _apply_visibility_and_flatten(X_flat: np.ndarray) -> np.ndarray:
    """Legacy helper used inside the pickled scikit-learn pipeline."""
    X = X_flat.reshape(X_flat.shape[0], 33, 4)
    X_xyz = X[..., :3] * X[..., 3:4]
    return X_xyz.reshape(X_xyz.shape[0], -1)


main_module = sys.modules.get("__main__")
if main_module is not None and not hasattr(main_module, "_apply_visibility_and_flatten"):
    setattr(main_module, "_apply_visibility_and_flatten", _apply_visibility_and_flatten)


def get_box(points: np.ndarray, margin: int = 20, h: int = 480, w: int = 640) -> np.ndarray:
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    x_min = max(x_min - margin, 0)
    y_min = max(y_min - margin, 0)
    x_max = min(x_max + margin, w)
    y_max = min(y_max + margin, h)
    return np.array((x_min, y_min, x_max, y_max))


def get_hand_boxes(landmarks: np.ndarray, h: int, w: int) -> Dict[str, np.ndarray]:
    values = landmarks.copy()
    values[..., 0] = values[..., 0] * w
    values[..., 1] = values[..., 1] * h
    right_hand_indices = [16, 18, 20, 22]
    left_hand_indices = [15, 17, 19, 21]
    right_hand_box = get_box(values[right_hand_indices, :2], h=h, w=w)
    left_hand_box = get_box(values[left_hand_indices, :2], h=h, w=w)
    return {"Right Hand": right_hand_box, "Left Hand": left_hand_box}


def calculate_iou(box1: np.ndarray, box2: List[float]) -> float:
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    intersection_area = max(0.0, x2_inter - x1_inter) * max(0.0, y2_inter - y1_inter)
    hand_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    if hand_area <= 0.0:
        return 0.0
    return intersection_area / hand_area


def is_point_in_bbox(point: Tuple[float, float], obj: List[float]) -> Optional[str]:
    x, y = point
    x1, y1, x2, y2 = obj[0], obj[1], obj[2], obj[3]
    if x1 <= x <= x2 and y1 <= y <= y2:
        return obj[6]
    return None


def mediapipe_pose_on_crop(frame_bgr: np.ndarray,
                           bbox: np.ndarray,
                           pose_detector: vision.PoseLandmarker,
                           downscale_max_side: Optional[int] = None,
                           draw_vis_numbers: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 <= x1 or y2 <= y1:
        return None, None

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None

    ch, cw = crop.shape[:2]
    if downscale_max_side is not None and downscale_max_side <= 0:
        downscale_max_side = None

    if downscale_max_side is None:
        scale = 1.0
        small = crop
    else:
        scale = max(ch, cw) / float(downscale_max_side)
        if scale <= 1.0:
            small = crop
        else:
            small = cv2.resize(crop, (int(cw / scale), int(ch / scale)), interpolation=cv2.INTER_LINEAR)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB),
    )
    result = pose_detector.detect(mp_image)
    if not result.pose_landmarks:
        return None, None

    H, W = h, w
    sch, scw = small.shape[:2]
    pose_landmarks = result.pose_landmarks[0]
    pose_proto = landmark_pb2.NormalizedLandmarkList()
    vis_proto = landmark_pb2.NormalizedLandmarkList()
    for lm in pose_landmarks:
        pose_proto.landmark.append(
            landmark_pb2.NormalizedLandmark(
                x=((lm.x * scw * scale) + x1) / float(W),
                y=((lm.y * sch * scale) + y1) / float(H),
                z=lm.z,
            )
        )
        vis_proto.landmark.append(
            landmark_pb2.NormalizedLandmark(
                visibility=getattr(lm, "visibility", 0.0),
                presence=getattr(lm, "presence", 0.0),
            )
        )

    annotated = None
    if draw_vis_numbers:
        annotated = frame_bgr.copy()
        solutions.drawing_utils.draw_landmarks(
            annotated,
            pose_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style(),
        )

    lmks = np.array([[lm.x, lm.y, lm.z] for lm in pose_proto.landmark], dtype=np.float32)
    vis = np.array([[getattr(v, "visibility", 0.0)] for v in vis_proto.landmark], dtype=np.float32)
    arr_33x4 = np.concatenate([lmks, vis], axis=1)
    return annotated, arr_33x4


def draw_box_with_label(img: Optional[np.ndarray],
                        box: np.ndarray,
                        label: str,
                        prob: Optional[float] = None,
                        color: Tuple[int, int, int] = (0, 255, 0)) -> None:
    if img is None or box is None:
        return
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    text = label + (f" {prob:.2f}" if prob is not None else "")
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y_text = max(0, y1 - th - 8)
    cv2.rectangle(img, (x1, y_text), (x1 + tw + 6, y_text + th + 6), color, -1)
    cv2.putText(
        img,
        text,
        (x1 + 3, y_text + th),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )


class FaceYOLO:
    def __init__(self, weights_path: str, device: str = "cpu") -> None:
        self.model = YOLO(weights_path)
        self.device = device

    def detect(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        results = self.model.predict(frame_bgr, device=self.device, verbose=False)[0]
        bboxes = []
        for idx, box in enumerate(results.boxes.xyxy):
            x1, y1, x2, y2 = map(float, box.tolist())
            bboxes.append({"Person": idx, "face": [x1, y1, x2, y2]})
        return bboxes


def detect_objects(yolo_model: YOLO,
                   frame: np.ndarray,
                   device: Optional[str]) -> List[List[float]]:
    predict_kwargs: Dict[str, Any] = dict(classes=[0, 63, 64, 66, 67, 73, 80], verbose=False)
    if device:
        predict_kwargs["device"] = device
    predictions = yolo_model.predict(frame, **predict_kwargs)[0]
    data: List[List[float]] = []
    for box in predictions.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        confidence = box.conf.item()
        class_id = int(box.cls.item())
        class_name = predictions.names[class_id]
        data.append([x1, y1, x2, y2, confidence, class_id, class_name])
    return data



def _apply_visibility_and_flatten(X_flat):
    """
    X_flat: (N, 33*4). Reshape to (N,33,4),
    multiply last channel (visibility) into xyz, keep xyz only (33,3),
    flatten to (N, 99).
    """
    X = X_flat.reshape(X_flat.shape[0], 33, 4)
    X_xyz = X[..., :3] * X[..., 3:4]   # multiply xyz by visibility
    return X_xyz.reshape(X_xyz.shape[0], -1)

    
class CombinedActionPipeline:
    def __init__(
        self,
        pipeline_path: str = MODEL_PIPELINE_PATH,
        person_model_path: str = YOLO_WEIGHTS,
        object_model_path: str = OBJECT_YOLO_WEIGHTS,
        face_model_path: str = FACE_YOLO_WEIGHTS,
        pose_task_path: str = POSE_TASK_PATH,
        device: str = "cuda",
        imgsz: int = DEFAULT_IMGSZ,
        max_persons: int = 3,
        detect_interval: int = 1,
        aggregate_detections: int = 1,
        pose_downscale: Optional[int] = None,
        visualize: bool = False,
        confidence_threshold: float = CONF_THRESH,
        iou_threshold: float = IOU_THRESH,
        enable_gaze: bool = True,
        enable_object_interactions: bool = True,
        debug_timings: bool = False,
        yolo_export_format: Optional[str] = None,
        yolo_force_export: bool = False,
        yolo_export_device: Optional[str] = None,
    ) -> None:
        self.pipeline = joblib.load(pipeline_path)
        export_device = yolo_export_device or device
        person_path = _maybe_export_yolo(person_model_path, yolo_export_format, yolo_force_export, export_device)
        object_path = _maybe_export_yolo(object_model_path, yolo_export_format, yolo_force_export, export_device)
        face_path = _maybe_export_yolo(face_model_path, yolo_export_format, yolo_force_export, export_device) if face_model_path else None

        self.person_detector = YOLO(person_path)
        self.object_detector = YOLO(object_path)
        # yaml_path = YOLO_YAML_PATH
        # self.object_detector = (YOLO(yaml_path, task="detect")).load(object_model_path)
        
        # YOLO(yaml_path, task="detect").load('/home/niki/pillar/yolov8n.pt')

        base_options = python.BaseOptions(model_asset_path=pose_task_path)
        pose_options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            num_poses=1,
            min_pose_detection_confidence=0.30,
            min_pose_presence_confidence=0.30,
            output_segmentation_masks=False,
        )
        self.pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

        self.face_detector = FaceYOLO(face_path, device=device) if face_path else None
        self.gaze_detector = gaze_prediction.GazeEstimator() if enable_gaze and self.face_detector is not None else None

        self.device = device
        self.imgsz = imgsz
        self.max_persons = max(1, max_persons)
        self.detect_interval = max(1, detect_interval)
        self.aggregate_every = max(1, aggregate_detections) if aggregate_detections else 0
        self.pose_downscale = pose_downscale
        self.visualize = visualize
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.enable_gaze = enable_gaze and self.gaze_detector is not None
        self.enable_object_interactions = enable_object_interactions
        self.debug_timings = debug_timings

        self.frame_idx = 0
        self.detection_events = 0
        self.aggregation_buffer: Dict[int, Dict[str, List[Any]]] = defaultdict(
            lambda: {"gaze": [], "posture": [], "interactions": []}
        )

    def reset(self) -> None:
        self.frame_idx = 0
        self.detection_events = 0
        self.aggregation_buffer.clear()

    def process_frame(self, frame: np.ndarray) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        frame_index = self.frame_idx
        self.frame_idx += 1

        if frame_index % self.detect_interval != 0:
            vis_frame = frame.copy() if self.visualize else None
            return {
                "frame_index": frame_index,
                "persons": [],
                "visualization": vis_frame,
                "timings": None,
            }, None

        vis_frame = frame.copy() if self.visualize else None
        frame_times: Optional[Dict[str, float]] = {} if self.debug_timings else None

        boxes: List[np.ndarray] = []
        confs: List[float] = []

        person_kwargs: Dict[str, Any] = dict(
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            verbose=False,
        )
        if self.device:
            person_kwargs["device"] = self.device

        t0 = time.perf_counter() if frame_times is not None else None
        person_results = self.person_detector.predict(frame, **person_kwargs)[0]
        if frame_times is not None and t0 is not None:
            frame_times["person_det"] = time.perf_counter() - t0

        if person_results.boxes is not None and len(person_results.boxes) > 0:
            det_boxes = person_results.boxes.xyxy.cpu().numpy()
            det_cls = person_results.boxes.cls.cpu().numpy()
            det_confs = person_results.boxes.conf.cpu().numpy()
            mask = det_cls == 0
            det_boxes, det_confs = det_boxes[mask], det_confs[mask]
            boxes = list(det_boxes[: self.max_persons])
            confs = list(det_confs[: self.max_persons])

        detected_objects: List[List[float]] = []
        if self.enable_object_interactions:
            t0 = time.perf_counter() if frame_times is not None else None
            object_device = self.device if self.device and self.device != "" else None
            detected_objects = detect_objects(self.object_detector, frame, object_device)
            if frame_times is not None and t0 is not None:
                frame_times["object_det"] = time.perf_counter() - t0

        detected_faces: List[Dict[str, Any]] = []
        if self.gaze_detector is not None and self.face_detector is not None:
            t0 = time.perf_counter() if frame_times is not None else None
            detected_faces = self.face_detector.detect(frame)
            if frame_times is not None and t0 is not None:
                frame_times["face_det"] = time.perf_counter() - t0

        gaze_targets = None
        if self.gaze_detector is not None and detected_faces:
            face_boxes = [fb["face"] for fb in detected_faces]
            t0 = time.perf_counter() if frame_times is not None else None
            gaze_targets = self.gaze_detector.gaze_estimation(frame, face_boxes)
            if frame_times is not None and t0 is not None:
                frame_times["gaze"] = time.perf_counter() - t0

        pose_time = 0.0
        pose_infer_time = 0.0
        persons: List[Dict[str, Any]] = []

        for pid, (bbox, conf) in enumerate(zip(boxes, confs)):
            pose_frame = vis_frame if self.visualize else frame
            t_pose = time.perf_counter() if frame_times is not None else None
            annotated, arr_33x4 = mediapipe_pose_on_crop(
                pose_frame,
                bbox,
                self.pose_detector,
                downscale_max_side=self.pose_downscale,
                draw_vis_numbers=self.visualize,
            )
            if frame_times is not None and t_pose is not None:
                pose_time += time.perf_counter() - t_pose

            if self.visualize and annotated is not None:
                vis_frame = annotated

            pred_label = "no-pose"
            pred_conf = None
            if arr_33x4 is not None:
                t_infer = time.perf_counter() if frame_times is not None else None
                x_in = arr_33x4.reshape(1, -1)
                probs = self.pipeline.predict_proba(x_in)[0]
                idx = int(np.argmax(probs))
                pred_label = str(self.pipeline.classes_[idx])
                pred_conf = float(np.max(probs))
                if frame_times is not None and t_infer is not None:
                    pose_infer_time += time.perf_counter() - t_infer

            hand_boxes: Optional[Dict[str, np.ndarray]] = None
            if arr_33x4 is not None:
                hand_boxes = get_hand_boxes(arr_33x4, h=frame.shape[0], w=frame.shape[1])
                if self.visualize and vis_frame is not None:
                    for hand_name, hand_box in hand_boxes.items():
                        draw_box_with_label(vis_frame, hand_box, f"P{pid}:{hand_name}", color=(255, 0, 0))

            gaze_point = None
            if gaze_targets and pid < len(gaze_targets):
                entry = gaze_targets[pid]
                if isinstance(entry, dict) and "Gaze Target" in entry:
                    gaze_point = entry["Gaze Target"]

            look_label = None
            interactions: List[str] = []
            if detected_objects:
                if gaze_point is not None:
                    for obj in detected_objects:
                        label = is_point_in_bbox(gaze_point, obj)
                        if label is not None:
                            look_label = label
                            break
                if hand_boxes is not None:
                    for obj in detected_objects:
                        if obj[6] == "person":
                            continue
                        for hand_name, hand_box in hand_boxes.items():
                            overlap = calculate_iou(hand_box, obj)
                            if obj[6] == "laptop" and overlap > 0.9:
                                interactions.append(f"{hand_name} on laptop")
                            elif obj[6] == "robobo" and overlap > 0.3:
                                interactions.append(f"{hand_name} close to robobo")

            if self.visualize and vis_frame is not None and detected_objects:
                for obj in detected_objects:
                    if obj[6] != "person":
                        draw_box_with_label(vis_frame, np.array(obj[:4]), obj[6], color=(255, 255, 0))

            if self.visualize and vis_frame is not None:
                color = (0, 255, 0) if pred_label != "no-pose" else (0, 0, 255)
                draw_box_with_label(vis_frame, bbox, f"P{pid}:{pred_label}", pred_conf, color=color)
                if gaze_point is not None:
                    cv2.circle(vis_frame, (int(gaze_point[0]), int(gaze_point[1])), 5, (0, 255, 255), -1)
                    cv2.putText(
                        vis_frame,
                        f"P{pid} gaze",
                        (int(gaze_point[0]) + 10, int(gaze_point[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                cv2.putText(
                    vis_frame,
                    f"P{pid} looking at {look_label}",
                    (10, 30 + pid * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                for i, inter in enumerate(interactions):
                    cv2.putText(
                        vis_frame,
                        f"P{pid} {inter}",
                        (10, 30 + (pid + i + 1) * 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

            persons.append(
                {
                    "person_id": pid,
                    "bbox": [float(v) for v in bbox],
                    "pose_label": pred_label,
                    "pose_confidence": pred_conf,
                    "gaze_target": look_label,
                    "interactions": interactions,
                    "gaze_point": gaze_point,
                }
            )

            if self.aggregate_every:
                bucket = self.aggregation_buffer[pid]
                bucket["gaze"].append(look_label if look_label is not None else "none")
                bucket["posture"].append(pred_label)
                bucket["interactions"].append(list(interactions) if interactions else [])

        if frame_times is not None:
            if pose_time > 0.0:
                frame_times["pose_landmark"] = pose_time
            if pose_infer_time > 0.0:
                frame_times["pose_classify"] = pose_infer_time

        self.detection_events += 1
        aggregate: Optional[Dict[str, Any]] = None
        if self.aggregate_every and self.detection_events % self.aggregate_every == 0:
            snapshot = {
                pid: {key: list(values) for key, values in data.items()}
                for pid, data in self.aggregation_buffer.items()
            }
            aggregate = {"detection_count": self.aggregate_every, "data": snapshot}
            self.aggregation_buffer.clear()

        if frame_times is not None and frame_times:
            total = sum(frame_times.values())
            frame_times["total"] = total

        return (
            {
                "frame_index": frame_index,
                "persons": persons,
                "visualization": vis_frame if self.visualize else None,
                "timings": frame_times,
            },
            aggregate,
        )
