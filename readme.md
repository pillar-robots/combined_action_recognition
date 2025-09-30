# Combined Action Perception for PILLAR-Robots

<a href="https://pillar-robots.eu">
  <img src="assets/pillar-logo.png" height="50px" alt="PILLAR-Robots Logo">
</a>

This repository packages the **combined action perception** stack used in the [PILLAR-Robots](https://pillar-robots.eu) project. It delivers:

- A **pip-installable core library** in `combined_action_pillar_core/`
- **ROS 2** integration in `combined_action_pillar_ros2/`
- A **standalone demo** in `demo/` for running on recorded footage without ROS

The pipeline fuses person detection, pose classification, gaze estimation (Gazelle), and simple hand/object interaction cues into a single reusable module.

---

## Contents

1. [Core Package (Pip Installable)](#core-package-pip-installable)  
2. [Demo Application](#demo-application)  
3. [ROS 2 Integration](#ros-2-integration)  
4. [Model Assets](#model-assets)  

---

## Core Package (Pip Installable)

`combined_action_pillar_core/` contains the Python implementation of the perception pipeline.

### Installation

```bash
cd combined_action_pillar_core/
pip install .
```

### Quick Start

```python
from combined_action_pillar_core import CombinedActionPipeline

pipeline = CombinedActionPipeline()
frame_result, aggregate = pipeline.process_frame(frame_bgr)
```

Pass keyword arguments to override defaults, e.g. `person_model_path`, `enable_gaze=False`, or `yolo_export_format="onnx"` to force an ONNX/TensorRT export.

---

## Demo Application

The `demo/` folder offers a CLI for running the combined pipeline on videos without ROS. It reads frames, performs the full perception stack, prints detections, and optionally writes annotated output.

### Usage

```bash
cd combined_action_pillar/demo
python demo.py --video samples/input.mp4 --output samples/output.mp4 --display --collect-timings
```

Key flags:

- `--video` (required): path to the input file
- `--output`: annotated video file (optional)
- `--display`: open an interactive preview window
- `--collect-timings`: print per-stage latency metrics each frame
- `--yolo-export-format {onnx,tensorrt,trt,engine}`: export YOLO weights before inference
- `--yolo-force-export`: re-export even if converted weights exist

Use `python demo.py --help` for the complete option list.

---

## ROS 2 Integration

`combined_action_pillar_ros2/` wraps the pipeline as a ROS 2 package ready for composition with the other perception pillars.

### What the Node Does

- Subscribes to an image topic (default `/camera/image_raw`)
- Runs person detection, pose classification, gaze estimation, and basic hand/object interaction checks
- Publishes bounding boxes on `combined_action/detections` (`perception_interfaces/Box2DArray`)
- Publishes aggregated JSON summaries on `combined_action/aggregate` (`std_msgs/String`)
- Publishes an annotated image on `combined_action/image` when `publish_visualization=true`

### Prerequisites

- ROS 2 Humble (or compatible)
- [`perception_interfaces`](https://github.com/pillar-robots/perception_interfaces)
- Other pillar packages providing upstream feeds (e.g. face tracker, gaze, emotion) as needed
- Optional: [`youtube_publisher_ros2`](https://github.com/pillar-robots/YouTube-publisher-ROS2) for video sources, `web_video_server` for browser streaming

### Building

Copy the package into your workspace and build the relevant dependencies:

```bash
cd ~/ros2_ws
colcon build --packages-select perception_interfaces combined_action_pillar_core combined_action_pillar_ros2
source install/setup.bash
```

### Launching

```bash
ros2 launch combined_action_pillar_ros2 combined_action.launch.py \
  image_topic:=/camera/image_raw \
  youtube_link:=/ros2_ws/src/combined_action_pillar/demo/samples/color_teacher.mp4 \
  publish_visualization:=true \
  yolo_export_format:=onnx
```

Useful parameters:

- `image_topic`: input image stream
- `youtube_link`: passed through to `youtube_publisher_ros2`; leave empty to disable
- `publish_visualization`: control publication of the annotated image
- `yolo_export_format`, `yolo_force_export`, `yolo_export_device`: drive ONNX/TensorRT exports

With `publish_visualization=true`, `web_video_server` (spawned by the launch file) hands the annotated stream to a browser via `http://localhost:8080/stream?topic=/combined_action/image`. Use SSH port forwarding (`ssh -N -L 8080:localhost:8080 user@host`) to view it remotely.

---

## Model Assets

Default checkpoints live under `combined_action_pillar_core/assets/models/` and are packaged with the module:

- `sitstand_pose_pipeline.joblib`
- `yolov8n.pt`
- `merged.pt` (object detector)
- `face_yolov8n.pt`
- `pose_landmarker_full.task`

Override any of these using the CLI flags or ROS parameters (`person_model_path`, `object_model_path`, `face_model_path`, `pose_task_path`, `pipeline_path`). Gaze estimation depends on Gazelle weights, already bundled; disable with `enable_gaze:=false` if not required.

---

## Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for detection
- [Gazelle](https://github.com/fkryan/gazelle) for gaze estimation backbones
- [MediaPipe](https://developers.google.com/mediapipe) pose landmarker
- The broader [PILLAR-Robots](https://pillar-robots.eu) WP2 perception stack

