# Combined Action ROS2 Wrapper

This package exposes the Combined Action perception pipeline as a ROS 2 node. It closely mirrors the
layout of other WP2 modules (e.g. `gaze_estimation_pillar_ros2`). The node subscribes to a raw image
stream, runs the combined action pipeline at a configurable cadence, and publishes detections and
aggregated summaries.

Launch files are provided in the `launch/` folder for quick integration in ROS 2 systems.
