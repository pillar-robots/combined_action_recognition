#!/usr/bin/env python3

"""ROS 2 node wrapping the Combined Action pipeline."""
from __future__ import annotations

import json
from typing import Any, Dict

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError

from perception_interfaces.msg import Box2D, Box2DArray

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

class CombinedActionNode(Node):
    def __init__(self) -> None:
        super().__init__('combined_action_node')

        # Declare parameters mirroring the configuration dataclass.
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('detect_interval', 1)
        self.declare_parameter('aggregate_detections', 3)
        self.declare_parameter('pose_downscale', 0)
        self.declare_parameter('publish_visualization', False)
        self.declare_parameter('pose_task_path', '')
        self.declare_parameter('pipeline_path', '')
        self.declare_parameter('person_model_path', '')
        self.declare_parameter('object_model_path', '')
        self.declare_parameter('face_model_path', '')
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('confidence_threshold', CONF_THRESH)
        self.declare_parameter('iou_threshold', IOU_THRESH)
        self.declare_parameter('max_persons', 3)
        self.declare_parameter('imgsz', DEFAULT_IMGSZ)
        self.declare_parameter('enable_gaze', True)
        self.declare_parameter('enable_object_interactions', True)
        self.declare_parameter('yolo_export_format', '')
        self.declare_parameter('yolo_force_export', False)
        self.declare_parameter('yolo_export_device', '')

        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        publish_visualization = self.get_parameter('publish_visualization').get_parameter_value().bool_value

        config_kwargs: Dict[str, Any] = {
            'detect_interval': max(1, int(self.get_parameter('detect_interval').get_parameter_value().integer_value)),
            'aggregate_detections': max(1, int(self.get_parameter('aggregate_detections').get_parameter_value().integer_value)),
            'pose_downscale': self._parse_optional_int('pose_downscale'),
            'visualize': publish_visualization,
            'device': self.get_parameter('device').get_parameter_value().string_value,
            'confidence_threshold': float(self.get_parameter('confidence_threshold').value),
            'iou_threshold': float(self.get_parameter('iou_threshold').value),
            'max_persons': max(1, int(self.get_parameter('max_persons').get_parameter_value().integer_value)),
            'imgsz': max(1, int(self.get_parameter('imgsz').get_parameter_value().integer_value)),
            'enable_gaze': bool(self.get_parameter('enable_gaze').get_parameter_value().bool_value),
            'enable_object_interactions': bool(self.get_parameter('enable_object_interactions').get_parameter_value().bool_value),
        }

        pipeline_path = self.get_parameter('pipeline_path').get_parameter_value().string_value or MODEL_PIPELINE_PATH
        person_model_path = self.get_parameter('person_model_path').get_parameter_value().string_value or YOLO_WEIGHTS
        object_model_path = self.get_parameter('object_model_path').get_parameter_value().string_value or OBJECT_YOLO_WEIGHTS
        face_model_param = self.get_parameter('face_model_path').get_parameter_value().string_value or FACE_YOLO_WEIGHTS

        config_kwargs['pipeline_path'] = pipeline_path
        config_kwargs['person_model_path'] = person_model_path
        config_kwargs['object_model_path'] = object_model_path
        config_kwargs['face_model_path'] = face_model_param if face_model_param else None
        pose_task_value = self.get_parameter('pose_task_path').get_parameter_value().string_value or POSE_TASK_PATH
        config_kwargs['pose_task_path'] = pose_task_value

        export_format = self.get_parameter('yolo_export_format').get_parameter_value().string_value.strip()
        export_device = self.get_parameter('yolo_export_device').get_parameter_value().string_value.strip()
        force_export = self.get_parameter('yolo_force_export').get_parameter_value().bool_value
        if export_format:
            config_kwargs['yolo_export_format'] = export_format
        if export_device:
            config_kwargs['yolo_export_device'] = export_device
        if force_export:
            config_kwargs['yolo_force_export'] = True

        try:
            self.pipeline = CombinedActionPipeline(**config_kwargs)
        except Exception as exc:  # pragma: no cover - initialization failure should be visible
            self.get_logger().error(f'Failed to initialise CombinedActionPipeline: {exc}')
            raise

        self.bridge = CvBridge()

        qos = 10
        self.image_sub = self.create_subscription(Image, self.image_topic, self.image_callback, qos)
        self.detections_pub = self.create_publisher(Box2DArray, 'combined_action/detections', qos)
        self.aggregate_pub = self.create_publisher(String, 'combined_action/aggregate', qos)
        self.image_pub = None
        if publish_visualization:
            self.image_pub = self.create_publisher(Image, 'combined_action/image', qos)

        self.get_logger().info(
            f"CombinedActionNode initialised with image_topic={self.image_topic}, "
            f"detect_interval={config_kwargs['detect_interval']}, "
            f"aggregate_detections={config_kwargs['aggregate_detections']}"
        )

    def _parse_optional_int(self, name: str) -> Any:
        param = self.get_parameter(name)
        if param is None:
            return None

        if param.type_ == Parameter.Type.NOT_SET:
            return None

        if param.type_ == Parameter.Type.INTEGER:
            val = int(param.value)
            return val if val > 0 else None

        if param.type_ == Parameter.Type.DOUBLE:
            val = int(param.value)
            return val if val > 0 else None

        return None

    def image_callback(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as exc:
            self.get_logger().error(f'CvBridge failed: {exc}')
            return

        frame_result, aggregate = self.pipeline.process_frame(frame)
        print('AAASDFSFDFDS')

        persons = frame_result.get('persons', [])
        if persons:
            detections_msg = Box2DArray()
            detections_msg.header = msg.header
            for person in persons:
                box_msg = Box2D()
                box_msg.header = msg.header
                box_msg.class_name = self._format_class_name(person)
                bbox = person.get('bbox', [0.0, 0.0, 0.0, 0.0])
                box_msg.x_min = float(bbox[0])
                box_msg.y_min = float(bbox[1])
                box_msg.x_max = float(bbox[2])
                box_msg.y_max = float(bbox[3])
                conf = person.get('pose_confidence') or 0.0
                box_msg.confidence = float(conf)
                detections_msg.boxes.append(box_msg)
            self.detections_pub.publish(detections_msg)

        if aggregate is not None:
            summary = String()
            summary.data = json.dumps(aggregate)
            self.aggregate_pub.publish(summary)

        visualization = frame_result.get('visualization')
        if self.image_pub and visualization is not None:
            try:
                image_msg = self.bridge.cv2_to_imgmsg(visualization, encoding='bgr8')
                image_msg.header = msg.header
                self.image_pub.publish(image_msg)
            except CvBridgeError as exc:
                self.get_logger().error(f'Failed to publish visualisation: {exc}')

    def _format_class_name(self, person: Dict[str, Any]) -> str:
        gaze_label = person.get('gaze_target') or 'none'
        interactions = person.get('interactions') or []
        interactions_text = ';'.join(interactions) if interactions else 'none'
        return f"person|pose={person.get('pose_label')}|gaze={gaze_label}|interactions={interactions_text}"


def main(args: Any = None) -> None:
    rclpy.init(args=args)
    node = CombinedActionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:  # pragma: no cover - manual shutdown
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':  # pragma: no cover
    main()
