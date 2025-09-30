#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    # Declare configurable arguments
    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/camera/image_raw',
        description='Topic for the input camera stream.'
    )
    youtube_link_arg = DeclareLaunchArgument(
        'youtube_link',
        default_value='',
        description='YouTube link to push the stream to. Leave empty to disable.'
    )
    publish_visualization_arg = DeclareLaunchArgument(
        'publish_visualization',
        default_value='true',
        description='Publish the annotated image stream from the combined action pipeline.'
    )

    # Convert to LaunchConfiguration for reuse
    image_topic = LaunchConfiguration('image_topic')
    youtube_link = LaunchConfiguration('youtube_link')
    publish_visualization = LaunchConfiguration('publish_visualization')

    combined_action_node = Node(
        package='combined_action_pillar_ros2',
        executable='combined_action_node',
        name='combined_action_node',
        output='screen',
        parameters=[{
            'image_topic': image_topic,
            'detect_interval': 1,
            'aggregate_detections': 3,
            'pose_downscale': 0,
            'publish_visualization': publish_visualization,
            'pipeline_path': '',
            'person_model_path': '',
            'object_model_path': '',
            'face_model_path': '',
            'device': 'cuda',
            'confidence_threshold': 0.3,
            'iou_threshold': 0.45,
            'max_persons': 3,
            'imgsz': 640,
            'enable_gaze': True,
            'enable_object_interactions': True
        }]
    )

    youtube_publisher_node = Node(
        package='youtube_publisher_ros2',
        executable='youtube_publisher_node',
        name='youtube_publisher_node',
        output='screen',
        parameters=[{
            'youtube_link': youtube_link,
            'image_topic': image_topic
        }],
        condition=IfCondition(
            PythonExpression(["'", youtube_link, "' != ''"])
        )
    )

    web_video_server_node = Node(
        package='web_video_server',
        executable='web_video_server',
        name='web_video_server',
        output='screen'
    )

    return LaunchDescription([
        image_topic_arg,
        youtube_link_arg,
        publish_visualization_arg,
        combined_action_node,
        youtube_publisher_node,
        web_video_server_node
    ])
