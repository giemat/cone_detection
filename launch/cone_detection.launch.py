import os
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition

def generate_launch_description():
    params_file_path = os.path.join(
        get_package_share_directory("cone_detection"),
        "config",
        "params.yaml"
    )

    # The current event (mission) 
    # Possible values (lowercase):
    # - "acceleration" : acceleration test
    # - "skidpad"      : figure-eight handling test
    # - "autocross"    : obstacle avoidance (slalom)
    # - "trackdrive"   : multiple laps on a track
    # - "endurance"    : long-distance reliability and efficiency test
    current_mission = LaunchConfiguration("mission")
    current_mission_arg = DeclareLaunchArgument(
        "mission",
        default_value="trackdrive",
        description="Select current mission"
    )

    # Benchmark overrides. Empty values keep the params.yaml settings.
    benchmark_args = {
        "model": "Model name in the models/ folder, e.g. Yolov8n1920 (empty = params.yaml paths)",
        "backend": "Inference backend: tensorrt | onnx (empty = params.yaml)",
        "precision": "Engine precision: fp32 | fp16 | int8 (empty = params.yaml)",
        "benchmark_csv": "Per-frame timing CSV path (empty = params.yaml)",
        "detections_csv": "Per-frame detections CSV path (empty = params.yaml)",
        "benchmark_label": "Free text label written to the CSV header",
    }
    benchmark_arg_actions = [
        DeclareLaunchArgument(name, default_value="", description=desc)
        for name, desc in benchmark_args.items()
    ]
    use_sim_time = LaunchConfiguration("use_sim_time")
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="False",
        description="Use /clock (simulator or `ros2 bag play --clock`)"
    )

    def model_overrides(model):
        """All model file paths for <model>, in the folder of params.yaml model_path."""
        with open(params_file_path) as f:
            params = yaml.safe_load(f)["/cone_detection_node"]["ros__parameters"]
        d = os.path.dirname(params["model_path"])
        path = lambda name: os.path.join(d, name)
        return {
            "model_path": path(f"{model}.onnx"),
            "fp32_engine_path": path(f"{model}_FP32.engine"),
            "engine_path": path(f"{model}.engine"),
            "quantized_engine_path": path(f"{model}_INT8.engine"),
            "onnx.fp16_model_path": path(f"{model}_fp16.onnx"),
            "onnx.int8_model_path": path(f"{model}_int8_qdq.onnx"),
        }

    def make_node(context):
        overrides = {"mission": current_mission, "use_sim_time": use_sim_time}
        mapping = {
            "backend": "backend",
            "precision": "precision",
            "benchmark_csv": "benchmark.csv_path",
            "detections_csv": "benchmark.detections_csv_path",
            "benchmark_label": "benchmark.label",
        }
        for arg, param in mapping.items():
            value = LaunchConfiguration(arg).perform(context)
            if value:
                overrides[param] = value
        model = LaunchConfiguration("model").perform(context)
        if model:
            overrides.update(model_overrides(model))
        return [Node(
            executable="cone_detection_node",
            package="cone_detection",
            name="cone_detection_node",
            parameters=[params_file_path, overrides],
            output="screen"
        )]

    cone_detection_node = OpaqueFunction(function=make_node)

    launch_rviz = LaunchConfiguration("rviz")
    launch_rviz_arg = DeclareLaunchArgument(
        "rviz",
        default_value="False",
        description="Launch RViz2"
    )
   
    rviz_config_path = os.path.join(
        get_package_share_directory("cone_detection"),
        "config",
        "config.rviz"
    )

    rviz_node = Node(
        executable="rviz2",
        package="rviz2",
        name="RViz2",
        arguments=["-d", rviz_config_path],
        output="screen",
        condition=IfCondition(launch_rviz)
    )

    launch_description = LaunchDescription()
    launch_description.add_action(current_mission_arg)
    launch_description.add_action(use_sim_time_arg)
    for action in benchmark_arg_actions:
        launch_description.add_action(action)
    launch_description.add_action(cone_detection_node)
    launch_description.add_action(launch_rviz_arg)
    launch_description.add_action(rviz_node)

    return launch_description
