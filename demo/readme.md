# Combined Action Demo

Standalone script for exercising the Combined Action perception pipeline without ROS 2.

## Usage

1. Install the core library (from the repository root):
   ```bash
   cd ../combined_action_pillar_core
   pip install .
   ```
2. Run the demo (replace the paths below with your own video / model locations):
   ```bash
   cd ../demo
   python demo.py --video samples/input.mp4 --output samples/output.mp4
   ```

All required checkpoints ship with the package, so you only need `--models-root`
if you want to point at alternative files. You can override any asset
individually with the `--pipeline-path`, `--person-model-path`,
`--object-model-path`, `--face-model-path`, or `--pose-task-path` options. Set
`--display` to preview the annotated stream.

## Samples

Drop sample videos in the `samples/` folder. Placeholders are provided without any
media to keep the repository lightweight.
