# HighJumpAnalyzer App

**HighJumpAnalyzer** is a Python-based application for high jump analysis using video and IMU data.

## Features
- Upload and analyze videos of a high jump.
- Compute metrics such as:
  - Takeoff Angle
  - Horizontal Velocity
  - Conversion Efficiency

## Installation
1. Download the latest `.exe` from the [Releases](https://github.com/sharre01/HighJumpAnalyzer/releases).
2. Run the `.exe` to start the application.

## Usage
1. Launch the application and upload your videos and IMU data through the user-friendly interface.
2. Navigate to the desired frame (e.g., the moment of takeoff) using the video playback controls.
3. The app uses the **TFLite MoveNet Pose Estimation algorithm** to analyze body positioning and calculate the metrics automatically.
