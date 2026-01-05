# Robotic Arm Teleoperation via Hand Gestures

A real-time hand gesture-controlled robotic arm simulation system that enables intuitive teleoperation of a 7-DOF Franka Panda manipulator using computer vision and inverse kinematics.

![Status](https://img.shields.io/badge/status-active-success)
![Python](https://img.shields.io/badge/python-3.7+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 🎯 Overview

This project demonstrates an end-to-end robotic teleoperation system that maps hand gestures captured via webcam to control a simulated robotic arm. The system uses MediaPipe for real-time hand tracking and PyBullet for physics simulation, enabling smooth control of a 7-degree-of-freedom manipulator with gripper control.

### Key Features

- **Real-time Hand Tracking**: MediaPipe-based hand landmark detection for precise gesture recognition
- **7-DOF Arm Control**: Full inverse kinematics control of Franka Panda robotic arm
- **Gripper Teleoperation**: Pinch-to-grasp gesture control with hysteresis for reliable state transitions
- **Smooth Motion Control**: Exponential moving average (EMA) filtering and rate limiting for stable trajectories
- **Object Manipulation**: Pick and place demonstrations with fine-adjust controls
- **Real-time Visualization**: On-screen overlay displaying system status and target poses

## 🚀 Demo Capabilities

The system enables a complete manipulation workflow:

1. **End-effector Control**: Move the robot's end-effector by tracking wrist position
2. **Gripper Control**: Pinch gesture to close/open gripper for object grasping
3. **Object Manipulation**: Pick up and place objects using fine-adjust controls
4. **Stable Teleoperation**: Smooth, predictable motion without jitter or sudden jumps

## 🛠️ Technical Stack

- **Computer Vision**: MediaPipe (hand landmark detection)
- **Physics Simulation**: PyBullet (robotic simulation environment)
- **Image Processing**: OpenCV (camera capture and visualization)
- **Control Algorithms**: Inverse kinematics, rate limiting, joint limit clamping
- **Programming**: Python 3.7+

## 📦 Installation

### Prerequisites

- Python 3.7 or higher
- Webcam/camera for hand tracking
- macOS, Linux, or Windows

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd robo-sim
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify MediaPipe model**
   - Ensure `hand_landmarker.task` is present in the project root
   - Download from [MediaPipe Solutions](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) if needed

## 🎮 Usage

### Running the Main Demo

Launch the Phase 1 teleoperation demo:

```bash
python teleop_ik.py
```

### Controls

- **Hand Gestures**: Move your hand in front of the camera to control the end-effector
  - Horizontal movement: Left/right (Y-axis)
  - Vertical movement: Up/down (Z-axis)
  
- **Pinch Gesture**: Bring thumb and index finger together to close gripper
  - Release pinch to open gripper
  
- **Keyboard Controls**:
  - `R`: Recenter hand position (set current position as neutral)
  - `W`: Fine-adjust up (increase Z)
  - `S`: Fine-adjust down (decrease Z)
  - `A`: Fine-adjust left (decrease Y)
  - `D`: Fine-adjust right (increase Y)
  - `Q`: Quit application

### Demo Workflow

1. Start the application
2. Position your hand in front of the camera
3. Wait for hand detection (system will auto-center)
4. Move your wrist to control the end-effector
5. Use pinch gesture to close/open gripper
6. Use fine-adjust keys (WASD) for precise positioning
7. Pick up the cube and place it at a new location

## 📁 Project Structure

```
robo-sim/
├── teleop_ik.py          # Main Phase 1 demo script (hand tracking + IK control)
├── arm_controller.py     # Robotic arm controller with IK, rate limiting, gripper control
├── hand_tracker.py       # MediaPipe hand tracking wrapper
├── simulation.py         # Basic PyBullet simulation utilities
├── teleoperation.py      # Simple 1-DOF teleoperation prototype
├── requirements.txt      # Python dependencies
├── hand_landmarker.task  # MediaPipe hand landmark model
└── sample_urdfs/         # Sample robot URDF files
```

## 🔧 Technical Details

### Control Architecture

- **Inverse Kinematics**: PyBullet's IK solver for end-effector pose control
- **Rate Limiting**: Maximum joint velocity constraints (0.02 rad/step) for smooth motion
- **Joint Limits**: Automatic clamping to prevent invalid configurations
- **EMA Smoothing**: Exponential moving average filter (α=0.85) for trajectory smoothing
- **Workspace Constraints**: Configurable workspace boundaries for safe operation

### Hand Tracking Pipeline

1. Camera captures frame
2. MediaPipe processes frame → hand landmarks
3. Wrist position mapped to robot workspace coordinates
4. EMA filter applied for smoothness
5. Target pose sent to arm controller
6. IK solver computes joint angles
7. Motors actuate with rate limiting and joint limits

### Gripper Control

- **Pinch Detection**: Euclidean distance between thumb tip (landmark 4) and index tip (landmark 8)
- **Hysteresis**: Dual thresholds prevent state chattering (close: 0.035, open: 0.050)
- **State Machine**: Open/closed states with reliable transitions

## ✅ Phase 1 Checklist

The current implementation (Phase 1) achieves:

- ✅ Smooth end-effector teleoperation (no jitter/snap)
- ✅ Reliable pinch-to-grasp gripper control
- ✅ Stable scene loading (table + cube)
- ✅ Repeatable object pickup and manipulation
- ✅ Clean startup/shutdown with error handling
- ✅ Real-time status overlay and visualization

## 🎓 Learning Outcomes

This project demonstrates proficiency in:

- **Robotics**: Inverse kinematics, joint control, workspace management
- **Computer Vision**: Real-time hand tracking and gesture recognition
- **Control Systems**: Filtering, rate limiting, state machines
- **Software Engineering**: Modular design, error handling, clean code
- **Integration**: Combining multiple libraries and systems in real-time

## 🔮 Future Enhancements (Phase 2+)

- Multi-DOF orientation control via hand rotation
- Dynamic object tracking and automated grasping
- Haptic feedback integration
- Multi-hand coordination
- Learning-based gesture recognition
- ROS integration for real robot deployment

## 📝 Notes

- The system uses CPU-based MediaPipe processing (`MEDIAPIPE_DISABLE_GPU=1`) for compatibility
- Workspace is constrained to a fixed X position (0.55m) for Phase 1 stability
- Fine-adjust controls are included to ensure reliable object manipulation
- The simulation uses PyBullet's built-in Franka Panda URDF model

## 📄 License

This project is available for educational and demonstration purposes.

## 👤 Author

Developed as a demonstration of robotic teleoperation and computer vision integration.

---

**Status**: Phase 1 Complete ✅ | Ready for demonstration
