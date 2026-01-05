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
- **Data Collection**: Phase 2 adds episode recording and replay capabilities

## 📁 Project Structure

```
robo-sim/
├── core/                    # Core robotics components
│   ├── arm_controller.py    # Robot arm controller with IK
│   ├── hand_tracker.py     # MediaPipe hand tracking
│   └── simulation.py       # Basic simulation utilities
├── phase2/                  # Phase 2: Data collection & replay
│   ├── data_recorder.py    # Episode recording pipeline
│   ├── task_evaluation.py  # Task definition and success metrics
│   ├── replay.py           # Episode replay system
│   ├── dataset_manager.py  # Dataset indexing and statistics
│   └── dataset_viewer.py  # Interactive dataset browser
├── scripts/                 # Executable scripts
│   ├── teleop_ik.py        # Phase 1: Main teleoperation demo
│   ├── teleop_record.py    # Phase 2: Teleoperation with recording
│   ├── teleoperation.py    # Simple 1-DOF prototype
│   └── test_arm_controller.py
├── data/                   # Data files
│   ├── episodes/          # Recorded episode files (.npz)
│   ├── models/           # ML models (hand_landmarker.task)
│   └── sample_urdfs/     # Sample robot URDF files
├── docs/                   # Documentation
│   ├── README.md          # Detailed documentation
│   └── PHASE2_GUIDE.md    # Phase 2 quick reference
└── requirements.txt       # Python dependencies
```

## 🚀 Quick Start

### Installation

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
   - Ensure `data/models/hand_landmarker.task` exists
   - Download from [MediaPipe Solutions](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) if needed

### Running the Demos

#### Phase 1: Basic Teleoperation
```bash
python scripts/teleop_ik.py
```

#### Phase 2: Teleoperation with Recording
```bash
python scripts/teleop_record.py
```

#### Replay Recorded Episodes
```bash
python phase2/replay.py --episode data/episodes/ep_XXXXX.npz
```

#### Manage Dataset
```bash
# Update dataset index
python phase2/dataset_manager.py --update

# View statistics
python phase2/dataset_manager.py --stats

# Browse dataset
python phase2/dataset_viewer.py
```

## 🎮 Controls

### Teleoperation Controls
- **Hand Gestures**: Move your hand to control the end-effector
  - Horizontal movement: Left/right (Y-axis)
  - Vertical movement: Up/down (Z-axis)
- **Pinch Gesture**: Bring thumb and index finger together to close gripper
- **Keyboard**:
  - `SPACE`: Start/Stop recording (Phase 2 only)
  - `R`: Recenter hand position
  - `WASD`: Fine-adjust controls (W=up, S=down, A=left, D=right)
  - `Q`: Quit

## 📊 Phase 2: Data Collection

Phase 2 adds complete data collection and replay capabilities:

- **Episode Recording**: Record demonstrations at 30 Hz
- **Success Evaluation**: Automated task completion evaluation
- **Episode Replay**: Reproducible trajectory playback
- **Dataset Management**: Index and statistics for collected data

See [docs/PHASE2_GUIDE.md](docs/PHASE2_GUIDE.md) for detailed Phase 2 documentation.

## 🛠️ Technical Stack

- **Computer Vision**: MediaPipe (hand landmark detection)
- **Physics Simulation**: PyBullet (robotic simulation environment)
- **Image Processing**: OpenCV (camera capture and visualization)
- **Control Algorithms**: Inverse kinematics, rate limiting, joint limit clamping
- **Programming**: Python 3.7+

## 📚 Documentation

- [Detailed Documentation](docs/README.md) - Complete project documentation
- [Phase 2 Guide](docs/PHASE2_GUIDE.md) - Phase 2 quick reference

## ✅ Status

- **Phase 1**: Complete ✅ - Basic teleoperation with gripper control
- **Phase 2**: Complete ✅ - Data collection and replay system

## 🔮 Future Enhancements (Phase 3+)

- **Imitation Learning**: Train models from collected demonstrations
- **Behavioral Cloning**: Learn policies from successful episodes
- **Multi-DOF Orientation**: Control via hand rotation
- **Dynamic Object Tracking**: Automated grasping with vision
- **ROS Integration**: Deploy to real robot hardware

## 📄 License

This project is available for educational and demonstration purposes.

---

**Status**: Phase 1 Complete ✅ | Phase 2 Complete ✅ | Ready for demonstration and data collection

