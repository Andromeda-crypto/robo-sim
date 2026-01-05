# Project Organization

## Folder Structure

The project has been organized into logical folders for better maintainability:

```
robo-sim/
├── core/                    # Core robotics components
│   ├── __init__.py
│   ├── arm_controller.py    # Robot arm controller with IK
│   ├── hand_tracker.py     # MediaPipe hand tracking
│   └── simulation.py        # Basic simulation utilities
│
├── phase2/                  # Phase 2: Data collection & replay
│   ├── __init__.py
│   ├── data_recorder.py    # Episode recording pipeline
│   ├── task_evaluation.py  # Task definition and success metrics
│   ├── replay.py           # Episode replay system
│   ├── dataset_manager.py  # Dataset indexing and statistics
│   └── dataset_viewer.py   # Interactive dataset browser
│
├── scripts/                 # Executable scripts
│   ├── __init__.py
│   ├── teleop_ik.py        # Phase 1: Main teleoperation demo
│   ├── teleop_record.py    # Phase 2: Teleoperation with recording
│   ├── teleoperation.py    # Simple 1-DOF prototype
│   ├── test_arm_controller.py
│   └── ik_sandbox.py
│
├── data/                    # Data files
│   ├── episodes/           # Recorded episode files (.npz)
│   ├── models/             # ML models
│   │   └── hand_landmarker.task
│   └── sample_urdfs/       # Sample robot URDF files
│       └── single_joint_arm.xml
│
├── docs/                    # Documentation
│   ├── DETAILED_README.md  # Complete project documentation
│   └── PHASE2_GUIDE.md     # Phase 2 quick reference
│
├── README.md               # Main project README
├── requirements.txt        # Python dependencies
└── ORGANIZATION.md         # This file
```

## Import Paths

All scripts use relative imports with path setup. The structure ensures:

- **Core components** are in `core/` and imported as `from core.module import Class`
- **Phase 2 components** are in `phase2/` and imported as `from phase2.module import Class`
- **Scripts** add the parent directory to `sys.path` to enable imports

## Running Scripts

All scripts can be run from the project root:

```bash
# Phase 1: Basic teleoperation
python scripts/teleop_ik.py

# Phase 2: Teleoperation with recording
python scripts/teleop_record.py

# Replay episodes
python phase2/replay.py --episode data/episodes/ep_XXXXX.npz

# Manage dataset
python phase2/dataset_manager.py --update --stats

# Browse dataset
python phase2/dataset_viewer.py
```

## Data Paths

- **Episodes**: `data/episodes/` (created automatically when recording)
- **Dataset Index**: `data/dataset_index.csv` (created by dataset_manager)
- **Models**: `data/models/hand_landmarker.task`
- **URDFs**: `data/sample_urdfs/`

## Notes

- All imports have been updated to work with the new structure
- Paths are relative to the project root
- Scripts automatically add the parent directory to Python path
- The structure is designed to be scalable for future phases

