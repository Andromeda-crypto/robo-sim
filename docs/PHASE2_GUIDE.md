# Phase 2 Quick Reference Guide

## Overview

Phase 2 adds data collection and replay capabilities to the teleoperation system. You can now:
- Record teleoperation episodes
- Replay recorded episodes
- Evaluate success metrics
- Manage a dataset of demonstrations

## Quick Start

### 1. Record an Episode

```bash
python teleop_record.py
```

**Controls:**
- `SPACE`: Start/Stop recording
- `R`: Recenter hand position
- `Q`: Quit
- `WASD`: Fine-adjust controls

**Task:** Pick the cube and place it in the green target zone.

### 2. Replay an Episode

```bash
# Replay a specific episode
python replay.py --episode episodes/ep_20240115_103000.npz

# Replay at 2x speed
python replay.py --episode episodes/ep_20240115_103000.npz --speed 2.0

# Replay all successful episodes
python replay.py --all --filter success
```

### 3. Manage Dataset

```bash
# Update dataset index (after recording episodes)
python dataset_manager.py --update

# View dataset statistics
python dataset_manager.py --stats

# List all episodes
python dataset_manager.py --list

# List only successful episodes
python dataset_manager.py --list --filter success
```

### 4. Browse Dataset

```bash
python dataset_viewer.py
```

Interactive browser to view and replay episodes.

## File Structure

```
robo-sim/
├── teleop_record.py      # Teleoperation with recording (Phase 2)
├── replay.py              # Episode replay system
├── dataset_manager.py     # Dataset indexing and statistics
├── dataset_viewer.py     # Interactive dataset browser
├── task_evaluation.py     # Task definition and success evaluation
├── data_recorder.py       # Episode recording pipeline
├── episodes/              # Recorded episode files (.npz)
└── dataset_index.csv     # Dataset index with metadata
```

## Success Criteria

An episode is considered successful if:
1. **Lifted**: Cube was lifted at least 15cm above initial height
2. **In Target Zone**: Final cube position is within the target zone
3. **Time Limit**: Completed within 20 seconds (optional)

## Episode Data Format

Episodes are saved as `.npz` files containing:
- **Metadata**: Episode ID, timestamp, duration, task info
- **States**: Joint positions, gripper state, end-effector pose, cube pose
- **Actions**: Target poses, gripper commands

## Dataset Collection Workflow

1. **Record episodes**: Use `teleop_record.py` to record demonstrations
2. **Update index**: Run `dataset_manager.py --update` to index all episodes
3. **Review**: Use `dataset_viewer.py` to browse and replay episodes
4. **Analyze**: Check statistics with `dataset_manager.py --stats`

## Tips for Recording

- **Start recording** before beginning the task
- **Stop recording** after completing (or failing) the task
- Use **fine-adjust keys** (WASD) for precise positioning
- Record **both successes and failures** for a complete dataset
- Aim for **15-30 successful episodes** for Phase 2

## Next Steps (Phase 3)

After collecting a dataset, Phase 3 could include:
- Training imitation learning models
- Behavioral cloning from demonstrations
- Policy learning from successful episodes

