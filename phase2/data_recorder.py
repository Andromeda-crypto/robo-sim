# data_recorder.py - Phase 2: Recording pipeline for episode data
#
# Records teleoperation episodes at 30-60 Hz with all required state information.
# Saves to NPZ format for efficient storage and loading.

import numpy as np
import time
import os
from datetime import datetime
from phase2.task_evaluation import get_task_info


class EpisodeRecorder:
    """
    Records robot state data during teleoperation episodes.
    """
    
    def __init__(self, sampling_rate=30.0, output_dir="data/episodes"):
        """
        Initialize recorder.
        
        Args:
            sampling_rate: Recording frequency in Hz (30-60 Hz recommended)
            output_dir: Directory to save episode files
        """
        self.sampling_rate = sampling_rate
        self.output_dir = output_dir
        self.min_interval = 1.0 / sampling_rate
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Recording state
        self.is_recording = False
        self.episode_data = None
        self.last_record_time = 0.0
        self.episode_start_time = None
        self.episode_id = None
        
    def start_episode(self, episode_id=None):
        """
        Start recording a new episode.
        
        Args:
            episode_id: Optional episode ID. If None, auto-generates.
        """
        if self.is_recording:
            print("Warning: Already recording. Call stop_episode() first.")
            return
        
        # Generate episode ID if not provided
        if episode_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            episode_id = f"ep_{timestamp}"
        
        self.episode_id = episode_id
        self.episode_start_time = time.time()
        self.last_record_time = 0.0
        
        # Initialize data structures
        self.episode_data = {
            'metadata': {
                'episode_id': episode_id,
                'timestamp': datetime.now().isoformat(),
                'sampling_rate': self.sampling_rate,
                'task_info': get_task_info(),
            },
            'states': {
                'timestamps': [],
                'joint_positions': [],
                'gripper_openings': [],
                'ee_positions': [],
                'ee_orientations': [],
                'cube_positions': [],
                'cube_orientations': [],
            },
            'actions': {
                'target_ee_positions': [],
                'gripper_commands': [],
            }
        }
        
        self.is_recording = True
        print(f"Started recording episode: {episode_id}")
    
    def record_step(self, arm_controller, target_ee_pos=None, gripper_command=None):
        """
        Record one timestep of data.
        
        Args:
            arm_controller: ArmController instance
            target_ee_pos: Optional target end-effector position [x, y, z]
            gripper_command: Optional gripper command (opening value)
        
        Returns:
            bool: True if data was recorded, False if skipped (rate limiting)
        """
        if not self.is_recording:
            return False
        
        # Rate limiting: only record at specified sampling rate
        current_time = time.time()
        elapsed = current_time - self.episode_start_time
        
        if elapsed - self.last_record_time < self.min_interval:
            return False  # Skip this frame
        
        self.last_record_time = elapsed
        
        # Get current state
        arm_pos, gripper_pos = arm_controller.get_all_joint_positions()
        ee_pos, ee_orn = arm_controller.get_end_effector_pose()
        cube_pos, cube_orn = arm_controller.get_cube_pose()
        
        # Record state
        self.episode_data['states']['timestamps'].append(elapsed)
        self.episode_data['states']['joint_positions'].append(arm_pos)
        self.episode_data['states']['gripper_openings'].append(gripper_pos[0])  # Use first gripper joint
        self.episode_data['states']['ee_positions'].append(ee_pos)
        self.episode_data['states']['ee_orientations'].append(ee_orn)
        self.episode_data['states']['cube_positions'].append(cube_pos)
        self.episode_data['states']['cube_orientations'].append(cube_orn)
        
        # Record actions (if provided)
        if target_ee_pos is not None:
            self.episode_data['actions']['target_ee_positions'].append(target_ee_pos)
        if gripper_command is not None:
            self.episode_data['actions']['gripper_commands'].append(gripper_command)
        
        return True
    
    def stop_episode(self):
        """
        Stop recording and return episode data.
        
        Returns:
            dict: Episode data dictionary
        """
        if not self.is_recording:
            print("Warning: Not currently recording.")
            return None
        
        self.is_recording = False
        
        # Convert lists to numpy arrays
        episode_data = self._convert_to_numpy(self.episode_data)
        
        # Update metadata with final duration
        duration = episode_data['states']['timestamps'][-1] if len(episode_data['states']['timestamps']) > 0 else 0.0
        episode_data['metadata']['duration'] = float(duration)
        episode_data['metadata']['num_timesteps'] = len(episode_data['states']['timestamps'])
        
        print(f"Stopped recording. Duration: {duration:.2f}s, Timesteps: {len(episode_data['states']['timestamps'])}")
        
        return episode_data
    
    def save_episode(self, episode_data=None, filename=None):
        """
        Save episode data to NPZ file.
        
        Args:
            episode_data: Episode data dict. If None, uses current episode.
            filename: Optional filename. If None, uses episode_id.
        
        Returns:
            str: Path to saved file
        """
        if episode_data is None:
            if self.episode_data is None:
                print("Error: No episode data to save.")
                return None
            episode_data = self._convert_to_numpy(self.episode_data)
        
        # Determine filename
        if filename is None:
            if self.episode_id is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"ep_{timestamp}.npz"
            else:
                filename = f"{self.episode_id}.npz"
        
        # Ensure .npz extension
        if not filename.endswith('.npz'):
            filename += '.npz'
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Save to NPZ (numpy's compressed format)
        # We need to flatten the nested structure for NPZ
        save_dict = {}
        
        # Metadata (save as strings/numbers)
        for key, value in episode_data['metadata'].items():
            if isinstance(value, dict):
                # Task info dict - convert to string representation
                save_dict[f'metadata_{key}'] = str(value)
            else:
                save_dict[f'metadata_{key}'] = value
        
        # States (numpy arrays)
        for key, value in episode_data['states'].items():
            if len(value) > 0:
                save_dict[f'states_{key}'] = np.array(value)
        
        # Actions (numpy arrays, may be empty)
        for key, value in episode_data['actions'].items():
            if len(value) > 0:
                save_dict[f'actions_{key}'] = np.array(value)
        
        np.savez_compressed(filepath, **save_dict)
        print(f"Saved episode to: {filepath}")
        
        return filepath
    
    def _convert_to_numpy(self, episode_data):
        """Convert list-based data to numpy arrays."""
        converted = {
            'metadata': episode_data['metadata'].copy(),
            'states': {},
            'actions': {}
        }
        
        # Convert state lists to numpy arrays
        for key, value_list in episode_data['states'].items():
            if len(value_list) > 0:
                converted['states'][key] = np.array(value_list)
            else:
                converted['states'][key] = np.array([])
        
        # Convert action lists to numpy arrays
        for key, value_list in episode_data['actions'].items():
            if len(value_list) > 0:
                converted['actions'][key] = np.array(value_list)
            else:
                converted['actions'][key] = np.array([])
        
        return converted


def load_episode(filepath):
    """
    Load episode data from NPZ file.
    
    Args:
        filepath: Path to .npz file
    
    Returns:
        dict: Episode data in same format as recorded
    """
    data = np.load(filepath, allow_pickle=True)
    
    # Reconstruct nested structure
    episode_data = {
        'metadata': {},
        'states': {},
        'actions': {}
    }
    
    # Extract metadata
    for key in data.keys():
        if key.startswith('metadata_'):
            meta_key = key.replace('metadata_', '')
            episode_data['metadata'][meta_key] = data[key].item() if data[key].size == 1 else str(data[key])
    
    # Extract states
    for key in data.keys():
        if key.startswith('states_'):
            state_key = key.replace('states_', '')
            episode_data['states'][state_key] = data[key]
    
    # Extract actions
    for key in data.keys():
        if key.startswith('actions_'):
            action_key = key.replace('actions_', '')
            episode_data['actions'][action_key] = data[key]
    
    return episode_data

