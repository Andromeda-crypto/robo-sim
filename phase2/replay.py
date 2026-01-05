# replay.py - Phase 2: Replay system for recorded episodes
#
# Replays recorded episodes using joint replay mode for deterministic reproduction.
# Usage:
#   python replay.py --episode episodes/ep_0001.npz
#   python replay.py --episode episodes/ep_0001.npz --speed 2.0  # 2x speed
#   python replay.py --all  # Replay all episodes

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import pybullet as p
from core.arm_controller import ArmController
from phase2.data_recorder import load_episode
from phase2.task_evaluation import evaluate_episode, visualize_target_zone, TARGET_ZONE


class EpisodeReplayer:
    """
    Replays recorded episodes using joint replay mode.
    """
    
    def __init__(self, arm_controller, playback_speed=1.0):
        """
        Initialize replayer.
        
        Args:
            arm_controller: ArmController instance
            playback_speed: Speed multiplier (1.0 = real-time, 2.0 = 2x speed)
        """
        self.ctrl = arm_controller
        self.playback_speed = playback_speed
    
    def replay_episode(self, episode_data, show_evaluation=True):
        """
        Replay an episode from loaded data.
        
        Args:
            episode_data: Episode data dictionary (from load_episode)
            show_evaluation: Whether to show evaluation results
        
        Returns:
            dict: Evaluation results
        """
        if 'states' not in episode_data:
            print("Error: Invalid episode data format")
            return None
        
        states = episode_data['states']
        
        if 'timestamps' not in states or len(states['timestamps']) == 0:
            print("Error: Episode has no timesteps")
            return None
        
        timestamps = states['timestamps']
        joint_positions = states['joint_positions']
        gripper_openings = states.get('gripper_openings', [])
        
        # Reset scene to initial state
        print("Resetting scene to initial state...")
        self.ctrl.reset_scene()
        
        # Wait a moment for reset to settle
        for _ in range(60):  # ~0.25 seconds at 240 Hz
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
        
        print(f"Replaying episode: {episode_data['metadata'].get('episode_id', 'unknown')}")
        print(f"  Duration: {episode_data['metadata'].get('duration', 0):.2f}s")
        print(f"  Timesteps: {len(timestamps)}")
        print(f"  Playback speed: {self.playback_speed}x")
        print("  Press 'q' in PyBullet window to quit early")
        
        # Calculate time per step
        if len(timestamps) > 1:
            original_dt = timestamps[1] - timestamps[0]
        else:
            original_dt = 1.0 / 30.0  # Default 30 Hz
        
        playback_dt = original_dt / self.playback_speed
        
        # Replay loop
        num_steps = len(timestamps)
        start_time = time.time()
        
        for i in range(num_steps):
            # Set joint positions directly (joint replay mode)
            for j_idx, joint_idx in enumerate(self.ctrl.arm_joints):
                if i < len(joint_positions) and j_idx < len(joint_positions[i]):
                    target_pos = joint_positions[i][j_idx]
                    p.setJointMotorControl2(
                        bodyUniqueId=self.ctrl.robot,
                        jointIndex=joint_idx,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=target_pos,
                        force=self.ctrl.motor_force,
                        positionGain=self.ctrl.position_gain,
                        velocityGain=self.ctrl.velocity_gain
                    )
            
            # Set gripper positions
            if len(gripper_openings) > i:
                gripper_opening = gripper_openings[i]
                self.ctrl.set_gripper(gripper_opening)
            
            # Step simulation
            p.stepSimulation()
            
            # Sleep to maintain playback speed
            if i < num_steps - 1:
                time.sleep(playback_dt)
            
            # Check for early exit (user closed window)
            if not p.isConnected():
                print("\nReplay interrupted (PyBullet window closed)")
                break
        
        elapsed = time.time() - start_time
        print(f"\nReplay complete. Actual time: {elapsed:.2f}s")
        
        # Evaluate the replayed episode
        if show_evaluation:
            print("\n=== Replay Evaluation ===")
            result = evaluate_episode(episode_data)
            print(f"Success: {'✓ YES' if result['success'] else '✗ NO'}")
            print(f"  - Lifted: {'✓' if result['lifted'] else '✗'}")
            print(f"  - In target zone: {'✓' if result['in_target_zone'] else '✗'}")
            print(f"  - Max height: {result['max_height']:.3f}m")
            print(f"  - Final position: {result['final_position']}")
            return result
        
        return None


def replay_single_episode(filepath, playback_speed=1.0, show_gui=True):
    """Replay a single episode file."""
    print(f"Loading episode: {filepath}")
    episode_data = load_episode(filepath)
    
    if episode_data is None:
        print("Failed to load episode")
        return
    
    # Initialize controller
    ctrl = ArmController(gui=show_gui)
    
    # Visualize target zone
    visualize_target_zone()
    
    try:
        # Create replayer
        replayer = EpisodeReplayer(ctrl, playback_speed=playback_speed)
        
        # Replay
        replayer.replay_episode(episode_data, show_evaluation=True)
        
        # Keep window open
        print("\nReplay finished. Close PyBullet window to exit.")
        while p.isConnected():
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
            
    finally:
        ctrl.disconnect()


def replay_all_episodes(directory="data/episodes", playback_speed=1.0, filter_success=False):
    """Replay all episodes in a directory."""
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return
    
    # Find all .npz files
    episode_files = [f for f in os.listdir(directory) if f.endswith('.npz')]
    
    if filter_success:
        # Only replay successful episodes
        episode_files = [f for f in episode_files if '_success' in f]
    
    if len(episode_files) == 0:
        print(f"No episodes found in {directory}")
        return
    
    episode_files.sort()
    print(f"Found {len(episode_files)} episodes to replay")
    
    # Initialize controller once
    ctrl = ArmController(gui=True)
    visualize_target_zone()
    replayer = EpisodeReplayer(ctrl, playback_speed=playback_speed)
    
    try:
        for i, filename in enumerate(episode_files):
            filepath = os.path.join(directory, filename)
            print(f"\n{'='*60}")
            print(f"Replaying {i+1}/{len(episode_files)}: {filename}")
            print(f"{'='*60}")
            
            episode_data = load_episode(filepath)
            if episode_data:
                replayer.replay_episode(episode_data, show_evaluation=True)
                
                # Pause between episodes
                if i < len(episode_files) - 1:
                    print("\nNext episode in 3 seconds...")
                    time.sleep(3.0)
            else:
                print(f"Failed to load {filename}")
        
        print("\nAll episodes replayed. Close PyBullet window to exit.")
        while p.isConnected():
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
            
    finally:
        ctrl.disconnect()


def main():
    parser = argparse.ArgumentParser(description='Replay recorded teleoperation episodes')
    parser.add_argument('--episode', type=str, help='Path to episode file (.npz)')
    parser.add_argument('--all', action='store_true', help='Replay all episodes in episodes/ directory')
    parser.add_argument('--filter', type=str, choices=['success', 'fail'], 
                       help='Filter episodes by success status (only with --all)')
    parser.add_argument('--speed', type=float, default=1.0, 
                       help='Playback speed multiplier (default: 1.0 = real-time)')
    parser.add_argument('--no-gui', action='store_true', help='Run without GUI (faster)')
    
    args = parser.parse_args()
    
    if args.all:
        filter_success = (args.filter == 'success')
        replay_all_episodes(playback_speed=args.speed, filter_success=filter_success)
    elif args.episode:
        if not os.path.exists(args.episode):
            print(f"Error: Episode file not found: {args.episode}")
            return
        replay_single_episode(args.episode, playback_speed=args.speed, show_gui=not args.no_gui)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python replay.py --episode episodes/ep_20240115_103000.npz")
        print("  python replay.py --episode episodes/ep_20240115_103000.npz --speed 2.0")
        print("  python replay.py --all")
        print("  python replay.py --all --filter success")


if __name__ == "__main__":
    main()

