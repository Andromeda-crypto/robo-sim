# teleop_record.py - Phase 2: Teleoperation with recording capability
#
# Extends teleop_ik.py with episode recording functionality.
# Press SPACE to start/stop recording episodes.

import os
import sys
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import time
import cv2
import pybullet as p
from core.arm_controller import ArmController
from core.hand_tracker import HandTracker
from phase2.data_recorder import EpisodeRecorder
from phase2.task_evaluation import evaluate_episode, visualize_target_zone, TARGET_ZONE
from scripts.teleop_ik import (
    SENS_Y, SENS_Z, PINCH_CLOSE_T, PINCH_OPEN_T,
    GRIP_OPEN, GRIP_CLOSED, FIXED_X, WORKSPACE_Y, WORKSPACE_Z,
    ALPHA_EMA, LOST_HOLD_SEC, FINE_STEP,
    pinch_distance, clamp, map_hand_to_pose, ema_filter, draw_overlay
)


def draw_recording_overlay(frame, is_recording, episode_id, num_timesteps):
    """Add recording status to overlay."""
    h, w = frame.shape[:2]
    

    if is_recording:
        # Red blinking indicator
        color = (0, 0, 255) if int(time.time() * 2) % 2 == 0 else (0, 100, 255)
        cv2.circle(frame, (w - 30, 30), 15, color, -1)
        cv2.putText(frame, "REC", (w - 60, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        if episode_id:
            cv2.putText(frame, f"Episode: {episode_id}", (w - 200, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        if num_timesteps > 0:
            cv2.putText(frame, f"Steps: {num_timesteps}", (w - 200, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return frame


def main():
    """Main teleoperation loop with recording."""
    
    # Initialize components
    print("Initializing Phase 2 teleoperation with recording...")
    ctrl = None
    tracker = None
    cap = None
    recorder = None
    target_zone_visual = None
    
    try:
        # Initialize controller
        ctrl = ArmController(gui=True)
        print("Robot controller initialized")

        target_zone_visual = visualize_target_zone()
        print(f"Target zone visualized at {TARGET_ZONE['center']}")
        
        # Initialize hand tracker
        tracker = HandTracker()
        print("Hand tracker initialized")
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Failed to open camera")
        print("Camera initialized")
        
        # Initialize recorder
        recorder = EpisodeRecorder(sampling_rate=30.0, output_dir="data/episodes")
        print("Data recorder initialized")
        
        print("\n=== Phase 2 Teleoperation Ready ===")
        print("Controls:")
        print("  SPACE: Start/Stop recording episode")
        print("  R: Recenter hand position")
        print("  Q: Quit")
        print("  WASD: Fine-adjust (W=up, S=down, A=left, D=right)")
        print("\nTask: Pick cube and place in green target zone")
        
        # State variables
        grip_state = "open"
        filtered_pos = None
        neutral_wrist = None
        base_pos = [FIXED_X, 0.0, 0.35]
        orn_fixed = p.getQuaternionFromEuler([0, 3.14159, 0])
        last_seen = 0.0
        fine_adjust_active = False
        
        # Recording state
        recording_episode_id = None
        num_recorded_steps = 0
        
        # Main loop
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to read frame from camera")
                break
            
            hands = tracker.process_frame(frame)
            try:
                frame = tracker.draw_landmarks(frame, hands)
            except (TypeError, AttributeError):
                pass
            
            now = time.time()
            target_ee_pos = None
            gripper_cmd = None
            
            if hands and len(hands) > 0:
                wrist = hands[0][0]
                if neutral_wrist is None:
                    neutral_wrist = (wrist.x, wrist.y)
                    print("Neutral wrist position set")
                
                # Map hand to robot pose
                pos, orn = map_hand_to_pose(hands[0], neutral_wrist, base_pos)
                filtered_pos = ema_filter(filtered_pos, pos, ALPHA_EMA)
                target_ee_pos = filtered_pos
                ctrl.set_target_pose(filtered_pos, orn)
                
                last_seen = now
                
                pinch_dist = pinch_distance(hands[0])
                if grip_state == "open" and pinch_dist < PINCH_CLOSE_T:
                    grip_state = "closed"
                elif grip_state == "closed" and pinch_dist > PINCH_OPEN_T:
                    grip_state = "open"
                
                # Control gripper
                gripper_cmd = GRIP_CLOSED if grip_state == "closed" else GRIP_OPEN
                ctrl.set_gripper(gripper_cmd)
                
            else:
                # Hand lost - hold last position
                if filtered_pos is not None and (now - last_seen) < LOST_HOLD_SEC:
                    ctrl.set_target_pose(filtered_pos, orn_fixed)
            
            # Record step if recording
            if recorder.is_recording:
                recorded = recorder.record_step(ctrl, target_ee_pos, gripper_cmd)
                if recorded:
                    num_recorded_steps += 1
            
            # Step simulation
            try:
                ctrl.step()
            except RuntimeError as e:
                print(f"Simulation error: {e}")
                break
            
            # Draw overlays
            frame = draw_overlay(frame, hands, grip_state, filtered_pos, fine_adjust_active)
            frame = draw_recording_overlay(frame, recorder.is_recording, 
                                          recording_episode_id, num_recorded_steps)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord("q"):
                # Stop recording if active
                if recorder.is_recording:
                    print("\nStopping recording before quit...")
                    episode_data = recorder.stop_episode()
                    if episode_data:
                        # Evaluate episode
                        result = evaluate_episode(episode_data)
                        print(f"\nEpisode evaluation: {'SUCCESS' if result['success'] else 'FAILED'}")
                        print(f"  Lifted: {result['lifted']}")
                        print(f"  In target zone: {result['in_target_zone']}")
                        print(f"  Details: {result['details']}")
                        
                        # Save episode
                        filepath = recorder.save_episode(episode_data)
                        print(f"Saved to: {filepath}")
                print("Quitting...")
                break
            
            elif key == ord(" "):  # SPACE key
                if not recorder.is_recording:
                    # Start recording
                    recorder.start_episode()
                    recording_episode_id = recorder.episode_id
                    num_recorded_steps = 0
                    print(f"\n>>> RECORDING STARTED: {recording_episode_id}")
                    print("Press SPACE again to stop recording")
                else:
                    # Stop recording
                    episode_data = recorder.stop_episode()
                    if episode_data:
                        # Evaluate episode
                        result = evaluate_episode(episode_data)
                        print(f"\n=== Episode Evaluation ===")
                        print(f"Episode ID: {recording_episode_id}")
                        print(f"Success: {'✓ YES' if result['success'] else '✗ NO'}")
                        print(f"  - Lifted: {'✓' if result['lifted'] else '✗'}")
                        print(f"  - In target zone: {'✓' if result['in_target_zone'] else '✗'}")
                        print(f"  - Max height: {result['max_height']:.3f}m")
                        print(f"  - Final position: {result['final_position']}")
                        print(f"  - Duration: {episode_data['metadata']['duration']:.2f}s")
                        print(f"  - Timesteps: {episode_data['metadata']['num_timesteps']}")
                        
                        # Save episode
                        filepath = recorder.save_episode(episode_data)
                        print(f"\nSaved to: {filepath}")
                        
                        # Add success status to filename
                        if result['success']:
                            success_filepath = filepath.replace('.npz', '_success.npz')
                            import shutil
                            shutil.move(filepath, success_filepath)
                            print(f"Renamed to: {success_filepath}")
                    
                    recording_episode_id = None
                    num_recorded_steps = 0
                    print("\n>>> RECORDING STOPPED")
                    print("Press SPACE to start new recording")
            
            elif key == ord("r") and hands and len(hands) > 0:
                # Recenter
                wrist = hands[0][0]
                neutral_wrist = (wrist.x, wrist.y)
                if filtered_pos is not None:
                    base_pos = filtered_pos.copy()
                print("Recentered to current position")
            
            # Fine-adjust keys
            elif key == ord("w") and filtered_pos is not None:
                filtered_pos[2] = clamp(filtered_pos[2] + FINE_STEP, WORKSPACE_Z[0], WORKSPACE_Z[1])
                ctrl.set_target_pose(filtered_pos, orn_fixed)
                fine_adjust_active = True
            elif key == ord("s") and filtered_pos is not None:
                filtered_pos[2] = clamp(filtered_pos[2] - FINE_STEP, WORKSPACE_Z[0], WORKSPACE_Z[1])
                ctrl.set_target_pose(filtered_pos, orn_fixed)
                fine_adjust_active = True
            elif key == ord("a") and filtered_pos is not None:
                filtered_pos[1] = clamp(filtered_pos[1] - FINE_STEP, WORKSPACE_Y[0], WORKSPACE_Y[1])
                ctrl.set_target_pose(filtered_pos, orn_fixed)
                fine_adjust_active = True
            elif key == ord("d") and filtered_pos is not None:
                filtered_pos[1] = clamp(filtered_pos[1] + FINE_STEP, WORKSPACE_Y[0], WORKSPACE_Y[1])
                ctrl.set_target_pose(filtered_pos, orn_fixed)
                fine_adjust_active = True
            else:
                fine_adjust_active = False
            
            # Display frame
            cv2.imshow("Teleoperation - Phase 2 (SPACE to record)", frame)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        if recorder and recorder.is_recording:
            episode_data = recorder.stop_episode()
            if episode_data:
                recorder.save_episode(episode_data)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean shutdown
        print("Cleaning up...")
        if recorder and recorder.is_recording:
            episode_data = recorder.stop_episode()
            if episode_data:
                recorder.save_episode(episode_data)
        if tracker is not None:
            try:
                tracker.close()
            except Exception:
                pass
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if ctrl is not None:
            ctrl.disconnect()
        print("Shutdown complete")


if __name__ == "__main__":
    main()

