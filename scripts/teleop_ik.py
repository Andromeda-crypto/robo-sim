# teleop_ik.py -: Hand gesture teleoperation with gripper control
# 
# Phase 1 Requirements:
# - Stable, predictable teleop
# - Pinch to open/close gripper
# - Pick up cube and place it
# - Clean 20-40s demo

import os
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import cv2
import pybullet as p
from core.arm_controller import ArmController
from core.hand_tracker import HandTracker



FIXED_X = 0.55 
WORKSPACE_Y = (-0.30, 0.30)
WORKSPACE_Z = (0.10, 0.60)  # Lowered min to allow reaching cube at z=0.05

# Hand-to-robot mapping sensitivity
SENS_Y = 0.6  # Horizontal (left-right)
SENS_Z = 0.6  # Vertical (up-down)


PINCH_CLOSE_T = 0.035  
PINCH_OPEN_T = 0.050   

# Gripper states
GRIP_OPEN = 0.04
GRIP_CLOSED = 0.0

# Smoothing parameters
ALPHA_EMA = 0.85  
LOST_HOLD_SEC = 0.3  


FINE_STEP = 0.01  # 1cm steps



def pinch_distance(hand_landmarks):
    """Compute 2D distance between thumb tip and index tip."""
    thumb = hand_landmarks[4]
    index = hand_landmarks[8]
    dx = thumb.x - index.x
    dy = thumb.y - index.y
    return (dx*dx + dy*dy) ** 0.5


def clamp(value, min_val, max_val):
    """Clamp value to range [min_val, max_val]."""
    return max(min_val, min(max_val, value))


def map_hand_to_pose(hand_landmarks, neutral_wrist, base_pos):
    """
    Map hand wrist position to robot end-effector pose.
    
    Args:
        hand_landmarks: MediaPipe hand landmarks
        neutral_wrist: (x, y) normalized coordinates of wrist when centered
        base_pos: [x, y, z] base position in robot frame
    
    Returns:
        pos: [x, y, z] target position
        orn: [qx, qy, qz, qw] target orientation (quaternion)
    """
    wrist = hand_landmarks[0]
    dx = wrist.x - neutral_wrist[0]
    dy = wrist.y - neutral_wrist[1]
    
    y = base_pos[1] + dx * SENS_Y
    z = base_pos[2] + dy * SENS_Z
    
    y = clamp(y, WORKSPACE_Y[0], WORKSPACE_Y[1])
    z = clamp(z, WORKSPACE_Z[0], WORKSPACE_Z[1])
    

    x = FIXED_X
    

    orn = p.getQuaternionFromEuler([0, 3.14159, 0])
    
    return [x, y, z], orn


def ema_filter(previous, new, alpha):
    """Exponential moving average filter."""
    if previous is None:
        return new
    return [alpha * a + (1 - alpha) * b for a, b in zip(previous, new)]


def draw_overlay(frame, hands, gripper_state, target_pos, fine_adjust_active):
    """
    Draw on-screen overlay with system status.
    
    Args:
        frame: OpenCV frame
        hands: List of detected hands
        gripper_state: "open" or "closed"
        target_pos: [x, y, z] target position
        fine_adjust_active: bool, whether fine-adjust mode is active
    """
    h, w = frame.shape[:2]
    

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, 130), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (255, 255, 255)
    thickness = 1
    y_offset = 30
    line_height = 25
    
    # Hands detected
    num_hands = 0 if not hands else len(hands)
    cv2.putText(frame, f"Hands: {num_hands}", (20, y_offset), 
                font, font_scale, color, thickness)
    
    # Gripper state
    grip_color = (0, 255, 0) if gripper_state == "closed" else (0, 200, 255)
    cv2.putText(frame, f"Gripper: {gripper_state.upper()}", (20, y_offset + line_height),
                font, font_scale, grip_color, thickness)
    
    # Target position
    if target_pos is not None:
        pos_str = f"Target: [{target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}]"
        cv2.putText(frame, pos_str, (20, y_offset + 2*line_height),
                    font, font_scale, color, thickness)
    
    # Fine-adjust indicator
    if fine_adjust_active:
        cv2.putText(frame, "FINE-ADJUST MODE", (20, y_offset + 3*line_height),
                    font, 0.5, (255, 255, 0), thickness)
    
    # Instructions
    cv2.putText(frame, "Q: quit | R: recenter | WASD: fine-adjust", (20, y_offset + 4*line_height),
                font, 0.4, (200, 200, 200), thickness)
    
    return frame



def main():
    """Main teleoperation demo loop."""
    
    # Initialize components
    print("Initializing Phase 1 demo...")
    ctrl = None
    tracker = None
    cap = None
    
    try:
        # Initialize controller (creates PyBullet GUI)
        ctrl = ArmController(gui=True)
        print("Robot controller initialized")
        
        # Initialize hand tracker
        tracker = HandTracker()
        print("Hand tracker initialized")
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Failed to open camera")
        print("Camera initialized")
        print("\n=== Phase 1 Demo Ready ===")
        print("Show your hand to the camera to begin")
        print("Press 'r' to recenter, 'q' to quit")
        print("WASD keys for fine-adjust (W=up, S=down, A=left, D=right)")
        
        # State variables
        grip_state = "open"
        filtered_pos = None
        neutral_wrist = None
        base_pos = [FIXED_X, 0.0, 0.35]  # Starting position
        orn_fixed = p.getQuaternionFromEuler([0, 3.14159, 0])
        last_seen = 0.0
        fine_adjust_active = False
        
        # Main loop
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to read frame from camera")
                break
            
            # Process hand tracking
            hands = tracker.process_frame(frame)
            
            # Draw hand landmarks
            try:
                frame = tracker.draw_landmarks(frame, hands)
            except (TypeError, AttributeError):
                # Handle case where hands is None or empty
                pass
            
            now = time.time()
            
            # Handle hand detection
            if hands and len(hands) > 0:
                wrist = hands[0][0]
                if neutral_wrist is None:
                    neutral_wrist = (wrist.x, wrist.y)
                    print("Neutral wrist position set")
                
                # Map hand to robot pose
                pos, orn = map_hand_to_pose(hands[0], neutral_wrist, base_pos)
                filtered_pos = ema_filter(filtered_pos, pos, ALPHA_EMA)
                ctrl.set_target_pose(filtered_pos, orn)
                
                last_seen = now
            
                pinch_dist = pinch_distance(hands[0])
                if grip_state == "open" and pinch_dist < PINCH_CLOSE_T:
                    grip_state = "closed"
                elif grip_state == "closed" and pinch_dist > PINCH_OPEN_T:
                    grip_state = "open"
                
                # Control gripper
                ctrl.set_gripper(GRIP_CLOSED if grip_state == "closed" else GRIP_OPEN)
                
            else:
                # Hand lost - hold last position for short time
                if filtered_pos is not None and (now - last_seen) < LOST_HOLD_SEC:
                    ctrl.set_target_pose(filtered_pos, orn_fixed)
            
            # Step simulation
            try:
                ctrl.step()
            except RuntimeError as e:
                print(f"Simulation error: {e}")
                break
            
            # Draw overlay
            frame = draw_overlay(frame, hands, grip_state, filtered_pos, fine_adjust_active)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord("q"):
                print("Quitting...")
                break
            
            elif key == ord("r") and hands and len(hands) > 0:
                wrist = hands[0][0]
                neutral_wrist = (wrist.x, wrist.y)
                if filtered_pos is not None:
                    base_pos = filtered_pos.copy()
                print("Recentered to current position")
            
            elif key == ord("w") and filtered_pos is not None:
                # Move up (increase z)
                filtered_pos[2] = clamp(filtered_pos[2] + FINE_STEP, WORKSPACE_Z[0], WORKSPACE_Z[1])
                ctrl.set_target_pose(filtered_pos, orn_fixed)
                fine_adjust_active = True
            elif key == ord("s") and filtered_pos is not None:
                # Move down (decrease z)
                filtered_pos[2] = clamp(filtered_pos[2] - FINE_STEP, WORKSPACE_Z[0], WORKSPACE_Z[1])
                ctrl.set_target_pose(filtered_pos, orn_fixed)
                fine_adjust_active = True
            elif key == ord("a") and filtered_pos is not None:
                # Move left (decrease y)
                filtered_pos[1] = clamp(filtered_pos[1] - FINE_STEP, WORKSPACE_Y[0], WORKSPACE_Y[1])
                ctrl.set_target_pose(filtered_pos, orn_fixed)
                fine_adjust_active = True
            elif key == ord("d") and filtered_pos is not None:
                # Move right (increase y)
                filtered_pos[1] = clamp(filtered_pos[1] + FINE_STEP, WORKSPACE_Y[0], WORKSPACE_Y[1])
                ctrl.set_target_pose(filtered_pos, orn_fixed)
                fine_adjust_active = True
            else:
                fine_adjust_active = False
            cv2.imshow("Hand Tracking ", frame)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean shutdown
        print("Cleaning up...")
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
