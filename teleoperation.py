# teleoperation.py - Operating our robotic arm using hand tracking


from simulation import RoboticArm
from hand_tracker import HandTracker
import cv2

def mao_landmarks_to_joint_positions(landmarks):
    # dummy implementation
    joint_positions = [0,1,2]
    """for lm in landmarks:
        joint_positions.append(lm.x) # example mappping
        return joint_positions[:2]"""
    

def main():
    urdf_path = "sample_urdfs/single_joint_arm.xml"
    robotic_arm = RoboticArm(urdf_path)
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break 

        landmarks = tracker.process_frame(frame)
        frame = tracker.draw_landmarks(frame, landmarks)
        cv2.imshow('hand Tracking', frame)

        if landmarks:
            joint_positions = mao_landmarks_to_joint_positions(landmarks[0])
            joint_indices = [0,1,2]
            robotic_arm.set_joint_position(joint_indices, joint_positions)
            robotic_arm.step_simulation(steps=1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()

    robotic_arm.disconnect()


if __name__ == "__main__":
    main()


