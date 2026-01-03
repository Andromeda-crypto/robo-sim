# teleoperation.py
# Minimal hand → joint teleoperation (1 DOF prototype)

import cv2
import math
from simulation import RoboticArm
from hand_tracker import HandTracker


def compute_angle(a, b, c):
    bax = a[0] - b[0]
    bay = a[1] - b[1]
    bcx = c[0] - b[0]
    bcy = c[1] - b[1]

    dot = bax * bcx + bay * bcy
    mag1 = math.hypot(bax, bay)
    mag2 = math.hypot(bcx, bcy)

    if mag1 == 0 or mag2 == 0:
        return math.pi

    cosang = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.acos(cosang)


def map_hand_to_joint(hand_landmarks):
    p5 = (hand_landmarks[5].x, hand_landmarks[5].y)
    p6 = (hand_landmarks[6].x, hand_landmarks[6].y)
    p8 = (hand_landmarks[8].x, hand_landmarks[8].y)

    angle = compute_angle(p5, p6, p8)

    curl = (math.pi - angle) / (math.pi * 0.7)
    curl = max(0.0, min(1.0, curl))
    joint_angle = -1.5 + curl * 3.0
    return joint_angle


def main():
    urdf_file = "sample_urdfs/single_joint_arm.xml"
    robot = RoboticArm(urdf_file)
    tracker = HandTracker()
    cap = cv2.VideoCapture(0)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            hands = tracker.process_frame(frame)
            frame = tracker.draw_landmarks(frame, hands)
            cv2.imshow("Hand Tracking", frame)

            if hands:
                joint_angle = map_hand_to_joint(hands[0])
                robot.set_joint_position([0], [joint_angle])
                robot.step_simulation(steps=1)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        robot.disconnect()


if __name__ == "__main__":
    main()
