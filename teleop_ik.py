import os
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import time
import cv2
import pybullet as p
from arm_controller import ArmController
from hand_tracker import HandTracker


WORKSPACE = {
    "x": (0.30, 0.70),
    "y": (-0.30, 0.30),
    "z": (0.20, 0.60),
}

ALPHA = 0.85
LOST_HOLD_SEC = 0.2


def lerp(a, b, t):
    return a + (b - a) * t


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def map_hand_to_pose(hand):
    wrist = hand[0]
    mid_mcp = hand[9]

    x = lerp(WORKSPACE["x"][0], WORKSPACE["x"][1], wrist.x)

    y_img = 1.0 - wrist.y
    z = lerp(WORKSPACE["z"][0], WORKSPACE["z"][1], y_img)

    dx = mid_mcp.x - wrist.x
    dy = mid_mcp.y - wrist.y
    hand_size = (dx * dx + dy * dy) ** 0.5

    size_min, size_max = 0.05, 0.20
    t = (hand_size - size_min) / (size_max - size_min)
    t = clamp(t, 0.0, 1.0)

    y = lerp(WORKSPACE["y"][0], WORKSPACE["y"][1], t)

    orn = p.getQuaternionFromEuler([0, 3.14159, 0])
    return [x, y, z], orn


def ema(prev, new, alpha):
    if prev is None:
        return new
    return [alpha * p0 + (1 - alpha) * n0 for p0, n0 in zip(prev, new)]


def main():
    ctrl = ArmController(gui=True)
    tracker = HandTracker()
    cap = cv2.VideoCapture(0)

    filtered_pos = None
    last_seen = 0.0
    orn_fixed = p.getQuaternionFromEuler([0, 3.14159, 0])

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            hands = tracker.process_frame(frame)
            frame = tracker.draw_landmarks(frame, hands)
            cv2.imshow("Hand Tracking", frame)

            now = time.time()

            if hands:
                pos, orn = map_hand_to_pose(hands[0])
                filtered_pos = ema(filtered_pos, pos, ALPHA)
                last_seen = now
                ctrl.set_target_pose(filtered_pos, orn)
            else:
                if filtered_pos is not None and (now - last_seen) < LOST_HOLD_SEC:
                    ctrl.set_target_pose(filtered_pos, orn_fixed)

            ctrl.step()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        ctrl.disconnect()


if __name__ == "__main__":
    main()
