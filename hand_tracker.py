import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions


class HandTracker:
    def __init__(
        self,
        model_path="hand_landmarker.task",
        num_hands=6,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        base_options = BaseOptions(model_asset_path=model_path)

        options = HandLandmarkerOptions(
        base_options=base_options,
        num_hands=num_hands,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        running_mode=vision.RunningMode.IMAGE,
)


        self.landmarker = HandLandmarker.create_from_options(options)

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = self.landmarker.detect(mp_image)
        self.landmarks = detection_result.hand_landmarks
        return self.landmarks

    def draw_landmarks(self, frame, hand_landmarks):
        if not hand_landmarks:
            return frame

        h, w, _ = frame.shape

        for hand in hand_landmarks:
            for landmark in hand:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        return frame

    def close(self):
        self.landmarker.close()


def main():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = tracker.process_frame(frame)
        frame = tracker.draw_landmarks(frame, landmarks)

        cv2.imshow("Hand Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



     



