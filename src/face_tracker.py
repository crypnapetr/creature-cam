"""
Face tracking module using MediaPipe Face Landmarker
Optimized for real-time performance on Apple Silicon
"""

import cv2
import numpy as np
import urllib.request
import os
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


@dataclass
class FaceLandmarks:
    """Face landmark data"""
    landmarks: np.ndarray  # (478, 3) array of normalized coordinates
    image_landmarks: np.ndarray  # (478, 2) array of pixel coordinates
    face_oval_indices: List[int]  # Indices for face oval contour
    left_eye_indices: List[int]
    right_eye_indices: List[int]
    lips_indices: List[int]
    nose_indices: List[int]


class FaceTracker:
    """Real-time face tracking using MediaPipe Face Landmarker"""

    # MediaPipe face mesh landmark indices for key features
    FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

    LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158,
                159, 160, 161, 246]

    RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388,
                 387, 386, 385, 384, 398]

    LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324,
            318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13,
            312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88]

    NOSE = [1, 2, 98, 327, 168]

    # MediaPipe model URL
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

    def __init__(self,
                 max_num_faces: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize face tracker

        Args:
            max_num_faces: Maximum number of faces to track
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for face tracking
        """
        # Download model if needed
        model_path = self._get_model_path()

        # Create face landmarker options
        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            running_mode=vision.RunningMode.VIDEO
        )

        # Create face landmarker
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.frame_timestamp_ms = 0

    def _get_model_path(self) -> Path:
        """Download and cache the face landmarker model"""
        # Model cache directory
        cache_dir = Path(__file__).parent.parent / "models"
        cache_dir.mkdir(exist_ok=True)

        model_path = cache_dir / "face_landmarker.task"

        # Download if not exists
        if not model_path.exists():
            print(f"Downloading face landmarker model...")
            print(f"This is a one-time download (~10 MB)")
            try:
                urllib.request.urlretrieve(self.MODEL_URL, model_path)
                print(f"Model downloaded to: {model_path}")
            except Exception as e:
                print(f"Error downloading model: {e}")
                raise

        return model_path

    def process_frame(self, frame: np.ndarray) -> Optional[FaceLandmarks]:
        """
        Process a frame and extract face landmarks

        Args:
            frame: BGR image from camera

        Returns:
            FaceLandmarks object if face detected, None otherwise
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Increment timestamp (in milliseconds)
        self.frame_timestamp_ms += 33  # ~30 FPS

        # Detect face landmarks
        result = self.landmarker.detect_for_video(mp_image, self.frame_timestamp_ms)

        if not result.face_landmarks or len(result.face_landmarks) == 0:
            return None

        # Get first face (we only track one face)
        face_landmarks = result.face_landmarks[0]

        # Convert to numpy arrays
        h, w = frame.shape[:2]
        landmarks_normalized = np.array([
            [lm.x, lm.y, lm.z] for lm in face_landmarks
        ])

        landmarks_pixel = np.array([
            [int(lm.x * w), int(lm.y * h)] for lm in face_landmarks
        ])

        return FaceLandmarks(
            landmarks=landmarks_normalized,
            image_landmarks=landmarks_pixel,
            face_oval_indices=self.FACE_OVAL,
            left_eye_indices=self.LEFT_EYE,
            right_eye_indices=self.RIGHT_EYE,
            lips_indices=self.LIPS,
            nose_indices=self.NOSE
        )

    def draw_landmarks(self, frame: np.ndarray, landmarks: FaceLandmarks,
                      draw_mesh: bool = True, draw_contours: bool = False) -> np.ndarray:
        """
        Draw face landmarks on frame

        Args:
            frame: BGR image
            landmarks: FaceLandmarks object
            draw_mesh: Whether to draw full face mesh
            draw_contours: Whether to draw contours only

        Returns:
            Frame with landmarks drawn
        """
        output_frame = frame.copy()

        if draw_mesh:
            # Draw all landmarks
            for point in landmarks.image_landmarks:
                cv2.circle(output_frame, tuple(point), 1, (0, 255, 0), -1)

        if draw_contours:
            # Draw face oval
            oval_points = landmarks.image_landmarks[landmarks.face_oval_indices]
            cv2.polylines(output_frame, [oval_points], True, (255, 0, 0), 2)

            # Draw eyes
            left_eye = landmarks.image_landmarks[landmarks.left_eye_indices]
            right_eye = landmarks.image_landmarks[landmarks.right_eye_indices]
            cv2.polylines(output_frame, [left_eye], True, (0, 255, 255), 2)
            cv2.polylines(output_frame, [right_eye], True, (0, 255, 255), 2)

            # Draw lips
            lips = landmarks.image_landmarks[landmarks.lips_indices]
            cv2.polylines(output_frame, [lips], True, (255, 255, 0), 2)

        return output_frame

    def get_face_mask(self, frame: np.ndarray, landmarks: FaceLandmarks) -> np.ndarray:
        """
        Create a binary mask of the face region

        Args:
            frame: BGR image
            landmarks: FaceLandmarks object

        Returns:
            Binary mask (255 for face, 0 for background)
        """
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        face_points = landmarks.image_landmarks[landmarks.face_oval_indices]
        cv2.fillPoly(mask, [face_points], 255)
        return mask

    def release(self):
        """Release MediaPipe resources"""
        self.landmarker.close()


def test_face_tracking():
    """Test face tracking functionality"""
    from camera_capture import CameraCapture
    import time

    print("Testing face tracking...")
    print("Press 'q' to quit")
    print("Press 'm' to toggle mesh visualization")
    print("Press 'c' to toggle contour visualization")

    with CameraCapture() as camera:
        if not camera.is_running:
            print("Failed to initialize camera")
            return

        tracker = FaceTracker()

        frame_count = 0
        face_detected_count = 0
        start_time = time.time()

        draw_mesh = False
        draw_contours = True

        try:
            while True:
                success, frame = camera.read_frame()

                if not success:
                    break

                frame_count += 1

                # Process frame
                landmarks = tracker.process_frame(frame)

                if landmarks is not None:
                    face_detected_count += 1

                    # Draw landmarks
                    frame = tracker.draw_landmarks(
                        frame, landmarks,
                        draw_mesh=draw_mesh,
                        draw_contours=draw_contours
                    )

                    # Show face detected indicator
                    cv2.putText(frame, "FACE DETECTED", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Calculate FPS
                elapsed = time.time() - start_time
                if elapsed > 0:
                    fps = frame_count / elapsed
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("Face Tracking Test", frame)

                # Handle keypresses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('m'):
                    draw_mesh = not draw_mesh
                    print(f"Mesh visualization: {draw_mesh}")
                elif key == ord('c'):
                    draw_contours = not draw_contours
                    print(f"Contour visualization: {draw_contours}")

        finally:
            tracker.release()
            cv2.destroyAllWindows()

            # Print final stats
            elapsed = time.time() - start_time
            avg_fps = frame_count / elapsed if elapsed > 0 else 0
            detection_rate = (face_detected_count / frame_count * 100) if frame_count > 0 else 0

            print(f"\nFace tracking test complete:")
            print(f"  Frames: {frame_count}")
            print(f"  Face detected: {face_detected_count} ({detection_rate:.1f}%)")
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Average FPS: {avg_fps:.1f}")


if __name__ == "__main__":
    test_face_tracking()
