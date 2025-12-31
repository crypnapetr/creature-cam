"""
Camera capture module for creature-cam
Handles webcam input with optimizations for Apple Silicon
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class CameraCapture:
    """Handles webcam capture with performance optimizations"""

    def __init__(self, camera_id: int = 0, width: int = 1280, height: int = 720):
        """
        Initialize camera capture

        Args:
            camera_id: Camera device ID (0 for default webcam)
            width: Capture width (default 720p for balance of quality/performance)
            height: Capture height
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False

    def start(self) -> bool:
        """
        Start camera capture

        Returns:
            True if successful, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_id)

            if not self.cap.isOpened():
                print(f"Error: Could not open camera {self.camera_id}")
                return False

            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            # Try to set 30 FPS (may not work on all cameras)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            # Verify settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

            print(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps} FPS")

            self.is_running = True
            return True

        except Exception as e:
            print(f"Error starting camera: {e}")
            return False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the camera

        Returns:
            Tuple of (success, frame)
            - success: True if frame was read successfully
            - frame: BGR image as numpy array, or None if failed
        """
        if not self.is_running or self.cap is None:
            return False, None

        ret, frame = self.cap.read()

        if not ret:
            print("Warning: Failed to read frame")
            return False, None

        return True, frame

    def release(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
            self.is_running = False
            print("Camera released")

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()


def test_camera():
    """Test camera capture functionality"""
    print("Testing camera capture...")
    print("Press 'q' to quit")

    with CameraCapture() as camera:
        if not camera.is_running:
            print("Failed to initialize camera")
            return

        frame_count = 0
        import time
        start_time = time.time()

        while True:
            success, frame = camera.read_frame()

            if not success:
                break

            frame_count += 1

            # Calculate FPS
            elapsed = time.time() - start_time
            if elapsed > 0:
                fps = frame_count / elapsed
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Camera Test", frame)

            # Quit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

        # Print final stats
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"\nCapture test complete:")
        print(f"  Frames: {frame_count}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Average FPS: {avg_fps:.1f}")


if __name__ == "__main__":
    test_camera()
