"""
Utility to list all available cameras
Helps identify which camera ID to use
"""

import cv2


def list_cameras(max_cameras=10):
    """List all available cameras"""
    print("Scanning for cameras...\n")

    available_cameras = []

    for camera_id in range(max_cameras):
        cap = cv2.VideoCapture(camera_id)

        if cap.isOpened():
            # Get camera info
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Try to read a frame to verify it works
            ret, frame = cap.read()

            if ret:
                print(f"✅ Camera {camera_id}:")
                print(f"   Resolution: {width}x{height}")
                print(f"   FPS: {fps}")

                # Try to get camera name (not always available)
                backend = cap.getBackendName()
                print(f"   Backend: {backend}")

                available_cameras.append(camera_id)
                print()

            cap.release()

    if not available_cameras:
        print("❌ No cameras found!")
    else:
        print(f"\nFound {len(available_cameras)} camera(s): {available_cameras}")
        print("\nTo use a specific camera, edit src/main.py:")
        print("Change line 56 from:")
        print("  self.camera = CameraCapture()")
        print("To:")
        print(f"  self.camera = CameraCapture(camera_id=X)")
        print("\nWhere X is the camera ID you want to use.")

        if 0 in available_cameras and len(available_cameras) > 1:
            print("\n⚠️  Camera 0 might be OBS Virtual Camera")
            print(f"   Try using camera_id={available_cameras[1]} for your real webcam")

    return available_cameras


if __name__ == "__main__":
    print("=" * 60)
    print("Camera Detection Utility")
    print("=" * 60 + "\n")

    cameras = list_cameras()

    print("\n" + "=" * 60)
