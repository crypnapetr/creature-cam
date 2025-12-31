"""
Creature Cam - Real-time creature face transformation
Main application loop
"""

import cv2
import time
import numpy as np
from pathlib import Path

from camera_capture import CameraCapture
from face_tracker import FaceTracker
from texture_engine import TextureEngine
from face_puppeteer import FacePuppeteer


class CreatureCam:
    """Main application for creature face transformations"""

    def __init__(self):
        """Initialize CreatureCam application"""
        # Get textures directory
        self.textures_dir = Path(__file__).parent.parent / "textures"

        # Initialize components
        self.camera = None
        self.tracker = None
        self.texture_engine = None
        self.puppeteer = None

        # Performance tracking
        self.frame_count = 0
        self.start_time = None
        self.fps = 0.0

        # State
        self.running = False
        self.show_landmarks = False
        self.show_stats = True
        self.use_puppeteer = False  # Toggle between texture and puppeteer mode

    def initialize(self) -> bool:
        """
        Initialize all components

        Returns:
            True if successful, False otherwise
        """
        print("Initializing Creature Cam...")

        # Initialize camera (camera_id=1 for real webcam, 0 is OBS Virtual Camera)
        self.camera = CameraCapture(camera_id=1)
        if not self.camera.start():
            print("Failed to initialize camera")
            return False

        # Initialize face tracker
        print("Loading face tracker...")
        self.tracker = FaceTracker()

        # Initialize texture engine
        print("Loading textures...")
        self.texture_engine = TextureEngine(self.textures_dir)

        # Check if we have any creatures loaded
        creatures = self.texture_engine.get_available_creatures()
        if not creatures:
            print("\nWARNING: No creature textures found!")
            print("Run 'python3 texture_engine.py' to create placeholder textures")
            print("Or add your own textures to:", self.textures_dir)
            return False

        print(f"\nAvailable creatures: {', '.join(creatures)}")
        print(f"Current creature: {self.texture_engine.current_creature}")

        # Try to initialize face puppeteering (optional)
        print("\nChecking for face puppeteer...")
        puppet_path = self.textures_dir / "trump-crab" / "skin.png"

        if puppet_path.exists():
            print(f"Found puppet image: {puppet_path.name}")
            try:
                self.puppeteer = FacePuppeteer(puppet_path)

                # Detect landmarks on puppet image
                print("Detecting landmarks on puppet...")
                puppet_landmarks = self.tracker.process_frame(self.puppeteer.puppet.image)

                if puppet_landmarks is not None:
                    self.puppeteer.set_puppet_landmarks(puppet_landmarks)
                    self.use_puppeteer = True  # Start in puppeteer mode
                    print("✅ Face puppeteering enabled!")
                    print("   Press 'p' to toggle between texture and puppeteer modes")
                else:
                    print("⚠️  No face detected in puppet image")
                    print("   Puppeteering disabled, using texture mode")
                    self.puppeteer = None

            except Exception as e:
                print(f"⚠️  Failed to initialize puppeteer: {e}")
                print("   Using texture mode only")
                self.puppeteer = None
        else:
            print("No puppet image found (texture mode only)")

        return True

    def run(self):
        """Run the main application loop"""
        if not self.initialize():
            print("Initialization failed")
            return

        print("\n=== Creature Cam Started ===")
        print("Controls:")
        print("  q - Quit")
        print("  l - Toggle landmark visualization")
        print("  s - Toggle stats")
        if self.puppeteer:
            print("  p - Toggle puppeteer/texture mode")
        print("  1-9 - Cycle through creatures")
        print("  + - Increase transformation intensity")
        print("  - - Decrease transformation intensity")
        print("==============================\n")

        self.running = True
        self.start_time = time.time()

        try:
            while self.running:
                self._process_frame()

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            self._cleanup()

    def _process_frame(self):
        """Process a single frame"""
        # Capture frame
        success, frame = self.camera.read_frame()
        if not success:
            self.running = False
            return

        self.frame_count += 1

        # Track face
        landmarks = self.tracker.process_frame(frame)

        # Apply transformation if face detected
        if landmarks is not None:
            # Use puppeteering or texture mode
            if self.use_puppeteer and self.puppeteer:
                frame = self.puppeteer.apply_puppeteering(frame, landmarks)
            else:
                frame = self.texture_engine.apply_creature_transformation(frame, landmarks)

            # Draw landmarks if enabled
            if self.show_landmarks:
                frame = self.tracker.draw_landmarks(
                    frame, landmarks,
                    draw_mesh=False,
                    draw_contours=True
                )

        # Display stats
        if self.show_stats:
            self._draw_stats(frame, landmarks is not None)

        # Show frame
        cv2.imshow("Creature Cam", frame)

        # Handle keyboard input
        self._handle_input()

    def _draw_stats(self, frame: np.ndarray, face_detected: bool):
        """Draw stats overlay on frame"""
        # Calculate FPS
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            self.fps = self.frame_count / elapsed

        # Prepare stats text
        mode = "PUPPETEER" if (self.use_puppeteer and self.puppeteer) else "TEXTURE"
        stats = [
            f"FPS: {self.fps:.1f}",
            f"Mode: {mode}",
            f"Creature: {self.texture_engine.current_creature or 'None'}",
            f"Intensity: {self.texture_engine.transformation_intensity:.0%}",
            f"Face: {'DETECTED' if face_detected else 'NOT DETECTED'}"
        ]

        # Draw background for stats
        y_offset = 10
        for i, stat in enumerate(stats):
            y = y_offset + i * 30
            # Background
            cv2.rectangle(frame, (5, y), (400, y + 25), (0, 0, 0), -1)
            # Text
            color = (0, 255, 0) if (i == 3 and face_detected) else (255, 255, 255)
            cv2.putText(frame, stat, (10, y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def _handle_input(self):
        """Handle keyboard input"""
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            # Quit
            self.running = False

        elif key == ord('l'):
            # Toggle landmarks
            self.show_landmarks = not self.show_landmarks
            print(f"Landmark visualization: {self.show_landmarks}")

        elif key == ord('s'):
            # Toggle stats
            self.show_stats = not self.show_stats
            print(f"Stats display: {self.show_stats}")

        elif key == ord('p'):
            # Toggle puppeteer mode
            if self.puppeteer:
                self.use_puppeteer = not self.use_puppeteer
                mode = "PUPPETEER" if self.use_puppeteer else "TEXTURE"
                print(f"Mode switched to: {mode}")
            else:
                print("Puppeteer not available")

        elif key == ord('+') or key == ord('='):
            # Increase intensity
            new_intensity = min(1.0, self.texture_engine.transformation_intensity + 0.1)
            self.texture_engine.set_intensity(new_intensity)
            print(f"Intensity: {new_intensity:.0%}")

        elif key == ord('-') or key == ord('_'):
            # Decrease intensity
            new_intensity = max(0.0, self.texture_engine.transformation_intensity - 0.1)
            self.texture_engine.set_intensity(new_intensity)
            print(f"Intensity: {new_intensity:.0%}")

        elif ord('1') <= key <= ord('9'):
            # Select creature by number
            creatures = self.texture_engine.get_available_creatures()
            index = key - ord('1')
            if index < len(creatures):
                self.texture_engine.set_creature(creatures[index])

    def _cleanup(self):
        """Clean up resources"""
        print("\nShutting down...")

        if self.tracker:
            self.tracker.release()

        if self.camera:
            self.camera.release()

        cv2.destroyAllWindows()

        # Print final stats
        if self.start_time:
            elapsed = time.time() - self.start_time
            avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
            print(f"\nSession stats:")
            print(f"  Total frames: {self.frame_count}")
            print(f"  Duration: {elapsed:.1f}s")
            print(f"  Average FPS: {avg_fps:.1f}")

        print("Creature Cam terminated")


def main():
    """Entry point"""
    app = CreatureCam()
    app.run()


if __name__ == "__main__":
    main()
