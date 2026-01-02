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
from vtuber_renderer import VTuberRenderer
from vtuber_character import CharacterAsset
from background_manager import BackgroundManager


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
        self.vtuber_renderer = None
        self.background_manager = None

        # Performance tracking
        self.frame_count = 0
        self.start_time = None
        self.fps = 0.0

        # State
        self.running = False
        self.show_landmarks = False
        self.show_stats = True
        self.use_puppeteer = False  # Toggle between texture and puppeteer mode
        self.use_vtuber = False  # Toggle for VTuber mode (full character)
        self.puppeteer_black_bg = False  # Black background for puppeteer mode

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
        # Use character.png if available, fallback to skin.png
        puppet_path = self.textures_dir / "trump-crab" / "character.png"
        if not puppet_path.exists():
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
                    print("✅ Face puppeteering enabled!")
                    print("   Press 'p' to toggle puppeteer mode")
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

        # Try to initialize VTuber mode (optional but preferred)
        print("\nChecking for VTuber mode...")
        backgrounds_dir = Path(__file__).parent.parent / "backgrounds"

        try:
            # Initialize background manager
            self.background_manager = BackgroundManager(backgrounds_dir)

            # Initialize character
            character_dir = self.textures_dir / "trump-crab"
            character_path = character_dir / "character.png"

            if character_path.exists() or (character_dir / "skin.png").exists():
                print(f"Loading character from: {character_dir.name}")
                character = CharacterAsset(character_dir)

                # Initialize character (auto-detect face region if needed)
                if character.initialize(auto_detect=False):
                    # Initialize VTuber renderer
                    self.vtuber_renderer = VTuberRenderer()
                    self.vtuber_renderer.set_background_manager(self.background_manager)
                    self.vtuber_renderer.set_character(character)
                    self.vtuber_renderer.set_face_tracker(self.tracker)

                    # Start in VTuber mode by default
                    self.use_vtuber = True
                    self.use_puppeteer = False

                    print("✅ VTuber mode enabled!")
                    print("   Press 'v' to toggle VTuber mode")
                    print("   Press 'p' to toggle puppeteer mode")
                    print("   🎭 Full character transformation with static background")
                else:
                    print("⚠️  Failed to initialize character")
                    print("   VTuber mode disabled")
            else:
                print(f"No character image found in {character_dir}")
                print("   VTuber mode disabled")

        except Exception as e:
            print(f"⚠️  Failed to initialize VTuber mode: {e}")
            print("   Using puppeteer/texture mode")
            self.vtuber_renderer = None

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
        if self.vtuber_renderer:
            print("  v - Toggle VTuber mode (full character)")
            print("  r - Reload character (after changing character.png)")
        if self.puppeteer:
            print("  p - Toggle puppeteer mode (face swap)")
            print("  b - Toggle black background (puppeteer mode)")
        print("  1-9 - Cycle through creatures (texture mode)")
        print("  + - Increase transformation intensity (texture mode)")
        print("  - - Decrease transformation intensity (texture mode)")
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

        # Apply transformation based on current mode
        if self.use_vtuber and self.vtuber_renderer:
            # VTuber mode: Full character replacement with static background
            frame = self.vtuber_renderer.render_frame(frame, landmarks)

            # Draw landmarks on VTuber output if enabled
            if self.show_landmarks and landmarks is not None:
                frame = self.tracker.draw_landmarks(
                    frame, landmarks,
                    draw_mesh=False,
                    draw_contours=True
                )

        elif landmarks is not None:
            # Puppeteer or texture mode (only if face detected)
            if self.use_puppeteer and self.puppeteer:
                # Puppeteer mode: Face swap on webcam
                # Optionally use black background
                if self.puppeteer_black_bg:
                    black_frame = np.zeros_like(frame)
                    frame = self.puppeteer.apply_puppeteering(black_frame, landmarks)
                else:
                    frame = self.puppeteer.apply_puppeteering(frame, landmarks)
            else:
                # Texture mode: Texture overlay
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
        if self.use_vtuber and self.vtuber_renderer:
            mode = "VTUBER"
        elif self.use_puppeteer and self.puppeteer:
            mode = "PUPPETEER"
        else:
            mode = "TEXTURE"

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

        elif key == ord('r'):
            # Reload character (for when you change character.png)
            print("\n🔄 Reloading character...")
            try:
                character_dir = self.textures_dir / "trump-crab"

                # Reload VTuber character
                if self.vtuber_renderer:
                    character = CharacterAsset(character_dir)
                    if character.initialize(auto_detect=True):
                        self.vtuber_renderer.set_character(character)
                        print("✅ VTuber character reloaded!")
                    else:
                        print("❌ Failed to reload VTuber character")

                # Reload Puppeteer character
                if self.puppeteer:
                    puppet_path = character_dir / "character.png"
                    if puppet_path.exists():
                        print("🔄 Reloading puppeteer...")
                        # Create new puppeteer
                        new_puppeteer = FacePuppeteer(puppet_path)

                        # Detect landmarks on new puppet image
                        puppet_landmarks = self.tracker.process_frame(new_puppeteer.puppet.image)

                        if puppet_landmarks is not None:
                            new_puppeteer.set_puppet_landmarks(puppet_landmarks)
                            self.puppeteer = new_puppeteer
                            print("✅ Puppeteer reloaded!")
                        else:
                            print("⚠️  No face detected in new character image")
                    else:
                        print("⚠️  character.png not found for puppeteer")

                print("✅ Reload complete!")

            except Exception as e:
                print(f"❌ Error reloading: {e}")

        elif key == ord('v'):
            # Toggle VTuber mode
            if self.vtuber_renderer:
                self.use_vtuber = not self.use_vtuber
                if self.use_vtuber:
                    # When enabling VTuber, disable puppeteer
                    self.use_puppeteer = False
                    print("Mode switched to: VTUBER (full character)")
                else:
                    print("Mode switched to: TEXTURE")
            else:
                print("VTuber mode not available")

        elif key == ord('b'):
            # Toggle black background in puppeteer mode
            if self.puppeteer:
                self.puppeteer_black_bg = not self.puppeteer_black_bg
                status = "ON" if self.puppeteer_black_bg else "OFF"
                print(f"Puppeteer black background: {status}")
            else:
                print("Puppeteer not available")

        elif key == ord('p'):
            # Toggle puppeteer mode
            if self.puppeteer:
                if self.use_vtuber:
                    # If VTuber is active, switch to puppeteer
                    self.use_vtuber = False
                    self.use_puppeteer = True
                    print("Mode switched to: PUPPETEER (face swap)")
                else:
                    # Toggle between puppeteer and texture
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
