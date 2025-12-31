"""
VTuber Renderer - Core rendering pipeline for full character transformation
Implements multi-layer compositing: background → body → animated face
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

# Handle imports whether running as script or module
try:
    from vtuber_character import CharacterAsset
    from background_manager import BackgroundManager
    from face_tracker import FaceLandmarks, FaceTracker
    from face_puppeteer import FacePuppeteer
except ImportError:
    from src.vtuber_character import CharacterAsset
    from src.background_manager import BackgroundManager
    from src.face_tracker import FaceLandmarks, FaceTracker
    from src.face_puppeteer import FacePuppeteer


class CharacterFaceWarper:
    """Warps character face region to match user expressions"""

    def __init__(self):
        """Initialize face warper"""
        self.puppeteer: Optional[FacePuppeteer] = None

    def set_character_face(self, character_image: np.ndarray, character_landmarks: FaceLandmarks):
        """
        Set the character face to warp

        Args:
            character_image: Full character image
            character_landmarks: Detected landmarks on character
        """
        # Create a temporary puppeteer for the character face
        # We'll use the existing FacePuppeteer infrastructure
        # but operate on the character's face
        print(f"ℹ️  Setting up character face warper with {len(character_landmarks.image_landmarks)} landmarks")

    def warp_character_face(self,
                           character_image: np.ndarray,
                           character_landmarks: FaceLandmarks,
                           user_landmarks: FaceLandmarks,
                           output_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Warp character's face to match user's facial expression

        Args:
            character_image: Character image
            character_landmarks: Character's face landmarks
            user_landmarks: User's real-time face landmarks
            output_shape: Output image shape (H, W, C)

        Returns:
            Warped face image
        """
        # For now, use the existing FacePuppeteer to warp the character face
        # This will be enhanced later for better performance
        warped = np.zeros(output_shape, dtype=np.uint8)

        # Use FacePuppeteer's warping logic
        from face_puppeteer import FacePuppeteer

        # Create temporary puppeteer
        temp_puppeteer = FacePuppeteer.__new__(FacePuppeteer)
        temp_puppeteer.puppet_landmarks_set = True
        temp_puppeteer.puppet = type('obj', (object,), {
            'image': character_image,
            'landmarks': character_landmarks,
            'triangulation': None
        })()

        # Compute triangulation
        temp_puppeteer.puppet.set_landmarks = lambda l: None
        from face_puppeteer import FacePuppet
        puppet = FacePuppet.__new__(FacePuppet)
        puppet.image = character_image
        puppet.landmarks = character_landmarks
        puppet._compute_triangulation()

        temp_puppeteer.puppet.triangulation = puppet.triangulation

        # Warp face
        warped = temp_puppeteer._warp_face(
            character_image,
            character_landmarks.image_landmarks,
            user_landmarks.image_landmarks,
            puppet.triangulation,
            output_shape
        )

        return warped


class VTuberRenderer:
    """Renders full VTuber character with animated face on static background"""

    def __init__(self, output_size: Tuple[int, int] = (1280, 720)):
        """
        Initialize VTuber renderer

        Args:
            output_size: Output frame size (width, height)
        """
        self.output_size = output_size
        self.character: Optional[CharacterAsset] = None
        self.background_manager: Optional[BackgroundManager] = None
        self.face_warper = CharacterFaceWarper()
        self.face_tracker: Optional[FaceTracker] = None

        # Character positioning (centered by default)
        self.character_position = (0, 0)  # (x, y) offset
        self.character_scale = 1.0

        # Performance cache
        self._cached_background: Optional[np.ndarray] = None
        self._cached_character_body: Optional[np.ndarray] = None

        print("✅ VTuber renderer initialized")

    def set_background_manager(self, background_manager: BackgroundManager):
        """
        Set background manager

        Args:
            background_manager: BackgroundManager instance
        """
        self.background_manager = background_manager
        self._cached_background = None  # Invalidate cache
        print("✅ Background manager configured")

    def set_character(self, character: CharacterAsset):
        """
        Set character asset

        Args:
            character: CharacterAsset instance
        """
        self.character = character

        # Verify character is initialized
        if character.get_face_landmarks() is None:
            print("⚠️  Character landmarks not computed")

        # Setup face warper
        self.face_warper.set_character_face(
            character.get_character_image(),
            character.get_face_landmarks()
        )

        # Invalidate caches
        self._cached_character_body = None

        print(f"✅ Character configured: {character.character_dir.name}")

    def set_face_tracker(self, tracker: FaceTracker):
        """Set face tracker instance"""
        self.face_tracker = tracker

    def render_frame(self, webcam_frame: np.ndarray, user_landmarks: Optional[FaceLandmarks] = None) -> np.ndarray:
        """
        Main rendering pipeline

        Args:
            webcam_frame: Current webcam frame (for landmark detection if needed)
            user_landmarks: Pre-detected user landmarks (optional)

        Returns:
            Rendered VTuber frame
        """
        if self.character is None or self.background_manager is None:
            return webcam_frame

        # Start with background layer
        output = self._get_background()

        # Composite static character body
        output = self._composite_body(output)

        # Composite animated face if user detected
        if user_landmarks is not None:
            output = self._composite_animated_face(output, webcam_frame, user_landmarks)

        return output

    def _get_background(self) -> np.ndarray:
        """Get background layer (cached)"""
        if self._cached_background is None:
            bg = self.background_manager.get_current()

            # Resize to output size if needed
            if bg.shape[:2] != (self.output_size[1], self.output_size[0]):
                bg = cv2.resize(bg, self.output_size)

            self._cached_background = bg

        return self._cached_background.copy()

    def _composite_body(self, background: np.ndarray) -> np.ndarray:
        """
        Composite static character body onto background

        Args:
            background: Background image

        Returns:
            Background with character body composited
        """
        if self.character is None:
            return background

        # Get character image and body mask
        character_img = self.character.get_character_image()
        body_mask = self.character.get_body_mask()

        if character_img is None or body_mask is None:
            return background

        # Resize character to fit output (if needed)
        char_h, char_w = character_img.shape[:2]
        out_h, out_w = self.output_size[1], self.output_size[0]

        # Calculate scale to fit character in output
        scale_w = out_w / char_w
        scale_h = out_h / char_h
        scale = min(scale_w, scale_h) * self.character_scale

        new_w = int(char_w * scale)
        new_h = int(char_h * scale)

        # Resize character and mask
        character_resized = cv2.resize(character_img, (new_w, new_h))
        body_mask_resized = cv2.resize(body_mask, (new_w, new_h))

        # Calculate position (centered by default + offset)
        x = (out_w - new_w) // 2 + self.character_position[0]
        y = (out_h - new_h) // 2 + self.character_position[1]

        # Ensure within bounds
        x = max(0, min(x, out_w - new_w))
        y = max(0, min(y, out_h - new_h))

        # Composite using alpha blending
        output = background.copy()
        output = self._alpha_blend(
            output,
            character_resized,
            body_mask_resized,
            (x, y)
        )

        return output

    def _composite_animated_face(self,
                                 base: np.ndarray,
                                 webcam_frame: np.ndarray,
                                 user_landmarks: FaceLandmarks) -> np.ndarray:
        """
        Composite animated character face

        Args:
            base: Base image (background + body)
            webcam_frame: Current webcam frame
            user_landmarks: User's face landmarks

        Returns:
            Base with animated face composited
        """
        if self.character is None or self.face_tracker is None:
            return base

        # Get character face landmarks
        character_landmarks = self.character.get_face_landmarks()
        if character_landmarks is None:
            return base

        # Warp character face to match user expression
        character_img = self.character.get_character_image()

        # Convert character image to BGR if it has alpha channel
        if character_img.shape[2] == 4:
            character_img_bgr = character_img[:, :, :3]
        else:
            character_img_bgr = character_img

        warped_face = self.face_warper.warp_character_face(
            character_img_bgr,
            character_landmarks,
            user_landmarks,
            webcam_frame.shape
        )

        # Create full head mask for complete coverage
        full_head_mask = self.face_tracker.create_full_head_mask(
            webcam_frame,
            user_landmarks
        )

        # Resize warped face and mask to match output size
        char_h, char_w = character_img.shape[:2]
        out_h, out_w = self.output_size[1], self.output_size[0]

        scale_w = out_w / char_w
        scale_h = out_h / char_h
        scale = min(scale_w, scale_h) * self.character_scale

        # Calculate position for face overlay (matching body position)
        x = (out_w - int(char_w * scale)) // 2 + self.character_position[0]
        y = (out_h - int(char_h * scale)) // 2 + self.character_position[1]

        # Composite warped face onto base
        output = self._seamless_blend(
            base,
            warped_face,
            full_head_mask,
            user_landmarks,
            offset=(x, y)
        )

        return output

    def _alpha_blend(self,
                    background: np.ndarray,
                    foreground: np.ndarray,
                    mask: np.ndarray,
                    position: Tuple[int, int]) -> np.ndarray:
        """
        Alpha blend foreground onto background at position

        Args:
            background: Background image
            foreground: Foreground image (RGBA or BGR)
            mask: Blending mask (grayscale)
            position: (x, y) position to place foreground

        Returns:
            Blended image
        """
        x, y = position
        fg_h, fg_w = foreground.shape[:2]
        bg_h, bg_w = background.shape[:2]

        # Ensure foreground fits in background
        if x + fg_w > bg_w or y + fg_h > bg_h:
            return background

        # Extract region of interest
        roi = background[y:y+fg_h, x:x+fg_w]

        # Convert foreground to BGR if RGBA
        if foreground.shape[2] == 4:
            fg_bgr = foreground[:, :, :3]
            alpha_channel = foreground[:, :, 3:4] / 255.0
        else:
            fg_bgr = foreground
            alpha_channel = np.ones((fg_h, fg_w, 1), dtype=np.float32)

        # Combine with mask
        mask_3ch = cv2.merge([mask, mask, mask]) / 255.0
        combined_alpha = alpha_channel * mask_3ch

        # Blend
        blended = (fg_bgr * combined_alpha + roi * (1 - combined_alpha)).astype(np.uint8)

        # Place back in background
        output = background.copy()
        output[y:y+fg_h, x:x+fg_w] = blended

        return output

    def _seamless_blend(self,
                       background: np.ndarray,
                       foreground: np.ndarray,
                       mask: np.ndarray,
                       landmarks: FaceLandmarks,
                       offset: Tuple[int, int] = (0, 0)) -> np.ndarray:
        """
        Seamlessly blend foreground onto background using mask

        Args:
            background: Background image
            foreground: Foreground image
            mask: Blending mask
            landmarks: Face landmarks for center point
            offset: (x, y) offset for positioning

        Returns:
            Blended image
        """
        # Resize foreground and mask to match background if needed
        if foreground.shape[:2] != background.shape[:2]:
            foreground = cv2.resize(foreground, (background.shape[1], background.shape[0]))
            mask = cv2.resize(mask, (background.shape[1], background.shape[0]))

        # Get center of face for seamless clone
        face_points = landmarks.image_landmarks[landmarks.face_oval_indices]
        center = tuple(np.mean(face_points, axis=0).astype(int))

        # Apply offset
        center = (center[0] + offset[0], center[1] + offset[1])

        # Ensure center is within bounds
        center = (
            max(0, min(center[0], background.shape[1] - 1)),
            max(0, min(center[1], background.shape[0] - 1))
        )

        try:
            # Use seamless clone for natural blending
            output = cv2.seamlessClone(
                foreground,
                background,
                mask,
                center,
                cv2.NORMAL_CLONE
            )
        except cv2.error as e:
            # Fallback to simple alpha blending
            print(f"⚠️  Seamless clone failed: {e}, using alpha blend")
            mask_3ch = cv2.merge([mask, mask, mask]) / 255.0
            output = (foreground * mask_3ch + background * (1 - mask_3ch)).astype(np.uint8)

        return output

    def set_character_position(self, x: int, y: int):
        """Set character position offset"""
        self.character_position = (x, y)
        self._cached_character_body = None  # Invalidate cache

    def set_character_scale(self, scale: float):
        """Set character scale (1.0 = normal size)"""
        self.character_scale = max(0.1, min(scale, 3.0))  # Limit 0.1x to 3.0x
        self._cached_character_body = None  # Invalidate cache


def test_vtuber_renderer():
    """Test VTuber renderer with character and webcam"""
    from camera_capture import CameraCapture
    import time

    print("="*70)
    print("VTuber Renderer Test")
    print("="*70)

    # Setup paths
    textures_dir = Path(__file__).parent.parent / "textures"
    backgrounds_dir = Path(__file__).parent.parent / "backgrounds"

    # Initialize components
    print("\nInitializing components...")

    # Camera
    camera = CameraCapture(camera_id=1)
    if not camera.start():
        print("❌ Failed to start camera")
        return

    # Face tracker
    tracker = FaceTracker()

    # Background manager
    bg_manager = BackgroundManager(backgrounds_dir)

    # Character
    character_dir = textures_dir / "trump-crab"
    character = CharacterAsset(character_dir)

    if not character.initialize(auto_detect=False):
        print("❌ Failed to initialize character")
        camera.release()
        tracker.release()
        return

    # VTuber renderer
    renderer = VTuberRenderer()
    renderer.set_background_manager(bg_manager)
    renderer.set_character(character)
    renderer.set_face_tracker(tracker)

    print("\n" + "="*70)
    print("Controls:")
    print("  q - Quit")
    print("  ← → - Move character left/right")
    print("  ↑ ↓ - Move character up/down")
    print("  + - - Scale character")
    print("="*70 + "\n")

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            success, frame = camera.read_frame()
            if not success:
                break

            frame_count += 1

            # Detect user face
            user_landmarks = tracker.process_frame(frame)

            # Render VTuber frame
            output = renderer.render_frame(frame, user_landmarks)

            # Show FPS
            elapsed = time.time() - start_time
            if elapsed > 0:
                fps = frame_count / elapsed
                cv2.putText(output, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Show face detection status
            status = "FACE DETECTED" if user_landmarks else "NO FACE"
            color = (0, 255, 0) if user_landmarks else (0, 0, 255)
            cv2.putText(output, status, (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow("VTuber Renderer", output)

            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == 81:  # Left arrow
                renderer.set_character_position(renderer.character_position[0] - 10,
                                               renderer.character_position[1])
            elif key == 83:  # Right arrow
                renderer.set_character_position(renderer.character_position[0] + 10,
                                               renderer.character_position[1])
            elif key == 82:  # Up arrow
                renderer.set_character_position(renderer.character_position[0],
                                               renderer.character_position[1] - 10)
            elif key == 84:  # Down arrow
                renderer.set_character_position(renderer.character_position[0],
                                               renderer.character_position[1] + 10)
            elif key == ord('+') or key == ord('='):
                renderer.set_character_scale(renderer.character_scale + 0.1)
            elif key == ord('-') or key == ord('_'):
                renderer.set_character_scale(renderer.character_scale - 0.1)

    finally:
        camera.release()
        tracker.release()
        cv2.destroyAllWindows()

        # Stats
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"\n✅ Test complete!")
        print(f"   Average FPS: {avg_fps:.1f}")


if __name__ == "__main__":
    test_vtuber_renderer()
