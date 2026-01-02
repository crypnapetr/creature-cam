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

        # Compute triangulation with extended landmarks
        temp_puppeteer.puppet.set_landmarks = lambda l: None
        from face_puppeteer import FacePuppet
        puppet = FacePuppet.__new__(FacePuppet)
        puppet.image = character_image
        puppet.landmarks = character_landmarks
        puppet._compute_triangulation()

        temp_puppeteer.puppet.triangulation = puppet.triangulation

        # Generate extended landmarks for target (user) face
        h, w = output_shape[0], output_shape[1]
        target_extended = temp_puppeteer._generate_target_extended_landmarks(
            user_landmarks.image_landmarks,
            user_landmarks,
            w, h
        )

        # Warp face using ADAPTIVE extended landmarks
        warped = temp_puppeteer._warp_face(
            character_image,
            puppet.extended_landmarks,  # Adaptive extended landmarks (478+ points)
            target_extended,             # Matching target extended landmarks (478+ points)
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
        self._cached_warped_face: Optional[np.ndarray] = None
        self._cached_frame_count: int = 0
        self._warp_every_n_frames: int = 3  # Only warp every 3 frames for better performance

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

        # Invalidate all caches when character changes
        self._cached_character_body = None
        self._cached_warped_face = None
        self._cached_background = None

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

        # Use cached body composite if available
        if self._cached_character_body is not None:
            return self._cached_character_body

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

        # Cache for future frames (body is static)
        self._cached_character_body = output

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

        # Performance optimization: Only warp every N frames
        self._cached_frame_count += 1
        should_warp = (self._cached_frame_count % self._warp_every_n_frames == 0)

        # If we have a cached face and shouldn't warp this frame, use cache
        if not should_warp and self._cached_warped_face is not None:
            return self._cached_warped_face

        # Get character face landmarks
        character_landmarks = self.character.get_face_landmarks()
        if character_landmarks is None:
            return base

        # Get character dimensions and scaling
        character_img = self.character.get_character_image()
        char_h, char_w = character_img.shape[:2]
        out_h, out_w = self.output_size[1], self.output_size[0]

        # Calculate character scale and position (same as body rendering)
        scale_w = out_w / char_w
        scale_h = out_h / char_h
        scale = min(scale_w, scale_h) * self.character_scale

        new_w = int(char_w * scale)
        new_h = int(char_h * scale)

        # Character position on output (centered + offset)
        char_x = (out_w - new_w) // 2 + self.character_position[0]
        char_y = (out_h - new_h) // 2 + self.character_position[1]

        # Convert character image to BGR if it has alpha channel
        if character_img.shape[2] == 4:
            character_img_bgr = character_img[:, :, :3]
        else:
            character_img_bgr = character_img

        # Get face regions for both user and character
        user_face_points = user_landmarks.image_landmarks[user_landmarks.face_oval_indices]
        user_face_w = user_face_points[:, 0].max() - user_face_points[:, 0].min()
        user_face_h = user_face_points[:, 1].max() - user_face_points[:, 1].min()

        char_face_points = character_landmarks.image_landmarks[character_landmarks.face_oval_indices]
        char_face_w = char_face_points[:, 0].max() - char_face_points[:, 0].min()
        char_face_h = char_face_points[:, 1].max() - char_face_points[:, 1].min()

        # Resize character image to output scale first
        character_resized = cv2.resize(character_img_bgr, (new_w, new_h))

        # Scale character landmarks to resized image
        resized_char_landmarks_pts = character_landmarks.image_landmarks.copy()
        resized_char_landmarks_pts = resized_char_landmarks_pts * scale

        # Import here to avoid circular dependency
        from face_tracker import FaceLandmarks

        resized_char_landmarks = FaceLandmarks(
            landmarks=character_landmarks.landmarks,
            image_landmarks=resized_char_landmarks_pts,
            face_oval_indices=character_landmarks.face_oval_indices,
            left_eye_indices=character_landmarks.left_eye_indices,
            right_eye_indices=character_landmarks.right_eye_indices,
            lips_indices=character_landmarks.lips_indices,
            nose_indices=character_landmarks.nose_indices
        )

        # Calculate destination face size on output canvas
        dst_y1 = char_y + int(char_face_points[:, 1].min() * scale)
        dst_y2 = char_y + int(char_face_points[:, 1].max() * scale)
        dst_x1 = char_x + int(char_face_points[:, 0].min() * scale)
        dst_x2 = char_x + int(char_face_points[:, 0].max() * scale)

        # Ensure destination is within bounds
        dst_y1 = max(0, min(dst_y1, out_h))
        dst_y2 = max(0, min(dst_y2, out_h))
        dst_x1 = max(0, min(dst_x1, out_w))
        dst_x2 = max(0, min(dst_x2, out_w))

        dst_h = dst_y2 - dst_y1
        dst_w = dst_x2 - dst_x1

        if dst_h <= 0 or dst_w <= 0:
            return base

        # Warp DIRECTLY to the destination size (no black space!)
        # This is much faster and eliminates black regions
        target_shape = (dst_h, dst_w, 3)

        # Create target landmarks at destination size
        # Scale user landmarks to fit destination region
        target_landmarks_pts = user_landmarks.image_landmarks.copy()

        # Get user face bounds
        user_face_min_x = user_face_points[:, 0].min()
        user_face_min_y = user_face_points[:, 1].min()
        user_face_max_x = user_face_points[:, 0].max()
        user_face_max_y = user_face_points[:, 1].max()
        user_face_center_x = (user_face_min_x + user_face_max_x) / 2
        user_face_center_y = (user_face_min_y + user_face_max_y) / 2

        # Transform user landmarks to target region: center, scale, position
        scale_x = dst_w / user_face_w if user_face_w > 0 else 1.0
        scale_y = dst_h / user_face_h if user_face_h > 0 else 1.0
        target_scale = min(scale_x, scale_y) * 0.95  # Slightly smaller to avoid edges

        target_landmarks_pts[:, 0] = (target_landmarks_pts[:, 0] - user_face_center_x) * target_scale + dst_w / 2
        target_landmarks_pts[:, 1] = (target_landmarks_pts[:, 1] - user_face_center_y) * target_scale + dst_h / 2

        from face_tracker import FaceLandmarks
        target_landmarks = FaceLandmarks(
            landmarks=user_landmarks.landmarks,
            image_landmarks=target_landmarks_pts,
            face_oval_indices=user_landmarks.face_oval_indices,
            left_eye_indices=user_landmarks.left_eye_indices,
            right_eye_indices=user_landmarks.right_eye_indices,
            lips_indices=user_landmarks.lips_indices,
            nose_indices=user_landmarks.nose_indices
        )

        # Warp character face directly to target size
        warped_face_region = self.face_warper.warp_character_face(
            character_resized,
            resized_char_landmarks,
            target_landmarks,
            target_shape
        )

        # Start with the base image
        warped_face = base.copy()

        # Blend the warped face region directly at the destination
        if warped_face_region.shape[0] == dst_h and warped_face_region.shape[1] == dst_w:
            roi = warped_face[dst_y1:dst_y2, dst_x1:dst_x2]

            # Create a tight mask for just this face region
            temp_frame_region = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
            face_mask_region = self.face_tracker.create_full_head_mask(
                temp_frame_region,
                target_landmarks
            )

            # Apply mask for smooth blending
            mask_3ch = cv2.merge([face_mask_region, face_mask_region, face_mask_region]) / 255.0
            blended = (warped_face_region * mask_3ch + roi * (1 - mask_3ch)).astype(np.uint8)

            warped_face[dst_y1:dst_y2, dst_x1:dst_x2] = blended

        # Cache the result for next frame
        self._cached_warped_face = warped_face

        return warped_face

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
