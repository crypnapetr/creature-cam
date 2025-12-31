"""
Texture engine for applying creature transformations to faces
Implements texture mapping and blending for real-time creature effects
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Dict, Tuple
from face_tracker import FaceLandmarks


class CreatureTexture:
    """Represents a creature texture set"""

    def __init__(self, texture_dir: Path):
        """
        Load creature textures from directory

        Expected files:
            - skin.png: Main skin/shell texture
            - eyes.png: Eye overlay (optional)
            - appendages.png: Additional features like claws/tentacles (optional)
        """
        self.texture_dir = texture_dir
        self.skin_texture: Optional[np.ndarray] = None
        self.eyes_texture: Optional[np.ndarray] = None
        self.appendages_texture: Optional[np.ndarray] = None

        self._load_textures()

    def _load_textures(self):
        """Load texture files"""
        skin_path = self.texture_dir / "skin.png"
        if skin_path.exists():
            self.skin_texture = cv2.imread(str(skin_path), cv2.IMREAD_UNCHANGED)
            print(f"Loaded skin texture: {skin_path}")

        eyes_path = self.texture_dir / "eyes.png"
        if eyes_path.exists():
            self.eyes_texture = cv2.imread(str(eyes_path), cv2.IMREAD_UNCHANGED)
            print(f"Loaded eyes texture: {eyes_path}")

        appendages_path = self.texture_dir / "appendages.png"
        if appendages_path.exists():
            self.appendages_texture = cv2.imread(str(appendages_path), cv2.IMREAD_UNCHANGED)
            print(f"Loaded appendages texture: {appendages_path}")


class TextureEngine:
    """Applies creature textures to faces in real-time"""

    def __init__(self, textures_dir: Path):
        """
        Initialize texture engine

        Args:
            textures_dir: Path to textures directory containing creature subdirs
        """
        self.textures_dir = textures_dir
        self.creatures: Dict[str, CreatureTexture] = {}
        self.current_creature: Optional[str] = None
        self.transformation_intensity = 0.7  # 70% creature, 30% human

        self._load_creatures()

    def _load_creatures(self):
        """Load all available creature textures"""
        if not self.textures_dir.exists():
            print(f"Warning: Textures directory not found: {self.textures_dir}")
            return

        for creature_dir in self.textures_dir.iterdir():
            if creature_dir.is_dir():
                creature_name = creature_dir.name
                try:
                    self.creatures[creature_name] = CreatureTexture(creature_dir)
                    print(f"Loaded creature: {creature_name}")
                except Exception as e:
                    print(f"Error loading creature {creature_name}: {e}")

        if self.creatures:
            self.current_creature = list(self.creatures.keys())[0]
            print(f"Default creature set to: {self.current_creature}")

    def set_creature(self, creature_name: str) -> bool:
        """
        Set the active creature

        Args:
            creature_name: Name of creature to activate

        Returns:
            True if successful, False if creature not found
        """
        if creature_name in self.creatures:
            self.current_creature = creature_name
            print(f"Switched to creature: {creature_name}")
            return True
        else:
            print(f"Creature not found: {creature_name}")
            return False

    def set_intensity(self, intensity: float):
        """
        Set transformation intensity

        Args:
            intensity: 0.0 (all human) to 1.0 (all creature)
        """
        self.transformation_intensity = np.clip(intensity, 0.0, 1.0)
        print(f"Transformation intensity: {self.transformation_intensity:.0%}")

    def apply_creature_transformation(self,
                                     frame: np.ndarray,
                                     landmarks: FaceLandmarks) -> np.ndarray:
        """
        Apply creature transformation to frame

        Args:
            frame: Original BGR frame
            landmarks: Face landmarks

        Returns:
            Transformed frame with creature applied
        """
        if not self.current_creature or self.current_creature not in self.creatures:
            return frame

        creature = self.creatures[self.current_creature]

        # Start with original frame
        output = frame.copy()

        # Apply skin texture if available
        if creature.skin_texture is not None:
            output = self._apply_skin_texture(output, landmarks, creature.skin_texture)

        # Apply eye texture if available
        if creature.eyes_texture is not None:
            output = self._apply_eyes_texture(output, landmarks, creature.eyes_texture)

        return output

    def _apply_skin_texture(self,
                           frame: np.ndarray,
                           landmarks: FaceLandmarks,
                           texture: np.ndarray) -> np.ndarray:
        """
        Apply skin/shell texture to face region with aggressive transformation

        Args:
            frame: Original frame
            landmarks: Face landmarks
            texture: Texture image (RGBA or BGR)

        Returns:
            Frame with texture applied
        """
        # Get face region
        face_points = landmarks.image_landmarks[landmarks.face_oval_indices]

        # Get bounding box of face
        x, y, w, h = cv2.boundingRect(face_points)

        # Ensure within frame bounds
        x = max(0, x)
        y = max(0, y)
        w = min(frame.shape[1] - x, w)
        h = min(frame.shape[0] - y, h)

        if w <= 0 or h <= 0:
            return frame

        # Resize texture to fit face region
        texture_resized = cv2.resize(texture, (w, h))

        # Create mask for face region
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [face_points], 255)

        # Extract face region mask
        face_mask = mask[y:y+h, x:x+w]

        # Blend texture with original frame
        face_region = frame[y:y+h, x:x+w].astype(np.float32)

        if texture_resized.shape[2] == 4:  # RGBA texture
            # Use alpha channel for blending
            alpha = texture_resized[:, :, 3:4] / 255.0
            texture_rgb = texture_resized[:, :, :3].astype(np.float32)
        else:  # BGR texture
            alpha = np.ones((h, w, 1), dtype=np.float32)
            texture_rgb = texture_resized.astype(np.float32)

        # Apply face mask to alpha
        alpha = alpha * (face_mask[:, :, np.newaxis] / 255.0)

        # ENHANCED BLENDING: Use multiply blend mode for more dramatic effect
        # This makes the texture actually change the skin color, not just tint it
        intensity = self.transformation_intensity

        # Method 1: Multiply blend (darkens, adds texture detail)
        multiply_blend = (face_region * texture_rgb / 255.0)

        # Method 2: Color replacement (replaces skin color with creature color)
        color_replace = texture_rgb

        # Combine multiply and replacement based on intensity
        # Lower intensity = multiply (subtle texture)
        # Higher intensity = color replacement (dramatic transformation)
        if intensity < 0.5:
            # Subtle: multiply blend
            blended = multiply_blend * alpha + face_region * (1 - alpha)
        else:
            # Dramatic: more color replacement
            replacement_factor = (intensity - 0.5) * 2  # 0.0 to 1.0
            blended = (
                multiply_blend * (1 - replacement_factor) +
                color_replace * replacement_factor
            ) * alpha + face_region * (1 - alpha)

        # Clip and convert back to uint8
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        # Place blended region back
        output = frame.copy()
        output[y:y+h, x:x+w] = blended

        return output

    def _apply_eyes_texture(self,
                           frame: np.ndarray,
                           landmarks: FaceLandmarks,
                           texture: np.ndarray) -> np.ndarray:
        """
        Apply creature eyes texture

        Args:
            frame: Current frame
            landmarks: Face landmarks
            texture: Eye texture image

        Returns:
            Frame with creature eyes applied
        """
        # Apply to both eyes
        for eye_indices in [landmarks.left_eye_indices, landmarks.right_eye_indices]:
            eye_points = landmarks.image_landmarks[eye_indices]

            # Get bounding box
            x, y, w, h = cv2.boundingRect(eye_points)

            # Ensure within bounds
            x = max(0, x)
            y = max(0, y)
            w = min(frame.shape[1] - x, w)
            h = min(frame.shape[0] - y, h)

            if w <= 0 or h <= 0:
                continue

            # Resize texture to fit eye
            texture_resized = cv2.resize(texture, (w, h))

            # Create mask for eye region
            mask = np.zeros((h, w), dtype=np.uint8)
            eye_points_shifted = eye_points - [x, y]
            cv2.fillPoly(mask, [eye_points_shifted], 255)

            # Blend texture
            eye_region = frame[y:y+h, x:x+w]

            if texture_resized.shape[2] == 4:  # RGBA
                alpha = texture_resized[:, :, 3:4] / 255.0
                texture_rgb = texture_resized[:, :, :3]
                alpha = alpha * (mask[:, :, np.newaxis] / 255.0)
                blended = (texture_rgb * alpha + eye_region * (1 - alpha)).astype(np.uint8)
            else:  # BGR
                alpha = mask[:, :, np.newaxis] / 255.0
                blended = (texture_resized * alpha + eye_region * (1 - alpha)).astype(np.uint8)

            frame[y:y+h, x:x+w] = blended

        return frame

    def get_available_creatures(self) -> list:
        """Get list of available creature names"""
        return list(self.creatures.keys())


def create_placeholder_textures():
    """Create placeholder textures for testing"""
    import os

    base_dir = Path(__file__).parent.parent / "textures"

    # Create crab textures
    crab_dir = base_dir / "crab"
    crab_dir.mkdir(parents=True, exist_ok=True)

    # Create a simple orange/red shell texture
    crab_skin = np.zeros((512, 512, 4), dtype=np.uint8)
    crab_skin[:, :, 0] = 50   # B
    crab_skin[:, :, 1] = 100  # G
    crab_skin[:, :, 2] = 255  # R - orange/red shell
    crab_skin[:, :, 3] = 200  # A - semi-transparent

    # Add some texture (simple pattern)
    for i in range(0, 512, 20):
        cv2.line(crab_skin, (i, 0), (i, 512), (40, 80, 200, 200), 2)
        cv2.line(crab_skin, (0, i), (512, i), (40, 80, 200, 200), 2)

    cv2.imwrite(str(crab_dir / "skin.png"), crab_skin)
    print(f"Created placeholder crab skin texture")

    # Create octopus textures
    octopus_dir = base_dir / "octopus"
    octopus_dir.mkdir(parents=True, exist_ok=True)

    # Create a purple/pink octopus skin texture
    octopus_skin = np.zeros((512, 512, 4), dtype=np.uint8)
    octopus_skin[:, :, 0] = 180  # B
    octopus_skin[:, :, 1] = 100  # G
    octopus_skin[:, :, 2] = 150  # R - purple/pink
    octopus_skin[:, :, 3] = 200  # A

    # Add sucker pattern
    for x in range(50, 512, 60):
        for y in range(50, 512, 60):
            cv2.circle(octopus_skin, (x, y), 15, (150, 80, 130, 220), -1)
            cv2.circle(octopus_skin, (x, y), 10, (100, 50, 100, 220), -1)

    cv2.imwrite(str(octopus_dir / "skin.png"), octopus_skin)
    print(f"Created placeholder octopus skin texture")


if __name__ == "__main__":
    print("Creating placeholder textures for testing...")
    create_placeholder_textures()
    print("Done! Textures created in ../textures/")
