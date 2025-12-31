"""
VTuber Character Asset Management
Handles loading, processing, and caching of full character images for VTuber mode
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Optional, Tuple, Dict
from dataclasses import dataclass, asdict

# Handle imports whether running as script or module
try:
    from face_tracker import FaceTracker, FaceLandmarks
except ImportError:
    from src.face_tracker import FaceTracker, FaceLandmarks


@dataclass
class FaceRegion:
    """Defines the face portion of a character image"""
    x: int
    y: int
    width: int
    height: int
    padding: int = 40

    @property
    def bounds(self) -> Tuple[int, int, int, int]:
        """Get (x, y, width, height) tuple"""
        return (self.x, self.y, self.width, self.height)

    @property
    def padded_bounds(self) -> Tuple[int, int, int, int]:
        """Get padded bounds for face extraction"""
        return (
            max(0, self.x - self.padding),
            max(0, self.y - self.padding),
            self.width + 2 * self.padding,
            self.height + 2 * self.padding
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'FaceRegion':
        """Create from dictionary"""
        return cls(**data)


class CharacterAsset:
    """Manages a VTuber character's image and metadata"""

    def __init__(self, character_dir: Path):
        """
        Initialize character asset

        Args:
            character_dir: Directory containing character files
        """
        self.character_dir = Path(character_dir)
        self.character_image: Optional[np.ndarray] = None
        self.face_region: Optional[FaceRegion] = None
        self.face_image: Optional[np.ndarray] = None
        self.body_mask: Optional[np.ndarray] = None
        self.face_landmarks: Optional[FaceLandmarks] = None

        # Cached data
        self.cache_file = self.character_dir / "character_cache.json"

    def load_character(self, image_filename: str = "character.png") -> bool:
        """
        Load full character image

        Args:
            image_filename: Name of character image file

        Returns:
            True if successful, False otherwise
        """
        image_path = self.character_dir / image_filename

        if not image_path.exists():
            print(f"❌ Character image not found: {image_path}")
            return False

        self.character_image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

        if self.character_image is None:
            print(f"❌ Failed to load character image: {image_path}")
            return False

        print(f"✅ Loaded character image: {image_path.name}")
        print(f"   Resolution: {self.character_image.shape[1]}x{self.character_image.shape[0]}")

        # Ensure image has alpha channel
        if self.character_image.shape[2] == 3:
            # Add alpha channel (fully opaque)
            alpha = np.ones((self.character_image.shape[0], self.character_image.shape[1], 1), dtype=np.uint8) * 255
            self.character_image = np.concatenate([self.character_image, alpha], axis=2)
            print("   Added alpha channel")

        return True

    def load_face_region(self, region_filename: str = "face_region.json") -> bool:
        """
        Load face region definition from JSON

        Args:
            region_filename: Name of face region JSON file

        Returns:
            True if successful, False otherwise
        """
        region_path = self.character_dir / region_filename

        if not region_path.exists():
            print(f"❌ Face region file not found: {region_path}")
            return False

        try:
            with open(region_path, 'r') as f:
                data = json.load(f)

            self.face_region = FaceRegion.from_dict(data)
            print(f"✅ Loaded face region: {self.face_region.bounds}")

            return True

        except Exception as e:
            print(f"❌ Failed to load face region: {e}")
            return False

    def auto_detect_face_region(self, padding: int = 40) -> bool:
        """
        Automatically detect face region using MediaPipe

        Args:
            padding: Padding around detected face

        Returns:
            True if successful, False otherwise
        """
        if self.character_image is None:
            print("❌ Character image not loaded")
            return False

        print("🔍 Auto-detecting face region...")

        # Initialize face tracker
        tracker = FaceTracker()

        # Convert to BGR for processing
        if self.character_image.shape[2] == 4:
            bgr_image = self.character_image[:, :, :3]
        else:
            bgr_image = self.character_image

        # Detect face landmarks
        landmarks = tracker.process_frame(bgr_image)

        if landmarks is None:
            print("❌ No face detected in character image")
            tracker.release()
            return False

        # Get face bounding box from face oval
        face_points = landmarks.image_landmarks[landmarks.face_oval_indices]
        x, y, w, h = cv2.boundingRect(face_points)

        # Create face region with padding
        self.face_region = FaceRegion(x, y, w, h, padding)

        print(f"✅ Face region detected: {self.face_region.bounds}")
        print(f"   Padded bounds: {self.face_region.padded_bounds}")

        # Cache the landmarks
        self.face_landmarks = landmarks

        tracker.release()
        return True

    def save_face_region(self, region_filename: str = "face_region.json") -> bool:
        """
        Save face region definition to JSON

        Args:
            region_filename: Name of output JSON file

        Returns:
            True if successful, False otherwise
        """
        if self.face_region is None:
            print("❌ No face region to save")
            return False

        region_path = self.character_dir / region_filename

        try:
            with open(region_path, 'w') as f:
                json.dump(self.face_region.to_dict(), f, indent=2)

            print(f"✅ Saved face region to: {region_path}")
            return True

        except Exception as e:
            print(f"❌ Failed to save face region: {e}")
            return False

    def extract_face_region(self) -> Optional[np.ndarray]:
        """
        Extract face region from character image

        Returns:
            Face image with alpha channel, or None if failed
        """
        if self.character_image is None or self.face_region is None:
            print("❌ Character image or face region not loaded")
            return None

        x, y, w, h = self.face_region.padded_bounds

        # Ensure bounds are within image
        img_h, img_w = self.character_image.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(img_w - x, w)
        h = min(img_h - y, h)

        if w <= 0 or h <= 0:
            print("❌ Invalid face region bounds")
            return None

        # Extract face region
        self.face_image = self.character_image[y:y+h, x:x+w].copy()

        print(f"✅ Extracted face region: {self.face_image.shape[1]}x{self.face_image.shape[0]}")

        return self.face_image

    def extract_body_mask(self) -> Optional[np.ndarray]:
        """
        Extract body mask (everything except face region)

        Returns:
            Binary mask where 255 = body, 0 = face, or None if failed
        """
        if self.character_image is None or self.face_region is None:
            print("❌ Character image or face region not loaded")
            return None

        img_h, img_w = self.character_image.shape[:2]

        # Create full mask (all body)
        self.body_mask = np.ones((img_h, img_w), dtype=np.uint8) * 255

        # Subtract face region (with padding)
        x, y, w, h = self.face_region.padded_bounds

        # Ensure bounds are within image
        x = max(0, x)
        y = max(0, y)
        w = min(img_w - x, w)
        h = min(img_h - y, h)

        # Set face region to 0 (not body)
        self.body_mask[y:y+h, x:x+w] = 0

        # Feather the edges for smooth blending
        self.body_mask = cv2.GaussianBlur(self.body_mask, (21, 21), 15)

        print(f"✅ Extracted body mask")

        return self.body_mask

    def compute_face_landmarks(self) -> Optional[FaceLandmarks]:
        """
        Compute face landmarks on character face
        This is done once and cached for performance

        Returns:
            FaceLandmarks or None if failed
        """
        if self.face_landmarks is not None:
            print("ℹ️  Using cached face landmarks")
            return self.face_landmarks

        if self.character_image is None:
            print("❌ Character image not loaded")
            return None

        print("🔍 Computing face landmarks...")

        # Initialize face tracker
        tracker = FaceTracker()

        # Convert to BGR for processing
        if self.character_image.shape[2] == 4:
            bgr_image = self.character_image[:, :, :3]
        else:
            bgr_image = self.character_image

        # Detect face landmarks
        self.face_landmarks = tracker.process_frame(bgr_image)

        if self.face_landmarks is None:
            print("❌ Failed to detect face landmarks")
        else:
            print(f"✅ Computed {len(self.face_landmarks.image_landmarks)} face landmarks")

        tracker.release()
        return self.face_landmarks

    def get_character_image(self) -> Optional[np.ndarray]:
        """Get full character image"""
        return self.character_image

    def get_face_image(self) -> Optional[np.ndarray]:
        """Get face region image"""
        return self.face_image

    def get_body_mask(self) -> Optional[np.ndarray]:
        """Get body mask"""
        return self.body_mask

    def get_face_landmarks(self) -> Optional[FaceLandmarks]:
        """Get cached face landmarks"""
        return self.face_landmarks

    def initialize(self, auto_detect: bool = False) -> bool:
        """
        Initialize character asset (convenience method)

        Args:
            auto_detect: If True, auto-detect face region instead of loading from file

        Returns:
            True if successful, False otherwise
        """
        print(f"\n{'='*70}")
        print(f"Initializing Character: {self.character_dir.name}")
        print(f"{'='*70}")

        # Load character image
        if not self.load_character():
            return False

        # Load or detect face region
        if auto_detect:
            if not self.auto_detect_face_region():
                return False
            # Save the detected region
            self.save_face_region()
        else:
            if not self.load_face_region():
                print("ℹ️  Face region not found, attempting auto-detection...")
                if not self.auto_detect_face_region():
                    return False
                self.save_face_region()

        # Extract face region
        if self.extract_face_region() is None:
            return False

        # Extract body mask
        if self.extract_body_mask() is None:
            return False

        # Compute face landmarks
        if self.compute_face_landmarks() is None:
            return False

        print(f"\n✅ Character initialized successfully!")
        print(f"{'='*70}\n")

        return True


def test_character_asset():
    """Test character asset loading and processing"""
    import sys

    # Check for character directory argument
    if len(sys.argv) > 1:
        character_dir = Path(sys.argv[1])
    else:
        # Default to trump-crab
        character_dir = Path(__file__).parent.parent / "textures" / "trump-crab"

    if not character_dir.exists():
        print(f"❌ Character directory not found: {character_dir}")
        return

    # Initialize character
    character = CharacterAsset(character_dir)

    # Check if character.png exists, if not use skin.png
    character_file = character_dir / "character.png"
    if not character_file.exists():
        skin_file = character_dir / "skin.png"
        if skin_file.exists():
            print(f"ℹ️  character.png not found, using skin.png")
            if character.initialize(auto_detect=True):
                # Save as character.png for future use
                cv2.imwrite(str(character_file), character.character_image)
                print(f"✅ Saved as character.png")
        else:
            print(f"❌ Neither character.png nor skin.png found")
            return
    else:
        if not character.initialize(auto_detect=False):
            return

    # Display results
    print("\n" + "="*70)
    print("Displaying character components...")
    print("Press any key to cycle through images, 'q' to quit")
    print("="*70 + "\n")

    images = [
        ("Full Character", character.get_character_image()),
        ("Face Region", character.get_face_image()),
        ("Body Mask", character.get_body_mask()),
    ]

    for title, img in images:
        if img is None:
            continue

        # Convert for display
        if len(img.shape) == 2:  # Grayscale
            display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:  # RGBA
            display = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            display = img

        # Resize for display
        h, w = display.shape[:2]
        scale = 800 / max(w, h)
        display = cv2.resize(display, (int(w * scale), int(h * scale)))

        # Add title
        cv2.putText(display, title, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Character Asset Viewer", display)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    print("\n✅ Test complete!")


if __name__ == "__main__":
    test_character_asset()
