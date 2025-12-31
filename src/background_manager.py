"""
Background Manager for VTuber Mode
Handles loading, resizing, and managing background images
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple


class BackgroundManager:
    """Manages background images for VTuber mode"""

    def __init__(self, backgrounds_dir: Path, output_size: Tuple[int, int] = (1280, 720)):
        """
        Initialize background manager

        Args:
            backgrounds_dir: Directory containing background images
            output_size: Target output size (width, height)
        """
        self.backgrounds_dir = Path(backgrounds_dir)
        self.output_size = output_size
        self.current_background: Optional[np.ndarray] = None
        self.current_background_name: Optional[str] = None
        self.backgrounds: dict = {}

        # Ensure backgrounds directory exists
        self.backgrounds_dir.mkdir(parents=True, exist_ok=True)

        # Load available backgrounds
        self._load_backgrounds()

    def _load_backgrounds(self):
        """Load all background images from directory"""
        if not self.backgrounds_dir.exists():
            print(f"⚠️  Backgrounds directory not found: {self.backgrounds_dir}")
            return

        # Supported image formats
        formats = ['*.png', '*.jpg', '*.jpeg', '*.bmp']

        background_files = []
        for fmt in formats:
            background_files.extend(self.backgrounds_dir.glob(fmt))

        if not background_files:
            print(f"ℹ️  No background images found in: {self.backgrounds_dir}")
            return

        print(f"\n{'='*70}")
        print(f"Loading backgrounds from: {self.backgrounds_dir}")
        print(f"{'='*70}")

        for bg_path in sorted(background_files):
            try:
                img = cv2.imread(str(bg_path))
                if img is not None:
                    # Resize to output size
                    img_resized = cv2.resize(img, self.output_size)
                    self.backgrounds[bg_path.stem] = img_resized
                    print(f"✅ Loaded: {bg_path.name} ({img.shape[1]}x{img.shape[0]} → {self.output_size[0]}x{self.output_size[1]})")
            except Exception as e:
                print(f"❌ Failed to load {bg_path.name}: {e}")

        if self.backgrounds:
            print(f"\n✅ Loaded {len(self.backgrounds)} background(s)")
        else:
            print(f"\n⚠️  No backgrounds loaded")

    def load_background(self, name_or_path: str) -> bool:
        """
        Load and set a background image

        Args:
            name_or_path: Background name (from directory) or file path

        Returns:
            True if successful, False otherwise
        """
        # Check if it's a name from loaded backgrounds
        if name_or_path in self.backgrounds:
            self.current_background = self.backgrounds[name_or_path]
            self.current_background_name = name_or_path
            print(f"✅ Switched to background: {name_or_path}")
            return True

        # Try as file path
        path = Path(name_or_path)
        if not path.exists():
            print(f"❌ Background not found: {name_or_path}")
            return False

        try:
            img = cv2.imread(str(path))
            if img is None:
                print(f"❌ Failed to load image: {path}")
                return False

            # Resize to output size
            self.current_background = cv2.resize(img, self.output_size)
            self.current_background_name = path.stem

            print(f"✅ Loaded background: {path.name} ({img.shape[1]}x{img.shape[0]} → {self.output_size[0]}x{self.output_size[1]})")
            return True

        except Exception as e:
            print(f"❌ Failed to load background: {e}")
            return False

    def create_solid_color_background(self, color: Tuple[int, int, int] = (40, 40, 40)) -> np.ndarray:
        """
        Create a solid color background

        Args:
            color: BGR color tuple

        Returns:
            Background image
        """
        bg = np.full((self.output_size[1], self.output_size[0], 3), color, dtype=np.uint8)
        print(f"✅ Created solid color background: RGB{color[::-1]}")
        return bg

    def create_gradient_background(self,
                                   color1: Tuple[int, int, int] = (60, 40, 20),
                                   color2: Tuple[int, int, int] = (20, 40, 60),
                                   direction: str = 'vertical') -> np.ndarray:
        """
        Create a gradient background

        Args:
            color1: Start color (BGR)
            color2: End color (BGR)
            direction: 'vertical' or 'horizontal'

        Returns:
            Background image
        """
        w, h = self.output_size
        bg = np.zeros((h, w, 3), dtype=np.uint8)

        if direction == 'vertical':
            for i in range(h):
                ratio = i / h
                color = tuple(int(c1 * (1 - ratio) + c2 * ratio) for c1, c2 in zip(color1, color2))
                bg[i, :] = color
        else:  # horizontal
            for i in range(w):
                ratio = i / w
                color = tuple(int(c1 * (1 - ratio) + c2 * ratio) for c1, c2 in zip(color1, color2))
                bg[:, i] = color

        print(f"✅ Created {direction} gradient background")
        return bg

    def get_current(self) -> np.ndarray:
        """
        Get current background, creating default if none set

        Returns:
            Current background image
        """
        if self.current_background is None:
            # Create default gradient background
            print("ℹ️  No background set, creating default gradient...")
            self.current_background = self.create_gradient_background()
            self.current_background_name = "default_gradient"

        return self.current_background.copy()

    def get_available_backgrounds(self) -> List[str]:
        """
        Get list of available background names

        Returns:
            List of background names
        """
        return list(self.backgrounds.keys())

    def set_output_size(self, width: int, height: int):
        """
        Change output size (will resize current background)

        Args:
            width: New width
            height: New height
        """
        self.output_size = (width, height)

        # Resize current background if set
        if self.current_background is not None:
            self.current_background = cv2.resize(self.current_background, self.output_size)

        # Resize all loaded backgrounds
        for name, bg in self.backgrounds.items():
            self.backgrounds[name] = cv2.resize(bg, self.output_size)

        print(f"✅ Updated output size to: {width}x{height}")


def create_default_background():
    """Create default background image"""
    backgrounds_dir = Path(__file__).parent.parent / "backgrounds"
    backgrounds_dir.mkdir(parents=True, exist_ok=True)

    default_path = backgrounds_dir / "default.png"

    if default_path.exists():
        print(f"ℹ️  Default background already exists: {default_path}")
        return

    print("Creating default background...")

    # Create a nice gradient
    manager = BackgroundManager(backgrounds_dir)
    gradient = manager.create_gradient_background(
        color1=(80, 60, 40),   # Dark blue-brown
        color2=(40, 60, 80),   # Dark brown-blue
        direction='vertical'
    )

    cv2.imwrite(str(default_path), gradient)
    print(f"✅ Created default background: {default_path}")


def test_background_manager():
    """Test background manager functionality"""
    print("\n" + "="*70)
    print("Background Manager Test")
    print("="*70 + "\n")

    backgrounds_dir = Path(__file__).parent.parent / "backgrounds"

    # Create default background if needed
    create_default_background()

    # Initialize manager
    manager = BackgroundManager(backgrounds_dir)

    # Show available backgrounds
    available = manager.get_available_backgrounds()
    if available:
        print(f"\nAvailable backgrounds: {', '.join(available)}")

    # Test getting current (should create default)
    current = manager.get_current()
    print(f"Current background: {current.shape}")

    # Display backgrounds
    print("\nDisplaying backgrounds...")
    print("Press any key to cycle through, 'q' to quit\n")

    # Show default gradient
    gradients = [
        ("Vertical Gradient", manager.create_gradient_background(direction='vertical')),
        ("Horizontal Gradient", manager.create_gradient_background(direction='horizontal')),
        ("Solid Dark", manager.create_solid_color_background((40, 40, 40))),
        ("Solid Blue", manager.create_solid_color_background((100, 50, 30))),
    ]

    # Add loaded backgrounds
    for name in available:
        manager.load_background(name)
        gradients.append((f"Loaded: {name}", manager.get_current()))

    for title, bg in gradients:
        display = bg.copy()

        # Add title
        cv2.putText(display, title, (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(display, f"Size: {bg.shape[1]}x{bg.shape[0]}", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        cv2.imshow("Background Manager Test", display)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    print("\n✅ Test complete!")


if __name__ == "__main__":
    test_background_manager()
