# Creature Cam - AI Development Guide

This document provides context for AI assistants (like Claude) working on the Creature Cam project.

## Project Overview

**Creature Cam** is a real-time face transformation application that turns webcam faces into creature hybrids using texture mapping and face tracking. Built for streaming (OBS, Twitch, YouTube).

**Key Goals:**
- Real-time performance (30+ FPS)
- Creature texture overlay on human faces
- Live switching between creatures
- Configurable transformation intensity (70% creature / 30% human by default)

## Technology Stack

- **Python 3.10+** with virtual environment
- **OpenCV** - Camera capture and image processing
- **MediaPipe** - Face mesh tracking (468 landmarks)
- **PyTorch** - GPU acceleration (Apple Silicon MPS support)
- **Pillow** - Texture manipulation
- **NumPy** - Numerical operations

**Platform:** macOS (Apple Silicon optimized)

## Project Structure

```
creature-cam/
├── venv/                    # Python virtual environment (not in git)
├── src/
│   ├── main.py             # Main application with UI loop
│   ├── camera_capture.py   # Webcam input wrapper
│   ├── face_tracker.py     # MediaPipe face mesh integration
│   └── texture_engine.py   # Creature texture mapping and blending
├── textures/
│   ├── crab/
│   │   └── skin.png       # Crab shell texture
│   └── octopus/
│       └── skin.png       # Octopus skin texture
├── requirements.txt
├── README.md
├── QUICKSTART.md
└── CLAUDE.md              # This file
```

## Core Components

### 1. CameraCapture (camera_capture.py)

**Purpose:** Abstraction layer for webcam access

**Key Methods:**
- `start()` - Initialize camera with 720p resolution at 30 FPS
- `read_frame()` - Capture single BGR frame
- `release()` - Clean up camera resources

**Performance:**
- Default resolution: 1280x720 (balance of quality/speed)
- Target FPS: 30
- Context manager support for automatic cleanup

**Testing:**
```bash
cd src
python3 camera_capture.py  # Shows live camera feed with FPS counter
```

### 2. FaceTracker (face_tracker.py)

**Purpose:** Real-time face landmark detection using MediaPipe

**Key Features:**
- 468 facial landmarks tracked per frame
- Face segmentation and masking
- Key region identification (eyes, lips, nose, face oval)

**Data Structure:**
```python
@dataclass
class FaceLandmarks:
    landmarks: np.ndarray           # (468, 3) normalized coordinates
    image_landmarks: np.ndarray     # (468, 2) pixel coordinates
    face_oval_indices: List[int]    # Face contour points
    left_eye_indices: List[int]
    right_eye_indices: List[int]
    lips_indices: List[int]
    nose_indices: List[int]
```

**Key Methods:**
- `process_frame(frame)` - Detect face and return landmarks
- `draw_landmarks(frame, landmarks)` - Visualize tracking
- `get_face_mask(frame, landmarks)` - Binary mask of face region

**Performance:**
- Runs at 30-60 FPS on Apple Silicon
- Min detection confidence: 0.5
- Min tracking confidence: 0.5

**Testing:**
```bash
cd src
python3 face_tracker.py
# Press 'm' for mesh, 'c' for contours, 'q' to quit
```

### 3. TextureEngine (texture_engine.py)

**Purpose:** Apply creature textures to detected faces

**Architecture:**
- `CreatureTexture` - Represents a single creature's texture set
- `TextureEngine` - Manages multiple creatures and applies transformations

**Texture Files (per creature):**
- `skin.png` - Main skin/shell texture (REQUIRED)
- `eyes.png` - Eye overlay (optional)
- `appendages.png` - Extra features like claws/tentacles (optional)

**Key Methods:**
- `set_creature(name)` - Switch active creature
- `set_intensity(float)` - Adjust transformation strength (0.0-1.0)
- `apply_creature_transformation(frame, landmarks)` - Main rendering

**Blending Algorithm:**
1. Detect face region using landmarks
2. Resize creature texture to fit face bounding box
3. Create mask from face oval contour
4. Blend texture with original using alpha compositing:
   ```
   result = texture * (alpha * intensity) + original * (1 - alpha * intensity)
   ```

**Creating Placeholder Textures:**
```bash
cd src
python3 texture_engine.py  # Generates test crab and octopus textures
```

### 4. CreatureCam (main.py)

**Purpose:** Main application loop integrating all components

**Application Flow:**
```
Initialize → Start Camera → Loop:
  1. Capture frame
  2. Track face (MediaPipe)
  3. Apply creature texture (if face detected)
  4. Draw UI overlay (stats, FPS)
  5. Display frame
  6. Handle keyboard input
```

**Keyboard Controls:**
- `q` - Quit
- `l` - Toggle landmark visualization
- `s` - Toggle stats
- `1-9` - Switch creatures
- `+/-` - Adjust intensity

**Performance Monitoring:**
- Real-time FPS calculation
- Face detection status
- Current creature and intensity display

## Development Commands

### Setup
```bash
cd /Users/asago/clients/creature-cam
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd src && python3 texture_engine.py  # Create test textures
```

### Run Application
```bash
source venv/bin/activate
cd src
python3 main.py
```

### Test Individual Components
```bash
# Camera only
python3 camera_capture.py

# Face tracking only
python3 face_tracker.py

# Texture generation
python3 texture_engine.py
```

## Performance Optimization

**Current Performance:**
- Target: 30 FPS minimum
- Achieved: 30-60 FPS on Apple Silicon

**Optimization Strategies:**
1. **Resolution:** 720p processing (1280x720) balances quality and speed
2. **Face Tracking:** MediaPipe optimized for real-time
3. **GPU Acceleration:** PyTorch with MPS backend (Apple Silicon)
4. **Texture Caching:** Textures loaded once at startup
5. **Alpha Blending:** Optimized NumPy operations

**If FPS drops below 30:**
- Reduce camera resolution in `camera_capture.py`
- Lower texture sizes (512x512 recommended)
- Disable landmark visualization (`l` key)
- Close other GPU-intensive apps

## Adding Creatures

### From Scratch

1. Create directory in `textures/`:
```bash
mkdir textures/jellyfish
```

2. Add `skin.png` (RGBA recommended):
   - Recommended size: 512x512 or larger
   - Use PNG with alpha channel for transparency
   - Higher alpha = more opaque creature features

3. Optionally add `eyes.png` and `appendages.png`

4. Restart application - creature auto-loaded

### From CreatureBox Videos

This project was designed to leverage textures from the CreatureBox project (`/Users/asago/clients/creaturebox/`).

**Extraction Process:**
1. Identify successful hybrid video (e.g., Trump-crab fusion)
2. Extract key frames with clear creature features
3. Use image editor (Photoshop/GIMP) to:
   - Isolate creature elements (shell, claws, skin texture)
   - Create alpha mask for transparency
   - Save as RGBA PNG
4. Place in `textures/[creature-name]/skin.png`

**Recommended Source Videos:**
- Videos with clean, well-lit creature features
- Minimal morphing/transitions (stable creature state)
- High-resolution textures visible

## Common Issues

### "ModuleNotFoundError: No module named 'cv2'"
**Cause:** Virtual environment not activated

**Fix:**
```bash
source venv/bin/activate
```

### "Camera not found" or "Could not open camera 0"
**Cause:** Camera permissions or wrong device ID

**Fix:**
1. Check System Preferences > Security & Privacy > Camera
2. Try different camera IDs in `camera_capture.py` (change `camera_id=0` to `camera_id=1`, etc.)

### Low FPS
**Cause:** GPU/CPU overload or camera resolution too high

**Fix:**
- Lower resolution: Edit `camera_capture.py`, change `width=1280, height=720` to `width=640, height=480`
- Check Activity Monitor for GPU usage
- Close other apps using camera

### "No creatures available"
**Cause:** Textures not created

**Fix:**
```bash
cd src
python3 texture_engine.py
```

### Textures not visible
**Cause:** Alpha channel or intensity too low

**Fix:**
- Press `+` key multiple times to increase intensity
- Check texture has alpha channel: `file textures/crab/skin.png` should show "PNG image data, ... with alpha"

## Future Enhancements (Planned)

### Phase 2: OBS Integration
- OBS Python plugin for virtual camera output
- Web-based creature selector UI
- Stream directly to OBS without preview window

### Phase 3: Advanced Features
- Animated textures (moving tentacles, breathing motion)
- Audio-reactive elements (creature reacts to voice)
- More creature types from CreatureBox library
- Texture extraction tool (automate CreatureBox video → texture pipeline)

### Phase 4: Cross-Platform
- Windows support (DirectShow camera, Windows virtual camera)
- Linux support

## Relationship to CreatureBox Project

**CreatureBox** (`/Users/asago/clients/creaturebox/`) generates static creature hybrid videos using:
- OpenAI GPT for creature descriptions
- LumaAI for video generation
- Fish.audio for Trump voice synthesis

**Creature Cam** applies similar concepts in real-time:
- Uses creature textures (potentially extracted from CreatureBox videos)
- Real-time face transformation instead of pre-rendered video
- Interactive controls for streaming/content creation

**Shared Learnings:**
- Creature fusion prompt engineering (from CreatureBox)
- Texture quality requirements
- Visual style and aesthetics

## Code Style and Conventions

- **Type hints:** Use where helpful for clarity
- **Docstrings:** All classes and public methods
- **Error handling:** Graceful degradation (e.g., fallback if texture missing)
- **Context managers:** For resource cleanup (camera, MediaPipe)
- **NumPy operations:** Prefer vectorized operations over loops
- **Comments:** Explain "why" not "what"

## Testing Guidelines

When making changes:

1. **Test camera capture:**
```bash
python3 camera_capture.py
```
Should show live feed at 30 FPS.

2. **Test face tracking:**
```bash
python3 face_tracker.py
```
Should detect face with green contours.

3. **Test full application:**
```bash
python3 main.py
```
Should show creature transformation in real-time.

4. **Performance check:**
- FPS should stay above 30
- Face detection should be consistent (>90% when facing camera)
- No visible lag between movement and display

## Dependencies Rationale

- **opencv-python** - Industry standard for computer vision, camera I/O
- **mediapipe** - Google's optimized face tracking, faster than dlib
- **torch** - GPU acceleration via MPS on Apple Silicon
- **Pillow** - Image loading and manipulation
- **numpy** - Numerical operations, essential for image processing

**Why not use:**
- **dlib:** Slower than MediaPipe for real-time face tracking
- **tensorflow:** MediaPipe already includes TensorFlow Lite models
- **obs-websocket-py:** Deferred to Phase 2 (OBS integration)

## Git Workflow (if versioning)

Recommended `.gitignore`:
```
venv/
__pycache__/
*.pyc
.DS_Store
textures/*/  # Optional: exclude generated textures
```

## Contact / Feedback

This is a personal project for streaming/content creation. For issues or questions, refer to the main README.md.

---

**Last Updated:** 2025-12-30
**Version:** 1.0 (Phase 1 - Core Engine)
