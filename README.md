# Creature Cam

Real-time creature face transformation for streaming and content creation.

## Overview

Creature Cam transforms your face into creature hybrids in real-time using face tracking and texture mapping. Perfect for OBS, Twitch, YouTube, and other streaming platforms.

## Features

- Real-time face tracking with MediaPipe (468 facial landmarks)
- Creature texture overlay with configurable intensity
- Optimized for Apple Silicon (Metal Performance Shaders)
- Multiple creature presets (crab, octopus, and more)
- 30+ FPS performance on modern hardware
- Simple keyboard controls for live switching

## Requirements

- macOS (Apple Silicon recommended)
- Python 3.10+
- Webcam
- Dependencies listed in `requirements.txt`

## Installation

1. Create and activate virtual environment:
```bash
cd /Users/asago/clients/creature-cam
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create placeholder textures (for testing):
```bash
cd src
python3 texture_engine.py
```

## Usage

### Running Creature Cam

```bash
cd /Users/asago/clients/creature-cam
source venv/bin/activate  # Activate virtual environment
cd src
python3 main.py
```

**See [QUICKSTART.md](QUICKSTART.md) for a condensed guide.**

### Controls

- `q` - Quit application
- `l` - Toggle landmark visualization (shows face mesh)
- `s` - Toggle stats display
- `1-9` - Switch between available creatures
- `+` / `-` - Adjust transformation intensity

### Testing Individual Components

Test camera capture:
```bash
cd src
python3 camera_capture.py
```

Test face tracking:
```bash
cd src
python3 face_tracker.py
```

## Project Structure

```
creature-cam/
├── src/
│   ├── main.py              # Main application
│   ├── camera_capture.py    # Webcam input handling
│   ├── face_tracker.py      # MediaPipe face mesh tracking
│   └── texture_engine.py    # Texture mapping and blending
├── textures/
│   ├── crab/
│   │   └── skin.png        # Crab shell texture
│   └── octopus/
│       └── skin.png        # Octopus skin texture
├── obs-plugin/              # (Future: OBS integration)
├── requirements.txt
└── README.md
```

## Creating Custom Creature Textures

To add your own creature:

1. Create a new directory in `textures/` with your creature name:
```bash
mkdir textures/myCreature
```

2. Add texture files (PNG with alpha channel recommended):
   - `skin.png` - Main skin/shell texture (required)
   - `eyes.png` - Eye overlay (optional)
   - `appendages.png` - Additional features (optional)

3. Restart the application - your creature will be automatically loaded

### Texture Tips

- Use RGBA format (PNG with transparency) for best results
- Recommended size: 512x512 or larger
- Higher alpha values = more visible creature features
- Textures will be automatically resized to fit face

## Performance

Target performance on Apple Silicon (M1/M2/M3):
- **30+ FPS** at 720p
- **60+ FPS** possible with optimization

If experiencing low FPS:
- Lower camera resolution in `camera_capture.py`
- Reduce texture sizes
- Close other GPU-intensive applications

## Leveraging CreatureBox Textures

This project can use textures extracted from CreatureBox-generated videos:

1. Find successful hybrid videos in `/Users/asago/clients/creaturebox/`
2. Extract key frames with clear creature features
3. Use image editing software to isolate creature elements
4. Save as PNG with alpha channel in `textures/[creature-name]/`

## Future Enhancements

Phase 2 (OBS Integration):
- [ ] OBS virtual camera plugin
- [ ] Web-based creature selector UI
- [ ] Preset management

Phase 3 (Advanced Features):
- [ ] Animated textures (moving tentacles, breathing)
- [ ] Audio-reactive elements
- [ ] Additional creature types
- [ ] Texture extraction tool for CreatureBox videos

## Troubleshooting

**Camera not detected:**
- Check camera permissions in System Preferences > Security & Privacy > Camera
- Try different camera_id in `camera_capture.py` (0, 1, 2, etc.)

**Low FPS:**
- Ensure no other applications are using the camera
- Check Activity Monitor for GPU usage
- Try lowering resolution

**No creatures available:**
- Run `python3 texture_engine.py` to create placeholder textures
- Check that `textures/` directory exists with creature subdirectories

**MediaPipe errors:**
- Ensure you installed dependencies: `pip3 install -r requirements.txt`
- Try reinstalling mediapipe: `pip3 install --upgrade mediapipe`

## Development

Built with insights from CreatureBox project:
- Prompt engineering for creature fusion
- Texture-based transformation approach
- Real-time performance optimization

## License

Personal project for streaming and content creation.
