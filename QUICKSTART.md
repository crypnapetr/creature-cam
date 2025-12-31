# Creature Cam - Quick Start Guide

## First Time Setup

1. **Navigate to the project directory:**
```bash
cd /Users/asago/clients/creature-cam
```

2. **Activate the virtual environment:**
```bash
source venv/bin/activate
```

3. **Run the application:**
```bash
cd src
python3 main.py
```

## Controls

Once running, use these keyboard shortcuts:

| Key | Action |
|-----|--------|
| `q` | Quit |
| `v` | **Toggle VTUBER mode** 🎭 (full character with static background) |
| `p` | **Toggle PUPPETEER mode** (face swapping with expression transfer) |
| `l` | Toggle landmark visualization |
| `s` | Toggle stats display |
| `1` | Switch to first creature (crab) - texture mode only |
| `2` | Switch to second creature (octopus) - texture mode only |
| `+` or `=` | Increase transformation intensity - texture mode only |
| `-` | Decrease transformation intensity - texture mode only |

## ⭐ VTuber Mode vs Puppeteer Mode vs Texture Mode

**VTUBER MODE** (Press `v`) - **🎭 NEW & RECOMMENDED!**:
- **Full character transformation** with static background
- Entire character body displayed (not just face)
- Character's face animates with your expressions
- **100% head coverage** - your real face is never visible
- Static background replaces webcam feed
- **This is the ultimate Trump-Crab experience!**

**PUPPETEER MODE** (Press `p`):
- Trump's face **moves with your expressions**
- His mouth opens when you talk
- Eyebrows move, eyes blink with yours
- Full face swapping with expression transfer
- Webcam background remains visible

**TEXTURE MODE** (Default when face not detected):
- Static texture overlay on your face
- No expression transfer
- Legacy mode from initial implementation

## Testing Components Individually

### Test Camera Only
```bash
source venv/bin/activate
cd src
python3 camera_capture.py
```
Press `q` to quit.

### Test Face Tracking
```bash
source venv/bin/activate
cd src
python3 face_tracker.py
```
- Press `q` to quit
- Press `m` to toggle mesh visualization
- Press `c` to toggle contour visualization

## Getting Trump-Creature Hybrid Textures

The default textures are basic patterns. For realistic Trump-Crab hybrid effects like CreatureBox:

### Extract from CreatureBox Videos

1. **Run the extraction tool:**
```bash
cd src
python3 extract_creature_textures.py
```

2. **Select a Trump hybrid video** and extract frames

3. **Choose the best frame** with clear creature features

4. **Edit in image software** (Photoshop/GIMP/Preview):
   - Crop to creature head/face
   - Increase saturation for vivid creature colors
   - Save as PNG: `textures/trump-crab/skin.png`

5. **Restart creature-cam** - press `3` for your new texture

### Pro Tips for Better Results

- **Increase intensity to 90-100%** (press `+` key 5-10 times) for dramatic transformation
- Use frames with bright, saturated creature colors
- Crab: orange/red shell patterns work best
- Octopus: purple/pink tentacle textures

## Troubleshooting

**"ModuleNotFoundError"**: Make sure you activated the virtual environment
```bash
source venv/bin/activate
```

**"Camera not found"**: Grant camera permissions in System Preferences > Security & Privacy > Camera

**Low FPS**: Close other applications using the camera or GPU

## Next Steps

- Try adjusting transformation intensity with `+` and `-` keys
- Add your own creature textures from CreatureBox videos
- Experiment with different creatures

For full documentation, see [README.md](README.md)
