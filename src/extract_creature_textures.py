"""
Extract creature textures from CreatureBox videos
Helps create high-quality textures for real-time transformation
"""

import cv2
import numpy as np
from pathlib import Path
import sys


def extract_frames_from_video(video_path, output_dir, num_frames=10):
    """
    Extract evenly-spaced frames from a video

    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        num_frames: Number of frames to extract
    """
    print(f"\nExtracting frames from: {video_path.name}")

    # Open video
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return

    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    print(f"   Total frames: {total_frames}")
    print(f"   Duration: {duration:.2f}s")
    print(f"   FPS: {fps:.1f}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate frame interval
    interval = total_frames // num_frames if total_frames > num_frames else 1

    extracted_count = 0

    for i in range(num_frames):
        frame_num = i * interval

        if frame_num >= total_frames:
            break

        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if not ret:
            continue

        # Save frame
        output_path = output_dir / f"frame_{i:03d}.jpg"
        cv2.imwrite(str(output_path), frame)
        extracted_count += 1

    cap.release()

    print(f"   ✅ Extracted {extracted_count} frames to: {output_dir}")
    return extracted_count


def show_frames_for_selection(frames_dir):
    """
    Display extracted frames in a grid for manual selection

    Args:
        frames_dir: Directory containing extracted frames
    """
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))

    if not frame_files:
        print("No frames found")
        return

    print(f"\nDisplaying {len(frame_files)} frames...")
    print("Press any key to cycle through frames")
    print("Press 'q' to quit")
    print("Note the frame number of the best creature close-up")

    for i, frame_path in enumerate(frame_files):
        frame = cv2.imread(str(frame_path))

        if frame is None:
            continue

        # Resize for display
        h, w = frame.shape[:2]
        scale = 800 / w
        display_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        # Add frame info
        cv2.putText(display_frame, f"Frame {i:03d} - {frame_path.name}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display_frame, "Press any key for next, 'q' to quit",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Frame Selection", display_frame)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


def main():
    """Extract frames from CreatureBox Trump hybrid videos"""

    print("=" * 70)
    print("Creature Texture Extraction Tool")
    print("=" * 70)

    # CreatureBox video directory
    creaturebox_dir = Path("/Users/asago/clients/creaturebox/src")

    # Find Trump hybrid videos
    trump_videos = sorted(creaturebox_dir.glob("trump_hybrid_*.mp4"))

    if not trump_videos:
        print("\n❌ No Trump hybrid videos found in CreatureBox")
        return

    print(f"\nFound {len(trump_videos)} Trump hybrid videos:")
    for i, video in enumerate(trump_videos, 1):
        print(f"  {i}. {video.name}")

    print("\nSelect a video (or 'all' to extract from all):")
    print("Enter number (1-{}), 'all', or 'q' to quit: ".format(len(trump_videos)), end="")

    choice = input().strip().lower()

    if choice == 'q':
        return

    videos_to_process = []

    if choice == 'all':
        videos_to_process = trump_videos
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(trump_videos):
                videos_to_process = [trump_videos[idx]]
            else:
                print("Invalid selection")
                return
        except ValueError:
            print("Invalid input")
            return

    # Extract frames from selected videos
    creature_cam_dir = Path("/Users/asago/clients/creature-cam")
    extracted_frames_dir = creature_cam_dir / "extracted_frames"

    for video in videos_to_process:
        video_output_dir = extracted_frames_dir / video.stem
        frames_extracted = extract_frames_from_video(video, video_output_dir, num_frames=10)

        if frames_extracted and frames_extracted > 0:
            # Show frames for selection
            print("\nWould you like to preview these frames? (y/n): ", end="")
            preview = input().strip().lower()

            if preview == 'y':
                show_frames_for_selection(video_output_dir)

    print("\n" + "=" * 70)
    print("Extraction complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Review extracted frames in: {}".format(extracted_frames_dir))
    print("2. Choose the best frame showing clear creature features")
    print("3. Use image editing software to:")
    print("   - Crop to just the creature head/face")
    print("   - Remove background (create alpha channel)")
    print("   - Isolate creature features (shell, claws, eyes, etc.)")
    print("4. Save processed textures to:")
    print("   /Users/asago/clients/creature-cam/textures/[creature-name]/")
    print("\nFor example:")
    print("   textures/trump-crab/skin.png")
    print("   textures/trump-crab/eyes.png")
    print("   textures/trump-crab/claws.png")


if __name__ == "__main__":
    main()
