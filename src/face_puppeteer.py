"""
Face puppeteering module - Real-time face swapping with expression transfer
Warps a source face (e.g., Trump) to match target facial expressions
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from face_tracker import FaceLandmarks


class FacePuppet:
    """Represents a source face that can be puppeteered"""

    def __init__(self, image_path: Path):
        """
        Load a face image to use as puppet

        Args:
            image_path: Path to source face image
        """
        self.image_path = image_path
        self.image = cv2.imread(str(image_path))

        if self.image is None:
            raise ValueError(f"Could not load image: {image_path}")

        self.landmarks: Optional[FaceLandmarks] = None
        self.triangulation: Optional[np.ndarray] = None
        self.extended_landmarks: Optional[np.ndarray] = None  # Includes virtual points for full head

        print(f"Loaded puppet face: {image_path.name}")

    def set_landmarks(self, landmarks: FaceLandmarks):
        """
        Set facial landmarks for the puppet

        Args:
            landmarks: FaceLandmarks detected on the puppet image
        """
        self.landmarks = landmarks

        # Compute Delaunay triangulation for face warping
        self._compute_triangulation()

    def _compute_triangulation(self):
        """Compute Delaunay triangulation of face landmarks"""
        if self.landmarks is None:
            return

        # Get image dimensions
        h, w = self.image.shape[:2]

        # Start with face landmarks
        points = self.landmarks.image_landmarks.copy()

        # Add extended points for full head coverage (hair, forehead, neck)
        points = self._add_extended_head_points(points, w, h)

        # Store extended landmarks for later use
        self.extended_landmarks = points

        # Create rectangle to bound the triangulation
        rect = (0, 0, w, h)

        # Compute Delaunay triangulation
        subdiv = cv2.Subdiv2D(rect)

        for point in points:
            subdiv.insert(tuple(point.astype(float)))

        # Get triangles
        triangles = subdiv.getTriangleList()

        # Convert to indices (match points to landmark indices)
        triangle_indices = []

        for t in triangles:
            # Triangle vertices
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])

            # Find indices in landmarks
            idx1 = self._find_point_index(pt1, points)
            idx2 = self._find_point_index(pt2, points)
            idx3 = self._find_point_index(pt3, points)

            if idx1 is not None and idx2 is not None and idx3 is not None:
                triangle_indices.append([idx1, idx2, idx3])

        self.triangulation = np.array(triangle_indices)
        print(f"Computed {len(self.triangulation)} triangles for face mesh")

    def _add_extended_head_points(self, points: np.ndarray, width: int, height: int) -> np.ndarray:
        """
        Add virtual landmarks beyond the face to extend coverage to hair, forehead, and neck

        Args:
            points: Original face landmarks
            width: Image width
            height: Image height

        Returns:
            Extended landmarks array with additional points for full head coverage
        """
        # Get face oval points for reference
        face_oval = points[self.landmarks.face_oval_indices]

        # Calculate face boundaries
        face_left = face_oval[:, 0].min()
        face_right = face_oval[:, 0].max()
        face_top = face_oval[:, 1].min()
        face_bottom = face_oval[:, 1].max()
        face_center_x = (face_left + face_right) / 2
        face_width = face_right - face_left
        face_height = face_bottom - face_top

        # Extension distances
        forehead_extend = int(face_height * 0.6)  # Extend upward for hair/forehead
        temple_extend = int(face_width * 0.25)    # Extend sideways for ears/hair
        neck_extend = int(face_height * 0.4)      # Extend downward for neck

        extended_points = []

        # 1. FOREHEAD/HAIRLINE POINTS - Arc above face
        num_forehead_pts = 9
        forehead_y = max(0, face_top - forehead_extend)
        for i in range(num_forehead_pts):
            # Arc from left temple to right temple
            t = i / (num_forehead_pts - 1)
            x = face_left - temple_extend + t * (face_width + 2 * temple_extend)
            # Slight arc (parabola)
            arc_offset = int(forehead_extend * 0.2 * (1 - (2*t - 1)**2))
            y = forehead_y - arc_offset
            extended_points.append([x, max(0, y)])

        # 2. TEMPLE POINTS - Left and right sides
        num_temple_pts = 5
        for i in range(num_temple_pts):
            t = i / (num_temple_pts - 1)
            y = face_top + t * face_height

            # Left temple
            extended_points.append([max(0, face_left - temple_extend), y])
            # Right temple
            extended_points.append([min(width-1, face_right + temple_extend), y])

        # 3. NECK POINTS - Below chin
        num_neck_pts = 7
        neck_y = min(height-1, face_bottom + neck_extend)
        for i in range(num_neck_pts):
            t = i / (num_neck_pts - 1)
            x = face_left - temple_extend*0.5 + t * (face_width + temple_extend)
            extended_points.append([x, neck_y])

        # 4. TOP OF HEAD POINTS - For hair coverage
        num_top_pts = 5
        top_y = max(0, forehead_y - forehead_extend * 0.3)
        for i in range(num_top_pts):
            t = i / (num_top_pts - 1)
            x = face_left + t * face_width
            extended_points.append([x, top_y])

        # 5. CORNER POINTS - Ensure edges are covered
        edge_margin = 5
        corners = [
            [edge_margin, edge_margin],  # Top-left
            [width - edge_margin, edge_margin],  # Top-right
            [edge_margin, height - edge_margin],  # Bottom-left
            [width - edge_margin, height - edge_margin],  # Bottom-right
            [face_center_x, edge_margin],  # Top-center
            [face_center_x, height - edge_margin],  # Bottom-center
        ]
        extended_points.extend(corners)

        # Combine original landmarks with extended points
        extended_array = np.vstack([points, np.array(extended_points, dtype=np.float32)])

        print(f"Extended landmarks: {len(points)} original + {len(extended_points)} virtual = {len(extended_array)} total")

        return extended_array

    def _find_point_index(self, pt: Tuple[float, float], points: np.ndarray, threshold=2.0) -> Optional[int]:
        """Find the index of a point in the landmarks array"""
        for i, landmark_pt in enumerate(points):
            dist = np.linalg.norm(landmark_pt - np.array(pt))
            if dist < threshold:
                return i
        return None


class FacePuppeteer:
    """Real-time face puppeteering engine"""

    def __init__(self, puppet_path: Path):
        """
        Initialize face puppeteering

        Args:
            puppet_path: Path to source face image (e.g., Trump face)
        """
        self.puppet = FacePuppet(puppet_path)
        self.puppet_landmarks_set = False

    def set_puppet_landmarks(self, landmarks: FaceLandmarks):
        """
        Set facial landmarks for the puppet face

        Args:
            landmarks: FaceLandmarks detected on the puppet image
        """
        self.puppet.set_landmarks(landmarks)
        self.puppet_landmarks_set = True
        print("Puppet landmarks configured")

    def _generate_target_extended_landmarks(self,
                                            target_points: np.ndarray,
                                            target_landmarks: FaceLandmarks,
                                            width: int,
                                            height: int) -> np.ndarray:
        """
        Generate extended landmarks for target face (mirrors puppet extension)

        Args:
            target_points: Original 478 MediaPipe landmarks
            target_landmarks: FaceLandmarks object for accessing indices
            width: Frame width
            height: Frame height

        Returns:
            Extended landmarks matching puppet structure
        """
        # Get face oval points for reference
        face_oval = target_points[target_landmarks.face_oval_indices]

        # Calculate face boundaries
        face_left = face_oval[:, 0].min()
        face_right = face_oval[:, 0].max()
        face_top = face_oval[:, 1].min()
        face_bottom = face_oval[:, 1].max()
        face_center_x = (face_left + face_right) / 2
        face_width = face_right - face_left
        face_height = face_bottom - face_top

        # Extension distances (same ratios as puppet)
        forehead_extend = int(face_height * 0.6)
        temple_extend = int(face_width * 0.25)
        neck_extend = int(face_height * 0.4)

        extended_points = []

        # 1. FOREHEAD/HAIRLINE POINTS
        num_forehead_pts = 9
        forehead_y = max(0, face_top - forehead_extend)
        for i in range(num_forehead_pts):
            t = i / (num_forehead_pts - 1)
            x = face_left - temple_extend + t * (face_width + 2 * temple_extend)
            arc_offset = int(forehead_extend * 0.2 * (1 - (2*t - 1)**2))
            y = forehead_y - arc_offset
            extended_points.append([x, max(0, y)])

        # 2. TEMPLE POINTS
        num_temple_pts = 5
        for i in range(num_temple_pts):
            t = i / (num_temple_pts - 1)
            y = face_top + t * face_height
            extended_points.append([max(0, face_left - temple_extend), y])
            extended_points.append([min(width-1, face_right + temple_extend), y])

        # 3. NECK POINTS
        num_neck_pts = 7
        neck_y = min(height-1, face_bottom + neck_extend)
        for i in range(num_neck_pts):
            t = i / (num_neck_pts - 1)
            x = face_left - temple_extend*0.5 + t * (face_width + temple_extend)
            extended_points.append([x, neck_y])

        # 4. TOP OF HEAD POINTS
        num_top_pts = 5
        top_y = max(0, forehead_y - forehead_extend * 0.3)
        for i in range(num_top_pts):
            t = i / (num_top_pts - 1)
            x = face_left + t * face_width
            extended_points.append([x, top_y])

        # 5. CORNER POINTS
        edge_margin = 5
        corners = [
            [edge_margin, edge_margin],
            [width - edge_margin, edge_margin],
            [edge_margin, height - edge_margin],
            [width - edge_margin, height - edge_margin],
            [face_center_x, edge_margin],
            [face_center_x, height - edge_margin],
        ]
        extended_points.extend(corners)

        # Combine original with extended
        extended_array = np.vstack([target_points, np.array(extended_points, dtype=np.float32)])

        return extended_array

    def apply_puppeteering(self,
                          frame: np.ndarray,
                          target_landmarks: FaceLandmarks) -> np.ndarray:
        """
        Apply face puppeteering - warp puppet face to match target expressions

        Args:
            frame: Current video frame
            target_landmarks: User's face landmarks

        Returns:
            Frame with puppet face applied
        """
        if not self.puppet_landmarks_set or self.puppet.triangulation is None:
            return frame

        # Generate extended landmarks for target face (same structure as puppet)
        h, w = frame.shape[:2]
        target_extended = self._generate_target_extended_landmarks(
            target_landmarks.image_landmarks,
            target_landmarks,
            w, h
        )

        # Start with the original frame to avoid black regions
        warped_face = frame.copy()

        # Warp puppet face onto the frame using EXTENDED landmarks
        self._warp_face_inplace(
            warped_face,
            self.puppet.image,
            self.puppet.extended_landmarks,  # Use extended source
            target_extended,                 # Use extended target
            self.puppet.triangulation
        )

        # Create full head mask for complete coverage (face + hair + ears + neck)
        # Use the enhanced mask from face_tracker for 100% coverage
        from face_tracker import FaceTracker
        temp_tracker = FaceTracker.__new__(FaceTracker)
        mask = temp_tracker.create_full_head_mask(frame, target_landmarks)

        # Seamless clone puppet face onto target frame
        output = self._seamless_blend(frame, warped_face, mask, target_landmarks)

        return output

    def _warp_face_inplace(self,
                           dst_image: np.ndarray,
                           src_image: np.ndarray,
                           src_landmarks: np.ndarray,
                           dst_landmarks: np.ndarray,
                           triangulation: np.ndarray):
        """
        Warp source face directly onto destination image (in-place)

        Args:
            dst_image: Destination image (modified in-place)
            src_image: Source face image
            src_landmarks: Source face landmarks
            dst_landmarks: Destination face landmarks
            triangulation: Triangle indices
        """
        for tri_indices in triangulation:
            # Source triangle points
            src_tri = src_landmarks[tri_indices].astype(np.float32)

            # Destination triangle points
            dst_tri = dst_landmarks[tri_indices].astype(np.float32)

            # Warp triangle directly onto dst_image
            self._warp_triangle(src_image, dst_image, src_tri, dst_tri)

    def _warp_face(self,
                   src_image: np.ndarray,
                   src_landmarks: np.ndarray,
                   dst_landmarks: np.ndarray,
                   triangulation: np.ndarray,
                   dst_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Warp source face to match destination landmarks using triangulation

        Args:
            src_image: Source face image
            src_landmarks: Source face landmarks
            dst_landmarks: Destination face landmarks
            triangulation: Triangle indices
            dst_shape: Destination image shape

        Returns:
            Warped face image
        """
        warped = np.zeros(dst_shape, dtype=np.uint8)

        for tri_indices in triangulation:
            # Source triangle points
            src_tri = src_landmarks[tri_indices].astype(np.float32)

            # Destination triangle points
            dst_tri = dst_landmarks[tri_indices].astype(np.float32)

            # Warp triangle
            self._warp_triangle(src_image, warped, src_tri, dst_tri)

        return warped

    def _warp_triangle(self,
                      src_image: np.ndarray,
                      dst_image: np.ndarray,
                      src_tri: np.ndarray,
                      dst_tri: np.ndarray):
        """
        Warp a single triangle from source to destination

        Args:
            src_image: Source image
            dst_image: Destination image (modified in-place)
            src_tri: Source triangle vertices (3x2)
            dst_tri: Destination triangle vertices (3x2)
        """
        # Ensure float32 type
        src_tri = src_tri.astype(np.float32)
        dst_tri = dst_tri.astype(np.float32)

        # Get bounding rectangles
        src_rect = cv2.boundingRect(src_tri.astype(np.int32))
        dst_rect = cv2.boundingRect(dst_tri.astype(np.int32))

        # Offset points by top-left corner of rectangle
        src_tri_cropped = src_tri - np.array([src_rect[0], src_rect[1]], dtype=np.float32)
        dst_tri_cropped = dst_tri - np.array([dst_rect[0], dst_rect[1]], dtype=np.float32)

        # Crop source rectangle
        src_cropped = src_image[src_rect[1]:src_rect[1] + src_rect[3],
                                src_rect[0]:src_rect[0] + src_rect[2]]

        if src_cropped.size == 0 or dst_rect[2] <= 0 or dst_rect[3] <= 0:
            return

        # Convert RGBA to BGR if needed (for character images with alpha channel)
        if src_cropped.shape[2] == 4:
            src_cropped = src_cropped[:, :, :3]

        # Compute affine transform
        warp_mat = cv2.getAffineTransform(src_tri_cropped, dst_tri_cropped)

        # Warp source crop to destination
        dst_cropped = cv2.warpAffine(
            src_cropped,
            warp_mat,
            (dst_rect[2], dst_rect[3]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )

        # Create mask for triangle
        mask = np.zeros((dst_rect[3], dst_rect[2]), dtype=np.uint8)
        cv2.fillConvexPoly(mask, dst_tri_cropped.astype(np.int32), 255)

        # Copy warped triangle to destination
        dst_y1 = dst_rect[1]
        dst_y2 = dst_rect[1] + dst_rect[3]
        dst_x1 = dst_rect[0]
        dst_x2 = dst_rect[0] + dst_rect[2]

        # Bounds checking
        h, w = dst_image.shape[:2]
        if dst_y1 < 0 or dst_y2 > h or dst_x1 < 0 or dst_x2 > w:
            return

        # Handle both BGR and RGBA destination images
        num_channels = dst_image.shape[2] if len(dst_image.shape) > 2 else 1

        if num_channels == 4:
            # RGBA destination - use 4 channel mask
            mask_4ch = cv2.merge([mask, mask, mask, mask]) / 255.0
            # Ensure dst_cropped has alpha channel
            if dst_cropped.shape[2] == 3:
                alpha = np.ones((dst_cropped.shape[0], dst_cropped.shape[1], 1), dtype=np.uint8) * 255
                dst_cropped = np.concatenate([dst_cropped, alpha], axis=2)

            dst_image[dst_y1:dst_y2, dst_x1:dst_x2] = (
                dst_image[dst_y1:dst_y2, dst_x1:dst_x2] * (1 - mask_4ch) +
                dst_cropped * mask_4ch
            ).astype(np.uint8)
        else:
            # BGR destination - use 3 channel mask
            mask_3ch = cv2.merge([mask, mask, mask]) / 255.0
            # Ensure dst_cropped is BGR
            if dst_cropped.shape[2] == 4:
                dst_cropped = dst_cropped[:, :, :3]

            dst_image[dst_y1:dst_y2, dst_x1:dst_x2] = (
                dst_image[dst_y1:dst_y2, dst_x1:dst_x2] * (1 - mask_3ch) +
                dst_cropped * mask_3ch
            ).astype(np.uint8)

    def _create_face_mask(self, landmarks: FaceLandmarks, shape: Tuple[int, int]) -> np.ndarray:
        """
        Create a mask for the face region

        Args:
            landmarks: Face landmarks
            shape: Image shape (height, width)

        Returns:
            Binary mask
        """
        mask = np.zeros(shape, dtype=np.uint8)

        # Use face oval
        face_points = landmarks.image_landmarks[landmarks.face_oval_indices]
        cv2.fillPoly(mask, [face_points], 255)

        # Feather the mask for smoother blending
        mask = cv2.GaussianBlur(mask, (15, 15), 10)

        return mask

    def _seamless_blend(self,
                       background: np.ndarray,
                       foreground: np.ndarray,
                       mask: np.ndarray,
                       landmarks: FaceLandmarks) -> np.ndarray:
        """
        Seamlessly blend foreground onto background using mask

        Args:
            background: Background image (target frame)
            foreground: Foreground image (warped puppet face)
            mask: Blending mask
            landmarks: Face landmarks for center point

        Returns:
            Blended image
        """
        # Get center of face for seamless clone
        face_points = landmarks.image_landmarks[landmarks.face_oval_indices]
        center = tuple(np.mean(face_points, axis=0).astype(int))

        try:
            # Use seamless clone for natural blending
            output = cv2.seamlessClone(
                foreground,
                background,
                mask,
                center,
                cv2.NORMAL_CLONE
            )
        except cv2.error:
            # Fallback to simple alpha blending
            mask_3ch = cv2.merge([mask, mask, mask]) / 255.0
            output = (foreground * mask_3ch + background * (1 - mask_3ch)).astype(np.uint8)

        return output


def test_face_puppeteering():
    """Test face puppeteering with a reference image"""
    from camera_capture import CameraCapture
    from face_tracker import FaceTracker
    import time

    print("=" * 70)
    print("Face Puppeteering Test")
    print("=" * 70)

    # Check for puppet reference image
    textures_dir = Path(__file__).parent.parent / "textures" / "trump-crab"
    puppet_path = textures_dir / "skin.png"

    if not puppet_path.exists():
        print(f"❌ Puppet image not found: {puppet_path}")
        print("Please create a Trump reference face image first")
        return

    print(f"\nUsing puppet: {puppet_path}")

    # Initialize components
    camera = CameraCapture(camera_id=1)
    if not camera.start():
        print("Failed to start camera")
        return

    tracker = FaceTracker()
    puppeteer = FacePuppeteer(puppet_path)

    # Detect landmarks on puppet image (one-time setup)
    print("\nDetecting landmarks on puppet image...")
    puppet_landmarks = tracker.process_frame(puppeteer.puppet.image)

    if puppet_landmarks is None:
        print("❌ No face detected in puppet image")
        print("Make sure the puppet image contains a clear face")
        camera.release()
        tracker.release()
        return

    puppeteer.set_puppet_landmarks(puppet_landmarks)
    print("✅ Puppet configured successfully")

    print("\n" + "=" * 70)
    print("Press 'q' to quit")
    print("=" * 70 + "\n")

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            success, frame = camera.read_frame()
            if not success:
                break

            frame_count += 1

            # Track user's face
            target_landmarks = tracker.process_frame(frame)

            if target_landmarks is not None:
                # Apply puppeteering
                frame = puppeteer.apply_puppeteering(frame, target_landmarks)

                cv2.putText(frame, "PUPPET ACTIVE", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show FPS
            elapsed = time.time() - start_time
            if elapsed > 0:
                fps = frame_count / elapsed
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Face Puppeteering", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        camera.release()
        tracker.release()
        cv2.destroyAllWindows()

        # Stats
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"\nTest complete:")
        print(f"  Average FPS: {avg_fps:.1f}")


if __name__ == "__main__":
    test_face_puppeteering()
