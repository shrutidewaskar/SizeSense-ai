"""MediaPipe pose detection for uploaded body images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from .visualization import add_status_banner


@dataclass
class PoseDetectionResult:
    """Container for a pose detection result."""

    image_bgr: np.ndarray
    pose_landmarks: Optional[object]
    detected: bool
    landmark_count: int
    confidence: Optional[float]
    message: str


class PoseDetector:
    """Thin wrapper around MediaPipe Pose for image-based analysis."""

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._mp_pose = mp.solutions.pose
        self._mp_draw = mp.solutions.drawing_utils
        self._drawing_styles = mp.solutions.drawing_styles
        self._pose = self._mp_pose.Pose(
            static_image_mode=True,
            model_complexity=model_complexity,
            enable_segmentation=False,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._pose.close()

    def detect_and_draw(self, image_bgr: np.ndarray, label: str = "Pose detected") -> PoseDetectionResult:
        """Detect pose landmarks and return an annotated BGR image."""
        if image_bgr is None:
            raise ValueError("image_bgr cannot be None")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self._pose.process(image_rgb)
        overlay = image_bgr.copy()

        detected = bool(results.pose_landmarks)
        landmark_count = 0
        confidence = None
        message = "No pose landmarks detected. Try a clearer full-body image."

        if detected:
            landmark_count = len(results.pose_landmarks.landmark)
            key_landmarks = [
                self._mp_pose.PoseLandmark.NOSE.value,
                self._mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                self._mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                self._mp_pose.PoseLandmark.LEFT_HIP.value,
                self._mp_pose.PoseLandmark.RIGHT_HIP.value,
            ]
            visibilities = [results.pose_landmarks.landmark[index].visibility for index in key_landmarks]
            confidence = float(np.mean(visibilities))
            self._mp_draw.draw_landmarks(
                overlay,
                results.pose_landmarks,
                self._mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self._drawing_styles.get_default_pose_landmarks_style(),
            )
            message = f"{landmark_count} landmarks detected with MediaPipe Pose."

        overlay = add_status_banner(overlay, label, message, success=detected)
        return PoseDetectionResult(
            image_bgr=overlay,
            pose_landmarks=results.pose_landmarks,
            detected=detected,
            landmark_count=landmark_count,
            confidence=confidence,
            message=message,
        )
