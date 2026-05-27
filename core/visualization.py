"""Lightweight image visualization helpers for the MVP."""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np


def resize_for_display(image: np.ndarray, max_width: int = 1200) -> np.ndarray:
    """Resize an OpenCV image while preserving the aspect ratio."""
    if image is None:
        return image

    height, width = image.shape[:2]
    if width <= max_width:
        return image

    scale = max_width / float(width)
    new_height = int(height * scale)
    return cv2.resize(image, (max_width, new_height), interpolation=cv2.INTER_AREA)


def add_status_banner(
    image: np.ndarray,
    title: str,
    subtitle: str,
    success: bool = True,
) -> np.ndarray:
    """Overlay a polished banner on the top of a frame."""
    if image is None:
        return image

    canvas = image.copy()
    height, width = canvas.shape[:2]
    banner_height = max(80, int(height * 0.12))
    overlay = canvas.copy()

    banner_color = (28, 116, 255) if success else (35, 35, 35)
    cv2.rectangle(overlay, (0, 0), (width, banner_height), banner_color, -1)
    canvas = cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0)

    cv2.putText(
        canvas,
        title,
        (24, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        subtitle,
        (24, 63),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (235, 242, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _point(result, landmark: mp.solutions.pose.PoseLandmark):
    if result.pose_landmarks is None:
        return None
    point = result.pose_landmarks.landmark[landmark.value]
    if getattr(point, "visibility", 0.0) < 0.25:
        return None
    height, width = result.image_bgr.shape[:2]
    return int(point.x * width), int(point.y * height)


def _draw_label(canvas, text, origin, color):
    x, y = origin
    width = max(120, len(text) * 10)
    cv2.rectangle(canvas, (x, y - 24), (x + width, y + 8), color, -1)
    cv2.putText(canvas, text, (x + 8, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def draw_measurement_overlay(image: np.ndarray, result, estimate, label: str) -> np.ndarray:
    """Annotate chest, waist, hip, and arm regions on an image."""
    if image is None:
        return image

    canvas = image.copy()
    pose = mp.solutions.pose.PoseLandmark

    chest_left = _point(result, pose.LEFT_SHOULDER)
    chest_right = _point(result, pose.RIGHT_SHOULDER)
    hip_left = _point(result, pose.LEFT_HIP)
    hip_right = _point(result, pose.RIGHT_HIP)
    left_elbow = _point(result, pose.LEFT_ELBOW)
    left_wrist = _point(result, pose.LEFT_WRIST)
    right_elbow = _point(result, pose.RIGHT_ELBOW)
    right_wrist = _point(result, pose.RIGHT_WRIST)

    def lerp(a, b, ratio):
        return int(a[0] + (b[0] - a[0]) * ratio), int(a[1] + (b[1] - a[1]) * ratio)

    if chest_left and chest_right:
        chest_y = int((chest_left[1] + chest_right[1]) / 2) - 18
        cv2.line(canvas, (chest_left[0], chest_y), (chest_right[0], chest_y), (37, 99, 235), 4)
        _draw_label(canvas, f"Chest {estimate.chest_cm:.1f} cm", (min(chest_left[0], chest_right[0]), chest_y - 8), (37, 99, 235))

    if chest_left and hip_left and chest_right and hip_right:
        waist_left = lerp(chest_left, hip_left, 0.58)
        waist_right = lerp(chest_right, hip_right, 0.58)
        waist_y = int((waist_left[1] + waist_right[1]) / 2)
        cv2.line(canvas, (waist_left[0], waist_y), (waist_right[0], waist_y), (14, 165, 233), 4)
        _draw_label(canvas, f"Waist {estimate.waist_cm:.1f} cm", (min(waist_left[0], waist_right[0]), waist_y - 8), (14, 165, 233))

    if hip_left and hip_right:
        hip_y = int((hip_left[1] + hip_right[1]) / 2) + 10
        cv2.line(canvas, (hip_left[0], hip_y), (hip_right[0], hip_y), (16, 185, 129), 4)
        _draw_label(canvas, f"Hip {estimate.hip_cm:.1f} cm", (min(hip_left[0], hip_right[0]), hip_y - 8), (16, 185, 129))

    arm_points = [
        (_point(result, pose.LEFT_SHOULDER), left_elbow, left_wrist, (245, 158, 11)),
        (_point(result, pose.RIGHT_SHOULDER), right_elbow, right_wrist, (139, 92, 246)),
    ]
    arm_origin = None
    for shoulder, elbow, wrist, color in arm_points:
        if shoulder and elbow and wrist:
            cv2.line(canvas, shoulder, elbow, color, 4)
            cv2.line(canvas, elbow, wrist, color, 4)
            arm_origin = shoulder
            break
        if shoulder and wrist:
            cv2.line(canvas, shoulder, wrist, color, 4)
            arm_origin = shoulder
            break

    if arm_origin is not None:
        _draw_label(canvas, f"Arm {estimate.arm_length_cm:.1f} cm", (arm_origin[0], max(30, arm_origin[1] - 32)), (245, 158, 11))

    cv2.rectangle(canvas, (18, 18), (220, 126), (10, 16, 28), -1)
    cv2.rectangle(canvas, (18, 18), (220, 126), (255, 255, 255), 1)
    cv2.putText(canvas, label, (34, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Confidence {estimate.confidence:.2f}", (34, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (203, 213, 225), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Chest | Waist | Hip | Arm", (34, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (148, 163, 184), 1, cv2.LINE_AA)
    return canvas
