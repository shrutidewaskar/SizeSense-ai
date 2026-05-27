"""Heuristic measurement estimation built on MediaPipe pose landmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

import mediapipe as mp
import numpy as np

from .pose_detector import PoseDetectionResult


PoseLandmark = mp.solutions.pose.PoseLandmark


@dataclass
class MeasurementEstimate:
    """Final prototype measurements and supporting metadata."""

    chest_cm: float
    waist_cm: float
    hip_cm: float
    arm_length_cm: float
    confidence: float
    notes: List[str] = field(default_factory=list)
    chest_width_cm: float = 0.0
    waist_width_cm: float = 0.0
    hip_width_cm: float = 0.0


def _landmark(result: PoseDetectionResult, landmark: PoseLandmark) -> Optional[object]:
    if result.pose_landmarks is None:
        return None
    return result.pose_landmarks.landmark[landmark.value]


def _point_xy(result: PoseDetectionResult, landmark: PoseLandmark) -> Optional[Tuple[float, float]]:
    point = _landmark(result, landmark)
    if point is None:
        return None
    return float(point.x), float(point.y)


def _visible(point: object, threshold: float = 0.35) -> bool:
    return point is not None and getattr(point, "visibility", 0.0) >= threshold


def _distance_cm(result: PoseDetectionResult, left: PoseLandmark, right: PoseLandmark, scale: float) -> Optional[float]:
    left_point = _point_xy(result, left)
    right_point = _point_xy(result, right)
    if left_point is None or right_point is None:
        return None
    return float(np.linalg.norm(np.subtract(left_point, right_point)) * scale)


def _body_height_px(result: PoseDetectionResult) -> Optional[float]:
    top_candidates = [
        PoseLandmark.NOSE,
        PoseLandmark.LEFT_EYE_INNER,
        PoseLandmark.RIGHT_EYE_INNER,
        PoseLandmark.LEFT_EAR,
        PoseLandmark.RIGHT_EAR,
        PoseLandmark.LEFT_SHOULDER,
        PoseLandmark.RIGHT_SHOULDER,
    ]
    bottom_candidates = [
        PoseLandmark.LEFT_ANKLE,
        PoseLandmark.RIGHT_ANKLE,
        PoseLandmark.LEFT_HEEL,
        PoseLandmark.RIGHT_HEEL,
        PoseLandmark.LEFT_FOOT_INDEX,
        PoseLandmark.RIGHT_FOOT_INDEX,
    ]

    top_points = []
    bottom_points = []

    for landmark in top_candidates:
        point = _landmark(result, landmark)
        if _visible(point):
            top_points.append(point.y)

    for landmark in bottom_candidates:
        point = _landmark(result, landmark)
        if _visible(point):
            bottom_points.append(point.y)

    if not top_points or not bottom_points:
        return None

    height_px = (max(bottom_points) - min(top_points)) * result.image_bgr.shape[0]
    if height_px <= 0:
        return None
    return float(height_px)


def _view_scale_cm_per_px(result: PoseDetectionResult, height_cm: float) -> Optional[float]:
    body_height_px = _body_height_px(result)
    if body_height_px is None or body_height_px <= 0:
        return None
    return float(height_cm / body_height_px)


def _average(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _ellipse_circumference(width_cm: float, depth_cm: float) -> float:
    semi_major = max(width_cm, 0.0) / 2.0
    semi_minor = max(depth_cm, 0.0) / 2.0
    if semi_major <= 0 or semi_minor <= 0:
        return 0.0
    return float(np.pi * (3 * (semi_major + semi_minor) - np.sqrt((3 * semi_major + semi_minor) * (semi_major + 3 * semi_minor))))


def _arm_length_cm(result: PoseDetectionResult, side: str, scale: float) -> Optional[float]:
    if side == "left":
        shoulder = PoseLandmark.LEFT_SHOULDER
        elbow = PoseLandmark.LEFT_ELBOW
        wrist = PoseLandmark.LEFT_WRIST
    else:
        shoulder = PoseLandmark.RIGHT_SHOULDER
        elbow = PoseLandmark.RIGHT_ELBOW
        wrist = PoseLandmark.RIGHT_WRIST

    shoulder_point = _point_xy(result, shoulder)
    elbow_point = _point_xy(result, elbow)
    wrist_point = _point_xy(result, wrist)
    if shoulder_point is None or wrist_point is None:
        return None

    if elbow_point is not None:
        arm_px = np.linalg.norm(np.subtract(shoulder_point, elbow_point)) + np.linalg.norm(np.subtract(elbow_point, wrist_point))
    else:
        arm_px = np.linalg.norm(np.subtract(shoulder_point, wrist_point)) * 0.94

    return float(arm_px * scale)


def _safe_blend(base_value: float, landmark_value: Optional[float], blend: float = 0.45) -> float:
    if landmark_value is None or landmark_value <= 0:
        return base_value
    return float((1.0 - blend) * base_value + blend * landmark_value)


def estimate_measurements(front_result: PoseDetectionResult, side_result: PoseDetectionResult, height_cm: float) -> MeasurementEstimate:
    """Estimate chest, waist, hip circumference, and arm length using simple geometry.

    The prototype blends height-based baseline ratios with landmark-derived scale factors.
    """

    front_scale = _view_scale_cm_per_px(front_result, height_cm)
    side_scale = _view_scale_cm_per_px(side_result, height_cm)
    scales = [scale for scale in [front_scale, side_scale] if scale is not None]
    scale = _average(scales) if scales else None

    notes: List[str] = []
    if scale is None:
        notes.append("Pose scale fallback used because body height could not be derived reliably from landmarks.")

    base_chest_width = height_cm * 0.255
    base_waist_width = height_cm * 0.225
    base_hip_width = height_cm * 0.275
    base_arm_length = height_cm * 0.355

    shoulder_widths: List[float] = []
    hip_widths: List[float] = []
    arm_lengths: List[float] = []

    if scale is not None:
        for result_scale, result in [(front_scale, front_result), (side_scale, side_result)]:
            if result_scale is None:
                continue

            shoulder_width = _distance_cm(result, PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER, result_scale)
            hip_width = _distance_cm(result, PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP, result_scale)
            left_arm = _arm_length_cm(result, "left", result_scale)
            right_arm = _arm_length_cm(result, "right", result_scale)

            if shoulder_width is not None:
                shoulder_widths.append(shoulder_width)
            if hip_width is not None:
                hip_widths.append(hip_width)
            if left_arm is not None:
                arm_lengths.append(left_arm)
            if right_arm is not None:
                arm_lengths.append(right_arm)

    shoulder_width_cm = _average(shoulder_widths)
    hip_width_cm = _average(hip_widths)
    arm_length_cm = _average(arm_lengths)

    chest_width_cm = _safe_blend(base_chest_width, shoulder_width_cm * 0.95 if shoulder_width_cm else None)
    waist_width_cm = _safe_blend(base_waist_width, ((shoulder_width_cm + hip_width_cm) / 2.0) * 0.74 if shoulder_width_cm and hip_width_cm else None)
    hip_width_cm = _safe_blend(base_hip_width, hip_width_cm * 1.02 if hip_width_cm else None)
    arm_length_cm = _safe_blend(base_arm_length, arm_length_cm)

    if not shoulder_widths:
        notes.append("Shoulder landmarks were partially missing, so chest width was blended with a height-based prior.")
    if not hip_widths:
        notes.append("Hip landmarks were partially missing, so hip width was blended with a height-based prior.")
    if not arm_lengths:
        notes.append("Arm length fell back to a height-based prior because elbow or wrist landmarks were unreliable.")

    if side_scale is not None:
        side_span_cm = _distance_cm(side_result, PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_HIP, side_scale)
        if side_span_cm is None:
            side_span_cm = _distance_cm(side_result, PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_HIP, side_scale)
        depth_factor = 0.68
        if side_span_cm is not None and height_cm > 0:
            depth_factor = float(np.clip(0.62 + (side_span_cm / height_cm) * 1.35, 0.56, 0.88))
    else:
        depth_factor = 0.68

    chest_cm = _ellipse_circumference(chest_width_cm, chest_width_cm * depth_factor)
    waist_cm = _ellipse_circumference(waist_width_cm, waist_width_cm * (depth_factor * 0.92))
    hip_cm = _ellipse_circumference(hip_width_cm, hip_width_cm * (depth_factor * 0.98))

    measurement_confidence = 0.38
    if front_result.detected:
        measurement_confidence += 0.18
    if side_result.detected:
        measurement_confidence += 0.18
    if scale is not None:
        measurement_confidence += 0.12
    if shoulder_widths:
        measurement_confidence += 0.06
    if hip_widths:
        measurement_confidence += 0.04
    measurement_confidence = float(np.clip(measurement_confidence, 0.0, 0.98))

    if not notes:
        notes.append("Measurements are estimated from pose landmarks, height scaling, and ellipse-based cross-section approximations.")

    return MeasurementEstimate(
        chest_cm=float(chest_cm),
        waist_cm=float(waist_cm),
        hip_cm=float(hip_cm),
        arm_length_cm=float(arm_length_cm),
        confidence=measurement_confidence,
        notes=notes,
        chest_width_cm=float(chest_width_cm),
        waist_width_cm=float(waist_width_cm),
        hip_width_cm=float(hip_width_cm),
    )
