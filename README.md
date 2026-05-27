# SizeSense AI

SizeSense AI is a prototype human body measurement system that estimates chest circumference, waist circumference, hip circumference, and arm length from a front-view image, a side-view image, and user height.

This submission prioritizes reasoning and implementation over polish. The model is intentionally heuristic and transparent so the engineering choices are easy to review.

## Approach

The pipeline uses MediaPipe Pose to detect body landmarks in both images, OpenCV to render overlays, and height-based scaling to convert pixel distances into approximate centimeters.

The estimator blends:

- Landmark-derived shoulder, hip, and arm geometry
- User height as the primary scale reference
- Side-view torso thickness as a weak depth proxy
- Ellipse-based circumference approximation for chest, waist, and hip

## Pipeline Overview

1. Upload a front-view image, a side-view image, and user height.
2. Detect pose landmarks with MediaPipe Pose for both views.
3. Derive a pixel-to-centimeter scale from the visible body height.
4. Estimate region widths for chest, waist, hip, and arm length from landmarks.
5. Use the side view to adjust depth assumptions.
6. Convert width and depth into circumference using an ellipse perimeter approximation.
7. Render pose overlays, measurement overlays, and metric cards in the Streamlit UI.

## Measurement Logic

- Chest circumference is estimated from a shoulder-width proxy and a depth factor.
- Waist circumference is estimated from an interpolated torso-width proxy between shoulders and hips.
- Hip circumference is estimated from hip width and the same depth framework.
- Arm length is estimated from shoulder, elbow, and wrist landmark chains.

### Assumptions

- The subject is standing roughly upright and facing the camera in both images.
- The images are full-body or close to full-body.
- The height input is accurate.
- Side-view pose landmarks are sufficient to provide a coarse depth proxy.
- Circumference estimates are approximate and intended for prototype validation rather than tailoring-grade accuracy.

## Accuracy Limitations

- Pose landmarks do not directly capture body silhouette thickness.
- Clothing, camera angle, pose, and cropping can affect the result.
- The current side-view depth estimate is heuristic and not a full 3D reconstruction.
- Estimates will degrade if ankles, shoulders, elbows, or wrists are not visible.

## What To Improve Next

- Add silhouette segmentation to improve width and depth estimation.
- Calibrate against labeled ground-truth measurements.
- Add per-measurement confidence scores and error bars.
- Introduce body-profile normalization for pose variation.
- Replace heuristics with a learned regression model once enough data exists.

## Tech Stack

- Streamlit
- MediaPipe
- OpenCV
- NumPy
- Pillow

## Folder Structure

```text
sizesense-ai/
├── app.py
├── requirements.txt
├── README.md
├── core/
│   ├── __init__.py
│   ├── measurement.py
│   ├── pose_detector.py
│   ├── visualization.py
│   └── utils.py
├── assets/
├── sample_images/
└── screenshots/
```

## Installation

1. Create and activate a Python virtual environment.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

## Run

Start the app with:

```bash
streamlit run app.py
```

## Deliverables Included

- Source code
- README/documentation
- Run instructions
- Basic UI/dashboard
- Visual overlays and prototype measurement outputs
"# SizeSense-ai" 
