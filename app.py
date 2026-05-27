from __future__ import annotations

from typing import Optional, Tuple

import streamlit as st
from PIL import Image

from core.measurement import MeasurementEstimate, estimate_measurements
from core.pose_detector import PoseDetectionResult, PoseDetector
from core.utils import bgr_to_pil, load_image, pil_to_bgr, resize_pil_image
from core.visualization import draw_measurement_overlay, resize_for_display


st.set_page_config(
    page_title="SizeSense AI",
    page_icon="📏",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(35, 116, 255, 0.12), transparent 28%),
            radial-gradient(circle at top right, rgba(0, 194, 168, 0.10), transparent 24%),
            linear-gradient(180deg, #f7fbff 0%, #edf4fb 100%);
    }
    .hero {
        padding: 1.5rem 1.6rem 1.2rem 1.6rem;
        border-radius: 22px;
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(16, 24, 40, 0.08);
        box-shadow: 0 18px 50px rgba(16, 24, 40, 0.06);
        margin-bottom: 1rem;
    }
    .eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 0.74rem;
        color: #5f6b85;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.6rem;
        line-height: 1.05;
        color: #0f172a;
    }
    .hero .subtitle {
        margin: 0.55rem 0 0.3rem 0;
        font-size: 1.08rem;
        color: #1d4ed8;
        font-weight: 650;
    }
    .hero .description {
        margin: 0;
        color: #475569;
        font-size: 0.98rem;
        max-width: 58rem;
    }
    .panel {
        padding: 1.1rem 1.1rem 0.95rem 1.1rem;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(15, 23, 42, 0.08);
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
        margin-bottom: 0.9rem;
    }
    .section-title {
        font-size: 1rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0 0 0.6rem 0;
    }
    .muted {
        color: #64748b;
        font-size: 0.92rem;
    }
    .stButton > button {
        border-radius: 12px;
        padding: 0.72rem 1.15rem;
        border: none;
        font-weight: 700;
        background: linear-gradient(135deg, #1d4ed8 0%, #0f766e 100%);
        color: white;
        box-shadow: 0 12px 24px rgba(29, 78, 216, 0.22);
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 14px 30px rgba(29, 78, 216, 0.28);
    }
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.92);
        border-radius: 18px;
        border: 1px solid rgba(15, 23, 42, 0.07);
        padding: 0.6rem 0.75rem;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.04);
    }
    div[data-testid="stMetric"] * {
        color: #111827 !important;
        opacity: 1 !important;
        -webkit-text-fill-color: #111827 !important;
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"],
    div[data-testid="stMetric"] [data-testid="stMetricDelta"],
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #111827 !important;
        opacity: 1 !important;
        -webkit-text-fill-color: #111827 !important;
    }
    div[data-testid="stMetric"] small,
    div[data-testid="stMetric"] span,
    div[data-testid="stMetric"] p {
        color: #374151 !important;
        opacity: 1 !important;
        -webkit-text-fill-color: #374151 !important;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f3f6fb 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #0f172a;
    }
    section[data-testid="stSidebar"] .stButton > button,
    section[data-testid="stSidebar"] button[kind="secondary"] {
        background: #10B981 !important;
        color: #ffffff !important;
        border: 1px solid #10B981 !important;
        border-radius: 10px !important;
        box-shadow: 0 12px 24px rgba(16, 185, 129, 0.18) !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover,
    section[data-testid="stSidebar"] button[kind="secondary"]:hover {
        background: #059669 !important;
        border-color: #059669 !important;
        color: #ffffff !important;
        box-shadow: 0 14px 28px rgba(5, 150, 105, 0.22) !important;
    }
    section[data-testid="stSidebar"] .stFileUploader button,
    section[data-testid="stSidebar"] .stFileUploader [role="button"] {
        background: #10B981 !important;
        color: #ffffff !important;
        border: 1px solid #10B981 !important;
        border-radius: 10px !important;
    }
    section[data-testid="stSidebar"] .stFileUploader button:hover,
    section[data-testid="stSidebar"] .stFileUploader [role="button"]:hover {
        background: #059669 !important;
        border-color: #059669 !important;
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="input"],
    section[data-testid="stSidebar"] [data-baseweb="textarea"],
    section[data-testid="stSidebar"] [data-baseweb="select"],
    section[data-testid="stSidebar"] .stFileUploader,
    section[data-testid="stSidebar"] .stFileUploader section,
    section[data-testid="stSidebar"] .stNumberInput {
        background: #ffffff !important;
        color: #0f172a !important;
        border-color: #d1d5db !important;
    }
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] textarea,
    section[data-testid="stSidebar"] [contenteditable="true"] {
        background: #ffffff !important;
        color: #0f172a !important;
        caret-color: #0f172a;
    }
    section[data-testid="stSidebar"] [data-baseweb="input"] input,
    section[data-testid="stSidebar"] [data-baseweb="textarea"] textarea,
    section[data-testid="stSidebar"] .stNumberInput input {
        background: #ffffff !important;
        color: #0f172a !important;
    }
    section[data-testid="stSidebar"] .stFileUploader label,
    section[data-testid="stSidebar"] .stFileUploader span,
    section[data-testid="stSidebar"] .stFileUploader button,
    section[data-testid="stSidebar"] .stNumberInput label,
    section[data-testid="stSidebar"] .stNumberInput span {
        color: #0f172a !important;
    }
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        color: #0f172a;
    }
</style>
"""


def initialize_state() -> None:
    """Initialize Streamlit session state keys."""
    defaults = {
        "analysis_complete": False,
        "front_original": None,
        "side_original": None,
        "front_overlay": None,
        "side_overlay": None,
        "front_measurement_overlay": None,
        "side_measurement_overlay": None,
        "front_result": None,
        "side_result": None,
        "measurement_result": None,
        "height_cm": 170.0,
        "processed_signatures": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource
def get_pose_detector() -> PoseDetector:
    """Create a single detector instance for the app session."""
    return PoseDetector()


def image_to_display(image: Image.Image, max_width: int = 1100) -> Image.Image:
    resized = resize_pil_image(image, max_width=max_width)
    return resized


def process_image(detector: PoseDetector, image: Image.Image, label: str) -> Tuple[Image.Image, PoseDetectionResult]:
    """Convert, analyze, and prepare a single image for display."""
    image_bgr = pil_to_bgr(image)
    result = detector.detect_and_draw(image_bgr, label=label)
    overlay = bgr_to_pil(resize_for_display(result.image_bgr))
    return overlay, result


def create_measurement_cards(estimate: MeasurementEstimate) -> None:
    metric_cols = st.columns(5)
    metric_specs = [
        ("Chest", f"{estimate.chest_cm:.1f} cm", "Ellipse estimate"),
        ("Waist", f"{estimate.waist_cm:.1f} cm", "Ellipse estimate"),
        ("Hip", f"{estimate.hip_cm:.1f} cm", "Ellipse estimate"),
        ("Arm Length", f"{estimate.arm_length_cm:.1f} cm", "Pose geometry"),
        ("Confidence", f"{estimate.confidence:.2f}", "Heuristic score"),
    ]
    for col, (label, value, delta) in zip(metric_cols, metric_specs):
        with col:
            st.metric(label, value, delta)


def render_sidebar_preview(title: str, image: Optional[Image.Image]) -> None:
    if image is None:
        st.caption(f"{title}: waiting for upload")
        return

    preview = image_to_display(image, max_width=420)
    st.image(preview, caption=title, use_container_width=True)


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="panel">
            <div class="section-title">Ready for analysis</div>
            <p class="muted">
                Upload a front-view image, a side-view image, and the user height to start a MediaPipe pose pass.
                Phase 1 focuses on clean landmark visualization and a polished workflow.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_analysis_results() -> None:
    front_image = st.session_state.front_original
    side_image = st.session_state.side_original
    front_overlay = st.session_state.front_overlay
    side_overlay = st.session_state.side_overlay
    front_measurement_overlay = st.session_state.front_measurement_overlay
    side_measurement_overlay = st.session_state.side_measurement_overlay
    front_result = st.session_state.front_result
    side_result = st.session_state.side_result
    measurement_result = st.session_state.measurement_result

    st.subheader("Analysis Results")
    st.caption("Original images and MediaPipe pose overlays are shown side by side for quick review.")

    tabs = st.tabs(["Original Images", "Landmark Overlays", "Measurement Visuals"])

    with tabs[0]:
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("#### Front View")
            st.image(front_image, use_container_width=True)
        with right:
            st.markdown("#### Side View")
            st.image(side_image, use_container_width=True)

    with tabs[1]:
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("#### Front View Overlay")
            st.image(front_overlay, use_container_width=True)
            if front_result is not None:
                st.caption(front_result.message)
        with right:
            st.markdown("#### Side View Overlay")
            st.image(side_overlay, use_container_width=True)
            if side_result is not None:
                st.caption(side_result.message)

    with tabs[2]:
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("#### Front View Measurement Overlay")
            st.image(front_measurement_overlay, use_container_width=True)
        with right:
            st.markdown("#### Side View Measurement Overlay")
            st.image(side_measurement_overlay, use_container_width=True)

    st.markdown("### Estimated Measurements")
    if measurement_result is not None:
        create_measurement_cards(measurement_result)
        for note in measurement_result.notes:
            st.caption(note)

    st.info(
        "Measurements are prototype estimates derived from landmarks, height scaling, and ellipse-based approximations."
    )


def main() -> None:
    initialize_state()
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Phase 1 MVP</div>
            <h1>SizeSense AI</h1>
            <p class="subtitle">Computer Vision-Based Human Body Measurement Estimation</p>
            <p class="description">Estimate body measurements using pose landmarks, silhouette analysis, and geometric scaling.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("### Uploads and Inputs")
    st.sidebar.caption("Provide both body views plus the user height in centimeters.")

    front_file = st.sidebar.file_uploader(
        "Front View Image",
        type=["png", "jpg", "jpeg", "webp"],
        help="Upload a clear, full-body front-view image.",
    )
    side_file = st.sidebar.file_uploader(
        "Side View Image",
        type=["png", "jpg", "jpeg", "webp"],
        help="Upload a clear, full-body side-view image.",
    )
    height_cm = st.sidebar.number_input(
        "Height (cm)",
        min_value=80.0,
        max_value=250.0,
        value=float(st.session_state.height_cm),
        step=0.5,
    )
    st.session_state.height_cm = height_cm

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Upload Preview")
    front_preview = load_image(front_file)
    side_preview = load_image(side_file)
    render_sidebar_preview("Front View", front_preview)
    render_sidebar_preview("Side View", side_preview)

    analyze_clicked = st.sidebar.button("Analyze Body Measurements", use_container_width=True)
    st.sidebar.caption("Supported formats: PNG, JPG, JPEG, WEBP")

    st.markdown("<div class='panel'><div class='section-title'>Session Summary</div></div>", unsafe_allow_html=True)
    summary_cols = st.columns(3)
    with summary_cols[0]:
        st.metric("Views", "2", "Front + side")
    with summary_cols[1]:
        st.metric("Height", f"{height_cm:.1f} cm", "User input")
    with summary_cols[2]:
        st.metric("Pose Model", "MediaPipe", "Image-based")

    current_signatures = (
        getattr(front_file, "name", None),
        getattr(front_file, "size", None),
        getattr(side_file, "name", None),
        getattr(side_file, "size", None),
    )
    processed_signatures = st.session_state.get("processed_signatures")
    if st.session_state.analysis_complete and processed_signatures != current_signatures:
        st.session_state.analysis_complete = False
        st.session_state.measurement_result = None

    if analyze_clicked:
        if front_file is None or side_file is None:
            st.warning("Please upload both front-view and side-view images before running analysis.")
        else:
            front_image = front_preview
            side_image = side_preview
            if front_image is None or side_image is None:
                st.error("One or more uploaded files could not be read as images.")
            else:
                progress = st.progress(0)
                detector = get_pose_detector()
                with st.spinner("Running MediaPipe Pose on both views..."):
                    progress.progress(20)
                    front_overlay, front_result = process_image(detector, front_image, "Front View")
                    progress.progress(60)
                    side_overlay, side_result = process_image(detector, side_image, "Side View")
                    progress.progress(100)

                st.session_state.front_original = image_to_display(front_image)
                st.session_state.side_original = image_to_display(side_image)
                st.session_state.front_overlay = front_overlay
                st.session_state.side_overlay = side_overlay
                measurement_result = estimate_measurements(front_result, side_result, height_cm)
                st.session_state.measurement_result = measurement_result
                st.session_state.front_measurement_overlay = bgr_to_pil(
                    draw_measurement_overlay(front_result.image_bgr, front_result, measurement_result, "Front View Measurements")
                )
                st.session_state.side_measurement_overlay = bgr_to_pil(
                    draw_measurement_overlay(side_result.image_bgr, side_result, measurement_result, "Side View Measurements")
                )
                st.session_state.front_result = front_result
                st.session_state.side_result = side_result
                st.session_state.analysis_complete = True
                st.session_state.processed_signatures = current_signatures

                if front_result.detected and side_result.detected:
                    st.success("Pose landmarks detected in both views.")
                elif front_result.detected or side_result.detected:
                    st.warning("Pose landmarks were detected in one view, but not the other.")
                else:
                    st.warning("No pose landmarks were detected in either image. Try clearer full-body uploads.")

    if st.session_state.analysis_complete:
        render_analysis_results()
    else:
        render_empty_state()


if __name__ == "__main__":
    main()
