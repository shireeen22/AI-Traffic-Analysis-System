import streamlit as st
from detector import process_video
import os

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="AI Smart Traffic Monitoring",
    page_icon="🚦",
    layout="wide"
)

os.makedirs("assets", exist_ok=True)

# ==================================================
# HEADER
# ==================================================

st.title("🚦 AI Smart Traffic Monitoring System")

st.markdown("""
This application uses **YOLOv8 + ByteTrack** to:

- 🚗 Detect Vehicles
- 🔍 Track Vehicles
- 📊 Count Vehicles
- 🚦 Analyze Traffic
- 📄 Generate Reports
""")

st.divider()

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.header("⚙ Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=1.00,
    value=0.50,
    step=0.05
)

st.sidebar.write(f"Current Confidence: **{confidence:.2f}**")

# ==================================================
# VIDEO UPLOAD
# ==================================================

uploaded_video = st.file_uploader(
    "📤 Upload Traffic Video",
    type=["mp4", "avi", "mov"]
)

video_path = None

if uploaded_video is not None:

    video_path = os.path.join(
        "assets",
        uploaded_video.name
    )

    with open(video_path, "wb") as f:
        f.write(uploaded_video.getbuffer())

    st.success("✅ Video Uploaded Successfully!")

# ==================================================
# START DETECTION
# ==================================================

if st.button("▶ Start Detection", use_container_width=True):

    if video_path is None:

        st.warning("Please upload a video first.")

    else:

        with st.spinner("Processing Video..."):
            results = process_video(video_path)
            st.session_state.results = results
        st.success("✅ Detection Completed!")

# ==================================================
# SHOW RESULTS
# ==================================================

if "results" in st.session_state:

    results = st.session_state.results

    st.divider()

    st.subheader("📊 Detection Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🚗 Cars", results.get("cars", 0))

    with col2:
        st.metric("🚌 Buses", results.get("buses", 0))

    with col3:
        st.metric("🚚 Trucks", results.get("trucks", 0))

    col4, col5, col6 = st.columns(3)

    with col4:
        st.metric("🏍 Motorcycles", results.get("motorcycles", 0))

    with col5:
        st.metric("🚦 Total Vehicles", results.get("total", 0))

    with col6:
        st.metric(
            "📈 Density",
            f"{results.get('density', 0)}%"
        )

    st.divider()

    # ==================================================
    # TRAFFIC STATUS
    # ==================================================

    traffic = results.get("traffic", "UNKNOWN")

    if traffic == "LIGHT":
        st.success("Traffic Status: LIGHT")

    elif traffic == "MODERATE":
        st.warning("Traffic Status: MODERATE")

    elif traffic == "HEAVY":
        st.error("raffic Status: HEAVY")

    else:
        st.info(f"Traffic Status: {traffic}")

    st.divider()

    # ==================================================
    # VIDEO
    # ==================================================

    st.subheader("🎥 Processed Video")

    if os.path.exists(results["processed_video"]):

        with open(results["processed_video"], "rb") as video_file:

            st.video(video_file.read())

    st.divider()

    # ==================================================
    # DOWNLOADS
    # ==================================================

    st.subheader("⬇ Downloads")

    col1, col2 = st.columns(2)

    with col1:

        if os.path.exists(results["processed_video"]):

            with open(results["processed_video"], "rb") as f:

                st.download_button(
                    "⬇ Download Processed Video",
                    f,
                    file_name="processed_traffic.mp4"
                )

    with col2:

        if os.path.exists(results["csv_report"]):

            with open(results["csv_report"], "rb") as f:

                st.download_button(
                    "⬇ Download CSV Report",
                    f,
                    file_name="traffic_report.csv"
                )