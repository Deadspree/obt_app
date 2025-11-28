import cv2
import numpy as np
from ultralytics import YOLO
from sort_tracker import Sort  # Simple Online and Realtime Tracker
from pathlib import Path
import streamlit as st
import matplotlib.pyplot as plt
import tempfile
def tracking_streamlit(input_video_name: str, model_weight: str = "best.pt"):
    """
    !Orange ball trajectory detection and save the output video
    """
    PROJECT_ROOT = Path(__file__).resolve().parent

    output_tempfile = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    output_path = output_tempfile.name

    model_path = PROJECT_ROOT/ "models" / "weights" / model_weight
    model = YOLO(model_path)
    cap = cv2.VideoCapture(input_video_name)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 3️⃣ Initialize SORT tracker
    tracker = Sort(max_age=10, min_hits=1, iou_threshold=0.3)

    # 4️⃣ Store trajectories per track_id
    trajectories = {}  # track_id -> list of (x_center, y_center)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect balls
        results = model.predict(frame, imgsz=640, verbose=False)
        dets = []

        # Convert YOLO boxes to SORT format
        for box, conf in zip(results[0].boxes.xyxy, results[0].boxes.conf):
            x1, y1, x2, y2 = box.cpu().numpy()
            score = float(conf.cpu().numpy())
            dets.append([x1, y1, x2, y2, score])

        # If no detections, create empty array
        if len(dets) == 0:
            dets = np.empty((0, 5), dtype=np.float32)
        else:
            dets = np.array(dets, dtype=np.float32)

        # Update tracker
        tracks = tracker.update(dets)
        # Draw tracks and store trajectories
        for track in tracks:
            x1, y1, x2, y2, track_id, _ = track
            x1, y1, x2, y2, track_id = int(x1), int(y1), int(x2), int(y2), int(track_id)
            x_center = int((x1 + x2) / 2)
            y_center = int((y1 + y2) / 2)

            # Store trajectory
            if track_id not in trajectories:
                trajectories[track_id] = []
            trajectories[track_id].append((x_center, y_center))

            # Draw bounding box
            color = tuple(np.random.randint(0,255,3).tolist())  # unique random color per track
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID:{track_id}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw trajectory lines
        for tid, points in trajectories.items():
            for i in range(1, len(points)):
                cv2.line(frame, points[i-1], points[i], (0,0,255), 2)
        

        out.write(frame)

    cap.release()
    out.release()

    # --- Draw trajectory graph ---
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot trajectories
    for tid, points in trajectories.items():
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, marker='o', label=f'Track id {tid}')

    ax.invert_yaxis()  # match image coordinates (top-left origin)
    ax.set_xlabel("X position (pixels)")
    ax.set_ylabel("Y position (pixels)")
    ax.set_title("Trajectory of Detected Balls")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

    # Save figure as PNG
    #fig.savefig(graph_path)
    return fig, output_path
    # Display in Streamlit
    st.pyplot(fig)

    # Close figure to free memory
    plt.close(fig)

def main():
    PROJECT_ROOT = Path(__file__).resolve().parent
    INPUT_DIR = PROJECT_ROOT / "input"
    OUTPUT_DIR = PROJECT_ROOT / "output"
    st.title("Orange Ball Tracker")
    # Initialize session state
    if "processing" not in st.session_state:
        st.session_state.processing = False
        # Video upload
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        # Save uploaded file temporarily

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        # Disable button if processing
        button_disabled = st.session_state.processing
        if st.button("Track Orange Ball", disabled=button_disabled):
            st.session_state.processing = True  # mark as processing
            #temp_output_path = OUTPUT_DIR / "temp_result.mp4"
            #temp_output_graph = OUTPUT_DIR / "trajectory_graph.png"
            with st.spinner("Video is being analyzed, please wait..."):

                # Call your tracking function
                fig, output_path = tracking_streamlit(input_video_name=video_path)
            st.session_state.processing = False  # done processing

            st.success("Tracking complete! Check Graph and Video")
            # Show the processed video
            st.pyplot(fig)
            st.video(str(output_path))
            

if __name__ == "__main__":
    main()
    
