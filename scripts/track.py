import cv2
import numpy as np
from ultralytics import YOLO
from sort_tracker import Sort  # Simple Online and Realtime Tracker
from pathlib import Path
import streamlit as st

def tracking(input_video_name: str, output_video_name: str, model_weight: str = "best.pt"):
    """
    !Orange ball trajectory detection and save the output video
    """
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    model_path = PROJECT_ROOT/ "models" / "weights" / model_weight
    input_path = PROJECT_ROOT/ "input" / input_video_name
    output_path = PROJECT_ROOT / "output" / output_video_name

    model = YOLO(model_path)
    cap = cv2.VideoCapture(input_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
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
    cv2.destroyAllWindows()

def main():
    input_video_name = "ball_test.mp4"
    output_video_name = "output2.mp4"
    tracking(input_video_name, output_video_name)

if __name__ == "__main__":
    main()
