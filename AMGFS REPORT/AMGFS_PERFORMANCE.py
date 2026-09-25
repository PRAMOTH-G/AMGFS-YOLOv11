import cv2
import time
import torch
import glob
import numpy as np
import pandas as pd
from ultralytics import YOLO

# GPU check
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Executing on Device: {device}")

# Load YOLOv11 Pose Model (Replaces MediaPipe; Detection + Keypoint Pose in 1 Single Pass!)
yolo_pose_model = YOLO('yolo11n-pose.pt')

def calculate_motion_score(prev_frame, curr_frame):
    if prev_frame is None:
        return 1.0
    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    frame_diff = cv2.absdiff(gray_prev, gray_curr)
    return np.mean(frame_diff) / 255.0

def run_benchmark_on_folder(tau_motion=0.03, use_amgfs=True):
    video_files = glob.glob('mini_dataset/*/*.mp4')
    
    total_frames = 0
    processed_frames = 0
    skipped_frames = 0
    frame_times = []

    for video_path in video_files:
        cap = cv2.VideoCapture(video_path)
        prev_frame = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1
            start_time = time.time()
            motion_score = calculate_motion_score(prev_frame, frame)

            # Proposed AMGFS Logic
            if not use_amgfs or motion_score >= tau_motion or total_frames == 1:
                processed_frames += 1
                # YOLOv11 Pose handles Human Detection + 17 Skeleton Keypoint extraction
                _ = yolo_pose_model(frame, verbose=False)
            else:
                skipped_frames += 1

            end_time = time.time()
            frame_times.append((end_time - start_time) * 1000)
            prev_frame = frame.copy()

        cap.release()

    avg_latency = np.mean(frame_times)
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0

    return {
        "Total Clips": len(video_files),
        "Total Frames": total_frames,
        "Processed Frames": processed_frames,
        "Skipped Frames": skipped_frames,
        "Avg Latency (ms)": round(avg_latency, 2),
        "FPS": round(fps, 2),
        "Mode": "Proposed AMGFS (YOLOv11 Pose)" if use_amgfs else "Standard Baseline"
    }

print("Running Standard Baseline Pipeline...")
baseline = run_benchmark_on_folder(use_amgfs=False)

print("Running Proposed AMGFS Pipeline (YOLOv11 Pose)...")
amgfs = run_benchmark_on_folder(tau_motion=0.03, use_amgfs=True)

results_df = pd.DataFrame([baseline, amgfs])
print("\n=== EXPERIMENTAL BENCHMARK RESULTS ===")
print(results_df.to_string(index=False))