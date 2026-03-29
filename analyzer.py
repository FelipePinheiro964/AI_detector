import cv2
import numpy as np
import os

from functions_v2 import (
    score_texture_uniformity,
    score_color_distribution,
    score_frequency_analysis,
    score_noise_pattern,
    score_compression_artifacts,
    score_temporal_consistency,
    score_motion_naturalness,
    score_facial_artifacts,
    WEIGHTS
)

YOLO_MODEL     = None
YOLO_AVAILABLE = False

try:
    from ultralytics import YOLO
    _path = os.path.join(os.path.dirname(__file__), 'yolov8n.pt')
    if os.path.exists(_path):
        YOLO_MODEL     = YOLO(_path)
        YOLO_AVAILABLE = True
        print("[VerifAI] YOLOv8 carregado.")
    else:
        print("[VerifAI] yolov8n.pt não encontrado — YOLO desativado.")
except ImportError:
    print("[VerifAI] ultralytics não instalado — YOLO desativado.")

def extract_frames(video_path, max_frames=60):
    cap   = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], 0, 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 24
    step  = max(1, total // max_frames)
    frames = []
    idx = 0
    while idx < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        idx += step
    cap.release()
    return frames, fps, total