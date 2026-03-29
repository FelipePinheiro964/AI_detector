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