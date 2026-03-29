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

def score_yolo_anomaly(frames, sample=10):
    if not YOLO_AVAILABLE or YOLO_MODEL is None:
        return None
    sample_frames    = frames[::max(1, len(frames) // sample)][:sample]
    low_conf_ratios  = []
    for frame in sample_frames:
        results   = YOLO_MODEL(frame, conf=0.25, verbose=False)
        boxes     = results[0].boxes
        if len(boxes) == 0:
            continue
        confs     = [float(b.conf) for b in boxes]
        low_conf_ratios.append(sum(1 for c in confs if c < 0.55) / len(confs))
    if not low_conf_ratios:
        return None
    avg = float(np.mean(low_conf_ratios))
    if avg > 0.6:  return 0.70
    if avg > 0.35: return 0.45
    return 0.15


def analyze(video_path):
    frames, fps, total_frames = extract_frames(video_path)
    if not frames:
        return {"error": "Não foi possível ler o arquivo."}

    tex, col, freq, noise, comp, face = [], [], [], [], [], []
    for frame in frames:
        tex.append(score_texture_uniformity(frame))
        col.append(score_color_distribution(frame))
        freq.append(score_frequency_analysis(frame))
        noise.append(score_noise_pattern(frame))
        comp.append(score_compression_artifacts(frame))
        fs = score_facial_artifacts(frame)
        if fs is not None:
            face.append(fs)

    temporal = score_temporal_consistency(frames)
    motion   = score_motion_naturalness(frames)
    yolo     = score_yolo_anomaly(frames)

    scores = {
        "texture_uniformity":     round(float(np.mean(tex)),   3),
        "color_anomaly":          round(float(np.mean(col)),   3),
        "frequency_artifacts":    round(float(np.mean(freq)),  3),
        "noise_pattern":          round(float(np.mean(noise)), 3),
        "compression_pattern":    round(float(np.mean(comp)),  3),
        "temporal_inconsistency": round(temporal,              3),
        "motion_unnaturalness":   round(motion,                3),
    }
    if face:
        scores["facial_artifacts"] = round(float(np.mean(face)), 3)
    if yolo is not None:
        scores["yolo_anomaly"] = round(yolo, 3)

    total_w = weighted_sum = 0
    for key, val in scores.items():
        w              = WEIGHTS.get(key, 1.0)
        weighted_sum  += val * w
        total_w       += w

    ai_probability = round(weighted_sum / total_w, 4)

    if ai_probability >= 0.60:
        verdict, confidence, color = "GERADO POR IA",      "Alta",       "red"
    elif ai_probability >= 0.45:
        verdict, confidence, color = "PROVAVELMENTE IA",   "Média",      "orange"
    elif ai_probability >= 0.30:
        verdict, confidence, color = "INCONCLUSIVO",       "Baixa",      "yellow"
    else:
        verdict, confidence, color = "PROVAVELMENTE REAL", "Média-Alta", "green"

    return {
        "ai_probability": ai_probability,
        "verdict":        verdict,
        "confidence":     confidence,
        "color":          color,
        "scores":         scores,
        "yolo_used":      yolo is not None,
        "details": {
            "faces_detected":  len(face) > 0,
            "total_frames":    total_frames,
            "frames_analyzed": len(frames),
            "fps":             round(fps, 2),
            "duration_sec":    round(total_frames / max(fps, 1), 2),
        }
    }