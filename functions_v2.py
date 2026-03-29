import cv2
import numpy as np

def score_texture_uniformity(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()

    if variance < 50:
        return 0.75     
    elif variance < 150:
        return 0.40
    elif variance > 3000:
        return 0.15     
    else:
        return 0.25
    
def score_color_distribution(frame):
    scores = []
    for ch in cv2.split(frame):
        hist = cv2.calcHist([ch], [0], None, [256], [0, 256]).flatten()
        hist /= (hist.sum() + 1e-7)
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        if entropy < 5.5:
            scores.append(0.65)
        elif entropy > 7.5:
            scores.append(0.15)
        else:
            scores.append(0.30)
    return float(np.mean(scores))

def score_frequency_analysis(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    magnitude = 20 * np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1)
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    quads = [
        magnitude[:cy, :cx].mean(),
        magnitude[:cy, cx:].mean(),
        magnitude[cy:, :cx].mean(),
        magnitude[cy:, cx:].mean(),
    ]
    quad_var = np.std(quads) / (np.mean(quads) + 1e-7)
    center_ratio = magnitude[cy-5:cy+5, cx-5:cx+5].mean() / (magnitude.mean() + 1e-7)
    score = 0.0
    if quad_var < 0.02:   score += 0.30
    if center_ratio > 12: score += 0.25
    return float(min(score, 0.75))

def score_temporal_consistency(frames):
    if len(frames) < 3:
        return 0.3
    diffs = []
    for i in range(1, len(frames)):
        f1 = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY).astype(float)
        f2 = cv2.cvtColor(frames[i],   cv2.COLOR_BGR2GRAY).astype(float)
        diffs.append(np.abs(f1 - f2).mean())
    diffs = np.array(diffs)
    cv = diffs.std() / (diffs.mean() + 1e-7)
    if cv > 1.5 or cv < 0.05: return 0.70
    if cv > 0.8:               return 0.50
    return 0.20

def score_motion_naturalness(frames):
    if len(frames) < 5:
        return 0.3
    flows = []
    for i in range(1, min(len(frames), 20)):
        g1 = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(frames[i],   cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag  = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        flows.append(mag.mean())
    flows = np.array(flows)
    if len(flows) < 2: return 0.3
    accel_var = np.diff(np.diff(flows)).var()
    if accel_var < 0.001: return 0.65
    if accel_var > 50:    return 0.60
    return 0.20

def score_facial_artifacts(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
    if len(faces) == 0:
        return None
    scores = []
    for (x, y, w, h) in faces:
        roi   = gray[y:y+h, x:x+w]
        left  = roi[:, :w//2]
        right = cv2.flip(roi[:, w//2:], 1)
        min_w = min(left.shape[1], right.shape[1])
        symmetry  = 1 - np.abs(left[:, :min_w].astype(float) - right[:, :min_w].astype(float)).mean() / 255
        edge_ratio = cv2.Canny(roi, 50, 150).sum() / (255 * roi.size + 1e-7)
        s = 0.0
        if symmetry > 0.92:   s += 0.35
        if edge_ratio < 0.03: s += 0.30
        scores.append(min(s, 0.85))
    return float(np.mean(scores))