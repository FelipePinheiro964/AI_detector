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