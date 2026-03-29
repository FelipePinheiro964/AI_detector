import cv2
import numpy as np

# IA gera imagens artificialmente lisas. O Laplaciano mede o quanto uma imagem tem 
# de bordas e detalhes, se for um resultado baixo = imagem lisa demais, tornando suspeito.

def score_texture_uniformity(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()

    if variance < 50:
        return 0.75      # muito liso → suspeito de IA
    elif variance < 150:
        return 0.40
    elif variance > 3000:
        return 0.15      # muito ruidoso → provavelmente real
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