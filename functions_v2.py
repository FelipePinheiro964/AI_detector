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