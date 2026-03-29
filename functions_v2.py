import cv2
import numpy as np

# IA gera imagens artificialmente lisas. O Laplaciano mede o quanto uma imagem tem 
# de bordas e detalhes, se for um resultado baixo = imagem lisa demais, tornando suspeito.

def score_texture_uniformity(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()