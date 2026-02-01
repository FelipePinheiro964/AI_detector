import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import time 
from plyer import notification
from functions import video

# Configuração inicial
st.set_page_config(page_title="Detector Protetor", page_icon="🛡️", layout="wide")
model = YOLO('yolov8n.pt') 

st.title("Detector de videos e fotos feitos por IA")
st.write("Envie uma foto ou vídeo para verificar se existem inconsistências visuais.")

st.sidebar.header("Configurações de Análise")
conf_threshold = st.sidebar.slider("Confiança Mínima", 0.0, 1.0, 0.5)

opcao = st.radio("Selecione o que deseja analisar:", ("Foto", "Vídeo"))

if opcao == "Foto":
    uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Converte o upload para imagem OpenCV
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Roda a detecção
        results = model(img_array, conf=conf_threshold)
        
        # Desenha resultados e mostra
        res_plotted = results[0].plot()
        st.image(res_plotted, caption='Resultado da Análise', use_container_width=True)
        
        # Verifica anomalias
        for box in results[0].boxes:
            if box.conf < 0.55:
                st.warning(f"Atenção: Detectado objeto com baixa confiança ({model.names[int(box.cls)]}). Isso pode ser sinal de manipulação.")
            else:
                st.write("A detecção não identificou anomalias na imagem.")

elif opcao == "Vídeo":
    uploaded_video = st.file_uploader("Escolha um vídeo...", type=["mp4", "mov", "avi"])
    
    if uploaded_video is not None:
        video(uploaded_video, model, conf_threshold)