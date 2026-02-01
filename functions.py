import streamlit as st
import cv2
import tempfile
import numpy as np
from plyer import notification

# Adicionamos 'uploaded_video', 'model' e 'conf_threshold' como parâmetros
def video(uploaded_video, model, conf_threshold):
    # Salva o vídeo temporariamente para o OpenCV ler
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_video.read())
    
    cap = cv2.VideoCapture(tfile.name)
    st_frame = st.empty() # Espaço para o vídeo
    alerta_site = st.empty() # Espaço para a mensagem
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        results = model(frame, conf=conf_threshold, verbose=False)
        
        for r in results:
            if any(box.conf < 0.5 for box in r.boxes):
                alerta_site.error("⚠️ ANOMALIA DETECTADA: Inconsistência visual identificada!")
                notification.notify(title="Alerta IA", message="Inconsistência no vídeo!", timeout=2)
        
        annotated_frame = results[0].plot()
        # Converte de BGR para RGB para o Streamlit mostrar certo
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        st_frame.image(annotated_frame, channels="RGB")
        
    cap.release()