import streamlit as st
import cv2
import tempfile
import numpy as np
from plyer import notification
from PIL import Image
import numpy as np

def foto(uploaded_file, model, conf_threshold):
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