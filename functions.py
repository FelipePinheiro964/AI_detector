import streamlit as st
import cv2
import tempfile
import numpy as np
from plyer import notification
from PIL import Image
from mss import mss
import time
import os

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
            st.warning(f"⚠️ Atenção: Detectado objeto com baixa confiança ({model.names[int(box.cls)]}). Isso pode ser sinal de manipulação.")
        else:
            st.write("A detecção não identificou anomalias na imagem.")

# Adicionamos 'uploaded_video', 'model' e 'conf_threshold' como parâmetros
# def video(uploaded_video, model, conf_threshold):
#     # Salva o vídeo temporariamente para o OpenCV ler
#     tfile = tempfile.NamedTemporaryFile(delete=False) 
#     tfile.write(uploaded_video.read())
    
#     cap = cv2.VideoCapture(tfile.name)
#     st_frame = st.empty() # Espaço para o vídeo
#     alerta_site = st.empty() # Espaço para a mensagem
    
#     ultimo_alerta = 0 # Variável para controlar o tempo
#     intervalo_seguranca = 10 # Tempo em segundos entre um alerta e outro

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret: break
        
#         results = model(frame, conf=conf_threshold, verbose=False)

#         agora = time.time() # Pega o horario atual
        
#         for r in results:
#             if any(box.conf < 0.5 for box in r.boxes):
#                 if agora - ultimo_alerta > intervalo_seguranca:
#                     alerta_site.error("⚠️ ANOMALIA DETECTADA: Inconsistência visual identificada!")
#                     notification.notify(title="Alerta IA", message="Inconsistência no vídeo!", timeout=2)
#                     ultimo_alerta = agora # Reseta o cronômetro

#         annotated_frame = results[0].plot()
#         # Converte de BGR para RGB para o Streamlit mostrar certo
#         annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
#         st_frame.image(annotated_frame, channels="RGB", use_container_width=True)
#         time.sleep(0.01)
        
#     cap.release()



def video2(uploaded_video, model, conf_threshold):
    # Usamos um sufixo para garantir que o Linux entenda o formato
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(uploaded_video.read())
        temp_path = tfile.name

    cap = cv2.VideoCapture(temp_path)
    
    # Criamos o container do vídeo e dos alertas
    st_frame = st.empty()
    alerta_site = st.empty()
    
    ultimo_alerta = 0
    intervalo_seguranca = 5

    # Verificamos se o vídeo abriu corretamente
    if not cap.isOpened():
        st.error("Erro ao carregar o vídeo. Tente outro formato.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Redimensiona o frame para processar mais rápido na nuvem
        frame = cv2.resize(frame, (640, 480))
        
        results = model(frame, conf=conf_threshold, verbose=False)
        
        agora = time.time()
        for r in results:
            if any(box.conf < 0.5 for box in r.boxes):
                if agora - ultimo_alerta > intervalo_seguranca:
                    alerta_site.error("⚠️ ANOMALIA DETECTADA!")
                    # Removi a notificação plyer aqui para não dar erro na nuvem
                    ultimo_alerta = agora

        # Converte e exibe
        annotated_frame = results[0].plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        # O pulo do gato: usamos use_container_width e um pequeno sleep
        st_frame.image(annotated_frame, channels="RGB", use_container_width=True)
        time.sleep(0.01) # Essencial para o navegador "respirar" e mostrar o frame

    cap.release()
    # Limpa o arquivo temporário para não encher o servidor
    os.remove(temp_path)
    st.success("Análise concluída!")

# Não funciona em nuvem
# def monitoramento_tempo_real(model, conf_threshold):
#     sct = mss()
#     monitor = sct.monitors[1] # Captura o monitor principal
    
#     st_frame = st.empty() # Espaço para o vídeo da tela
#     alerta_site = st.empty()
    
#     # Botão para parar o monitoramento (Streamlit reinicia o script se clicado)
#     if st.button("Parar Monitoramento"):
#         st.rerun()

#     while True:
#         # Captura o frame da tela
#         sct_img = sct.grab(monitor)
#         frame = np.array(sct_img)
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

#         results = model(frame, conf=conf_threshold, verbose=False)
        
#         # Lógica de Alerta
#         for r in results:
#             if any(box.conf < 0.5 for box in r.boxes):
#                 alerta_site.error("⚠️ POSSÍVEL FRAUDE OU IA DETECTADA NA TELA!")
#                 notification.notify(title="Alerta de Tela", message="Inconsistência detectada!", timeout=1)

#         # Prepara imagem para o Streamlit
#         annotated_frame = results[0].plot()
#         annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
#         st_frame.image(annotated_frame, channels="RGB")