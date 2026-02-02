import streamlit as st
from ultralytics import YOLO
# from functions import video,  foto, monitoramento_tempo_real

from functions import video,  foto, video2

# Configuração inicial
st.set_page_config(page_title="Detector Protetor", page_icon="🛡️", layout="wide")
model = YOLO('yolov8n.pt') 

st.title("Detector de videos e fotos feitos por IA")
st.write("Envie uma foto ou vídeo para verificar se existem inconsistências visuais.")

st.sidebar.header("Configurações de Análise")
conf_threshold = st.sidebar.slider("Confiança Mínima", 0.0, 1.0, 0.5)

# opcao = st.radio("Selecione o que deseja analisar:", ("Foto", "Vídeo", "Monitorar Tela"))
opcao = st.radio("Selecione o que deseja analisar:", ("Foto", "Vídeo", "video2"))

if opcao == "Foto":
    uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        foto(uploaded_file, model, conf_threshold)


# elif opcao == "Vídeo":
#     uploaded_video = st.file_uploader("Escolha um vídeo...", type=["mp4", "mov", "avi"])
    
#     if uploaded_video is not None:
        video(uploaded_video, model, conf_threshold)

elif opcao == "video2":
    uploaded_video = st.file_uploader("Escolha um vídeo...", type=["mp4", "mov", "avi"])
    
    if uploaded_video is not None:
        video(uploaded_video, model, conf_threshold)

# elif opcao == "Monitorar Tela":
#     st.info("O sistema está analisando sua tela inteira agora.")
#     monitoramento_tempo_real(model, conf_threshold)

# Marca d'agua
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 10px;
        width: 100%;
        color: gray;
        text-align: right;
        padding-right: 20px;
        font-size: 18px;
    }
    </style>
    <div class="footer">Está é uma versão incial de um projeto de indentificação para videos/fotos feitos por IA | Desenvolvido por: Felipe Pinheiro | Estudante de Banco de Dados - PUCRS</div>
    """,
    unsafe_allow_html=True
)
