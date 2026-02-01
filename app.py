import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile

# Configuração inicial
st.set_page_config(page_title="Detector Protetor", page_icon="🛡️", layout="wide")
model = YOLO('yolov8n.pt') 

st.title("Detector de videos e fotos feitos por IA")
st.write("Envie uma foto ou vídeo para verificar se existem inconsistências visuais.")

st.sidebar.header("Configurações de Análise")
conf_threshold = st.sidebar.slider("Confiança Mínima", 0.0, 1.0, 0.5)

opcao = st.radio("Selecione o que deseja analisar:", ("Foto", "Vídeo"))