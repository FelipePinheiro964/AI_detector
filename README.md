# AI_detector

Detector de conteúdo gerado por Inteligência Artificial.
Projeto desenvolvido com foco em acessibilidade para idosos.

---

## Versões

### v1 — Streamlit + YOLOv8

Primeira versão hospedada no Streamlit Cloud.
Usava YOLOv8 para detecção de objetos e Streamlit como interface.
Arquivos: `app.py`, `functions.py`

### v2 — Flask + 8 Indicadores Forenses + YOLOv8

Interface web própria em Flask, redesenhada para idosos.
Análise por 8 indicadores forenses + YOLOv8 integrado.
Arquivos na pasta `AI_detector_v2/`

---

## Estrutura v2

```
AI_detector_v2/
├── app2.py          — Servidor Flask (rotas)
├── analyzer.py      — Extração de frames e score final
├── functions_v2.py  — 8 indicadores de detecção
├── templates/
│   └── index.html   — Interface web (frontend)
└── yolov8n.pt       — Modelo YOLOv8 (copiado da v1)
```

## Como rodar a v2

```bash
cd AI_detector_v2
pip install flask opencv-python-headless numpy Pillow scipy ultralytics
python app2.py
```

Acesse o link disponibilizado ao rodar o comando no terminal

## Como rodar a v1

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Indicadores de detecção (v2)

| #   | Indicador  | Técnica                                |
| --- | ---------- | -------------------------------------- |
| 1   | Textura    | Variância do Laplaciano                |
| 2   | Cores      | Entropia do histograma RGB             |
| 3   | Frequência | Transformada de Fourier (FFT 2D)       |
| 4   | Ruído      | Autocorrelação do resíduo Gaussiano    |
| 5   | Compressão | Diferença em bordas de blocos 8×8      |
| 6   | Temporal   | Coeficiente de variação entre frames   |
| 7   | Movimento  | Fluxo óptico de Farneback              |
| 8   | Rostos     | Haar Cascade + análise de simetria     |
| 9   | YOLOv8     | Objetos com baixa confiança (opcional) |

---

## Proteção de dados (LGPD)

- Arquivos enviados são apagados imediatamente após a análise
- Nenhum dado é armazenado ou compartilhado
- A pasta `static/uploads/` está no `.gitignore`
