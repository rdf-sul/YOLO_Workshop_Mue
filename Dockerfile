# ─────────────────────────────────────────────────────────────
# YOLO Bauteile – gemeinsames Image
# Basis: ultralytics:latest-jupyter  +  Streamlit
# ─────────────────────────────────────────────────────────────
FROM ultralytics/ultralytics:latest-jupyter

# Streamlit nachinstallieren
RUN pip install --no-cache-dir streamlit

# App-Dateien
WORKDIR /workspace
COPY yolo_bauteile_app.py ./
RUN mkdir -p /workspace/models /workspace/data

EXPOSE 8501 8888
