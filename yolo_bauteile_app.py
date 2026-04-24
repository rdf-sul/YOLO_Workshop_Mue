"""
YOLO Bauteil-Prüfer
====================
Streamlit-Anwendung zur Erkennung und Zählung elektronischer Bauteile
mit einem trainierten YOLO11-Modell.

Starten: streamlit run yolo_bauteile_app.py
Starten: uv run yolo solutions inference model=best.pt -> best.pt durch eigenes Modell ersetzen, falls nötig
"""

import streamlit as st
import cv2
import numpy as np
import time
import tempfile
from pathlib import Path
from datetime import datetime
from PIL import Image

# ─────────────────────────────────────────────────────────────
# Seitenkonfiguration
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YOLO Bauteil-Prüfer",
    page_icon="🔌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .check-pass {
    background: #dcfce7; border: 2px solid #16a34a;
    border-radius: 12px; padding: 18px 24px;
    text-align: center; font-size: 1.6rem;
    font-weight: 700; color: #14532d;
  }
  .check-fail {
    background: #fef2f2; border: 2px solid #dc2626;
    border-radius: 12px; padding: 18px 24px;
    text-align: center; font-size: 1.6rem;
    font-weight: 700; color: #7f1d1d;
  }
  .count-card {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 12px 16px;
    text-align: center; margin-bottom: 6px;
  }
  .count-ok   { border-left: 5px solid #16a34a; }
  .count-fail { border-left: 5px solid #dc2626; }
  .count-num  { font-size: 2.2rem; font-weight: 800; line-height: 1.1; }
  .history-row {
    background: #f8fafc; border-radius: 8px;
    padding: 8px 14px; margin-bottom: 5px;
    font-size: 0.88rem; color: #475569;
    border-left: 4px solid #94a3b8;
  }
  .history-pass { border-left-color: #16a34a; }
  .history-fail { border-left-color: #dc2626; }
  div[data-testid="stSidebar"] { min-width: 320px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Session State initialisieren
# ─────────────────────────────────────────────────────────────
defaults = {
    "model":        None,
    "model_path":   "",
    "class_names":  [],
    "history":      [],
    "run_webcam":   False,
    "check_count":  0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────
# Hilfsfunktionen
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Modell wird geladen …")
def load_model(path: str):
    from ultralytics import YOLO
    return YOLO(path)


def run_inference(model, frame_bgr, conf: float, iou: float):
    """Führt Inferenz aus und gibt (annotiertes Bild, Zähldict) zurück."""
    results = model.predict(
        frame_bgr,
        conf=conf,
        iou=iou,
        verbose=False,
    )
    counts = {}
    annotated = results[0].plot()
    for box in results[0].boxes:
        name = model.names[int(box.cls[0])]
        counts[name] = counts.get(name, 0) + 1
    return annotated, counts


def check_pass(counts: dict, targets: dict) -> bool:
    """True wenn alle Ziel-Mengen erfüllt sind."""
    for cls, n in targets.items():
        if n > 0 and counts.get(cls, 0) < n:
            return False
    return True


def add_history(counts: dict, targets: dict, passed: bool, source: str):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.history.append({
        "time":    ts,
        "passed":  passed,
        "counts":  dict(counts),
        "targets": dict(targets),
        "source":  source,
    })
    st.session_state.check_count += 1


def show_result_cards(counts: dict, targets: dict):
    """Zeigt Zählkarten für alle aktiven Klassen."""
    cols = st.columns(max(len(targets), 1))
    for i, (cls, target) in enumerate(targets.items()):
        if target == 0:
            continue
        found  = counts.get(cls, 0)
        ok     = found >= target
        border = "count-ok" if ok else "count-fail"
        icon   = "✅" if ok else "❌"
        color  = "#16a34a" if ok else "#dc2626"
        with cols[i % len(cols)]:
            st.markdown(f"""
            <div class="count-card {border}">
              <div class="count-num" style="color:{color}">{found}</div>
              <div style="font-size:0.85rem;color:#64748b">von {target}</div>
              <div style="font-weight:700;margin-top:4px">{icon} {cls}</div>
            </div>""", unsafe_allow_html=True)


def show_overall(passed: bool):
    if passed:
        st.markdown('<div class="check-pass">✅ PASST – Bauteilsatz vollständig!</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="check-fail">❌ UNVOLLSTÄNDIG – Bauteile fehlen!</div>',
                    unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔌 YOLO Bauteil-Prüfer")
    st.caption("RDF Nürnberg · Elektrotechnik")
    st.divider()

    # ── Modell laden ─────────────────────────────────────────
    st.subheader("📦 Modell")
    model_file = st.file_uploader(
        "YOLO-Modell (.pt) hochladen",
        type=["pt"],
        help="Trainiertes YOLO11-Modell (best.pt oder last.pt)",
    )
    if model_file is not None:
        tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        tmp.write(model_file.read())
        tmp.flush()
        if tmp.name != st.session_state.model_path:
            st.session_state.model      = load_model(tmp.name)
            st.session_state.model_path = tmp.name
            st.session_state.class_names = list(
                st.session_state.model.names.values()
            )
            st.success(f"Modell geladen · {len(st.session_state.class_names)} Klassen")

    if st.session_state.model:
        st.caption(f"Klassen: {', '.join(st.session_state.class_names)}")
    else:
        st.info("Bitte zuerst ein Modell laden.")

    st.divider()

    # ── Eingabequelle ────────────────────────────────────────
    st.subheader("📷 Eingabequelle")
    source = st.radio(
        "Quelle wählen",
        ["📸 Bild hochladen", "🎥 Video hochladen",
         "📹 Webcam 0", "📱 Webcam 1 (Handy)"],
        label_visibility="collapsed",
    )

    st.divider()

    # ── Erkennungsparameter ──────────────────────────────────
    st.subheader("⚙️ Parameter")
    conf = st.slider("Konfidenz-Schwelle", 0.10, 0.95, 0.40, 0.05,
                     help="Mindest-Konfidenz für eine Erkennung")
    iou  = st.slider("IoU-Schwelle (NMS)", 0.10, 0.95, 0.45, 0.05,
                     help="IoU-Schwelle für Non-Maximum Suppression")

    st.divider()

    # ── Ziel-Mengen ──────────────────────────────────────────
    st.subheader("🎯 Bauteilsatz-Ziel")
    st.caption("0 = diese Klasse wird nicht geprüft")

    targets = {}
    class_list = (st.session_state.class_names
                  if st.session_state.class_names
                  else ["LED", "Taster", "Widerstand"])

    for cls in class_list:
        targets[cls] = st.number_input(
            f"{cls}", min_value=0, max_value=50,
            value=1, step=1, key=f"target_{cls}"
        )

    active_targets = {k: v for k, v in targets.items() if v > 0}

    st.divider()

    # ── Verlauf ──────────────────────────────────────────────
    if st.session_state.history:
        st.subheader(f"📋 Verlauf ({st.session_state.check_count} Checks)")
        for entry in reversed(st.session_state.history[-8:]):
            icon  = "✅" if entry["passed"] else "❌"
            cls_str = " · ".join(
                f"{c}:{n}" for c, n in entry["counts"].items()
            ) or "keine Objekte"
            css = "history-pass" if entry["passed"] else "history-fail"
            st.markdown(
                f'<div class="history-row {css}">'
                f'<b>{icon} {entry["time"]}</b> — {entry["source"]}<br>'
                f'{cls_str}</div>',
                unsafe_allow_html=True,
            )
        if st.button("🗑️ Verlauf löschen"):
            st.session_state.history = []
            st.session_state.check_count = 0
            st.rerun()


# ─────────────────────────────────────────────────────────────
# HAUPTBEREICH
# ─────────────────────────────────────────────────────────────
if not st.session_state.model:
    st.markdown("## 👈 Bitte zuerst ein Modell in der Seitenleiste laden")
    st.stop()

model = st.session_state.model

# ══════════════════════════════════════════════════════════════
# BILD
# ══════════════════════════════════════════════════════════════
if source == "📸 Bild hochladen":
    st.header("📸 Bild analysieren")
    uploaded = st.file_uploader(
        "Bild auswählen", type=["jpg", "jpeg", "png", "bmp", "webp"]
    )
    if uploaded:
        img_pil  = Image.open(uploaded).convert("RGB")
        img_bgr  = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        ann, counts = run_inference(model, img_bgr, conf, iou)

        col1, col2 = st.columns([3, 1])
        with col1:
            st.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB),
                     caption="Erkannte Objekte", use_container_width=True)
        with col2:
            passed = check_pass(counts, active_targets)
            show_overall(passed)
            st.markdown("---")
            show_result_cards(counts, active_targets)
            st.markdown("---")
            if st.button("➕ Als Check speichern", type="primary",
                         use_container_width=True):
                add_history(counts, active_targets, passed, "Bild")
                st.success("Gespeichert!")
    else:
        st.info("Bitte ein Bild hochladen.")

# ══════════════════════════════════════════════════════════════
# VIDEO
# ══════════════════════════════════════════════════════════════
elif source == "🎥 Video hochladen":
    st.header("🎥 Video analysieren")
    uploaded = st.file_uploader(
        "Video auswählen", type=["mp4", "mov", "avi", "mkv"]
    )
    if uploaded:
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.write(uploaded.read())
        tmp.flush()

        col1, col2 = st.columns([3, 1])

        with col1:
            play     = st.button("▶️ Analyse starten", type="primary")
            stop_btn = st.button("⏹️ Stoppen")
            frame_ph = st.empty()

        with col2:
            result_ph = st.empty()
            card_ph   = st.empty()
            save_ph   = st.empty()

        if play:
            cap = cv2.VideoCapture(tmp.name)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            frame_num = 0
            last_counts = {}
            last_passed = False

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Nur jeden 3. Frame inferieren (Performance)
                if frame_num % 3 == 0:
                    ann, last_counts = run_inference(model, frame, conf, iou)
                    last_passed = check_pass(last_counts, active_targets)
                    frame_ph.image(
                        cv2.cvtColor(ann, cv2.COLOR_BGR2RGB),
                        caption=f"Frame {frame_num}",
                        use_container_width=True,
                    )
                    with result_ph.container():
                        show_overall(last_passed)
                    with card_ph.container():
                        show_result_cards(last_counts, active_targets)

                frame_num += 1
                time.sleep(1 / fps)

            cap.release()

            with save_ph.container():
                if st.button("➕ Letzten Frame als Check speichern",
                             type="primary", use_container_width=True):
                    add_history(last_counts, active_targets,
                                last_passed, "Video")
                    st.success("Gespeichert!")
    else:
        st.info("Bitte ein Video hochladen.")

# ══════════════════════════════════════════════════════════════
# WEBCAM
# ══════════════════════════════════════════════════════════════
else:
    cam_id = 0 if source == "📹 Webcam 0" else 1
    cam_label = "Webcam 0" if cam_id == 0 else "Webcam 1 (Handy)"
    st.header(f"📹 {cam_label}")

    col_ctrl, col_info = st.columns([1, 2])
    with col_ctrl:
        start_btn = st.button(
            "▶️ Webcam starten" if not st.session_state.run_webcam else "⏹️ Stoppen",
            type="primary", use_container_width=True
        )
        if start_btn:
            st.session_state.run_webcam = not st.session_state.run_webcam
            st.rerun()

    with col_info:
        if cam_id == 1:
            st.info(
                "📱 **Handy als Webcam:** DroidCam oder EpocCam App installieren "
                "und Webcam-ID 1 auswählen."
            )

    if st.session_state.run_webcam:
        col1, col2 = st.columns([3, 1])

        frame_ph  = col1.empty()
        result_ph = col2.empty()
        card_ph   = col2.empty()
        save_ph   = col2.empty()

        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            st.error(f"❌ Webcam {cam_id} konnte nicht geöffnet werden.")
            st.session_state.run_webcam = False
            st.stop()

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        last_counts = {}
        last_passed = False
        frame_num   = 0
        stop_flag   = save_ph.button("⏹️ Webcam stoppen",
                                     use_container_width=True)

        while st.session_state.run_webcam and not stop_flag:
            ret, frame = cap.read()
            if not ret:
                st.warning("Kein Bild von Webcam empfangen.")
                break

            if frame_num % 2 == 0:
                ann, last_counts = run_inference(model, frame, conf, iou)
                last_passed = check_pass(last_counts, active_targets)

                frame_ph.image(
                    cv2.cvtColor(ann, cv2.COLOR_BGR2RGB),
                    caption=f"{cam_label} – live",
                    use_container_width=True,
                    channels="RGB",
                )
                with result_ph.container():
                    show_overall(last_passed)
                with card_ph.container():
                    show_result_cards(last_counts, active_targets)

                # Auto-Save bei Erfolg (einmaliges Auslösen)
                if last_passed:
                    save_ph.markdown(
                        '<div style="text-align:center;margin-top:8px">'
                        '<span style="font-size:1.1rem;color:#16a34a">'
                        '✅ Satz erkannt!</span></div>',
                        unsafe_allow_html=True
                    )
                    if save_ph.button("➕ Als Check speichern",
                                      key=f"save_{frame_num}",
                                      type="primary",
                                      use_container_width=True):
                        add_history(last_counts, active_targets,
                                    last_passed, cam_label)
                        cap.release()
                        st.session_state.run_webcam = False
                        st.rerun()

            frame_num += 1

        cap.release()
        st.session_state.run_webcam = False

