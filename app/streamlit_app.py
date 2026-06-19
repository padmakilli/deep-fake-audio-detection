"""Streamlit web app: upload audio, get Genuine/Deepfake + confidence."""
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import streamlit as st

from deepfake_audio.inference import load_model, predict_file
from deepfake_audio.utils.audio import load_audio

CHECKPOINT = os.environ.get("DFA_CHECKPOINT", "models/best_model.pt")
CONFIG = os.environ.get("DFA_CONFIG", "config/config.yaml")

st.set_page_config(page_title="Deepfake Audio Detector", page_icon="🎙️", layout="centered")
st.title("🎙️ Deepfake Audio Detector")
st.caption("Upload a speech recording — the model classifies it as Genuine (Human) or "
           "Deepfake (AI-Generated) and reports a confidence score.")


@st.cache_resource(show_spinner=True)
def _get_model():
    return load_model(CHECKPOINT, CONFIG)


uploaded = st.file_uploader("Audio file", type=["wav", "flac", "mp3", "ogg", "m4a"])

if uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name
    st.audio(uploaded)

    if not os.path.exists(CHECKPOINT):
        st.error(f"No trained model at '{CHECKPOINT}'. Train first: `python scripts/train.py`.")
    else:
        with st.spinner("Analysing..."):
            model, cfg, threshold = _get_model()
            result = predict_file(tmp_path, model, cfg, threshold)

        genuine = result["is_genuine"]
        st.markdown(f"### {'✅' if genuine else '⚠️'} {result['label']}")
        st.metric("Confidence", f"{result['confidence']:.2f}%")
        st.progress(min(1.0, result["confidence"] / 100.0))
        with st.expander("Details"):
            st.json(result)

        try:
            import librosa, librosa.display
            import matplotlib.pyplot as plt
            y = load_audio(tmp_path, cfg.audio.sample_rate)
            mel = librosa.power_to_db(
                librosa.feature.melspectrogram(
                    y=y, sr=cfg.audio.sample_rate, n_mels=cfg.features.n_mels,
                    n_fft=cfg.features.n_fft, hop_length=cfg.features.hop_length), ref=np.max)
            fig, ax = plt.subplots(figsize=(7, 3))
            librosa.display.specshow(mel, sr=cfg.audio.sample_rate,
                                     hop_length=cfg.features.hop_length,
                                     x_axis="time", y_axis="mel", ax=ax)
            ax.set_title("Log-mel spectrogram")
            st.pyplot(fig)
        except Exception:
            pass
    os.unlink(tmp_path)
else:
    st.info("Upload a .wav (or flac/mp3/ogg/m4a) file to begin.")
