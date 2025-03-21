import streamlit as st
import numpy as np
import librosa
import tempfile
import soundfile as sf
import matplotlib.pyplot as plt
from Utils_ import create_model
import noisereduce as nr
import pyaudio
import wave
import torch
import torchaudio
from io import BytesIO
import os
from AudioReader import read_wav
import SpeechSeparation

# Config
st.set_page_config(page_title="Voice Separation & Gender", page_icon="🎙️", layout="wide")
st.title("🎙️ Voice Separation & Gender")
st.markdown("""
<style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
</style>
""", unsafe_allow_html=True)

# Load Gender Recognition Model
@st.cache_resource
def load_gender_model():
    model = create_model()
    model.load_weights("model/model.h5")
    return model

if "gender_model" not in st.session_state:
    st.session_state.gender_model = load_gender_model()

# Load Speech Separation Model
model_checkpoint = 'model/best.pt'
gpuid = "0, 1, 2, 3, 4, 5, 6, 7"
yaml_path = 'options/train/train.yml'
model = SpeechSeparation.SpeechSeparation(model_checkpoint, yaml_path, gpuid)

# Utility functions
def record_audio(duration=10, sample_rate=16000):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=sample_rate, input=True, frames_per_buffer=CHUNK)
    st.info("🎙️ Đang ghi âm... Vui lòng nói trong vài giây.")
    frames = [stream.read(CHUNK) for _ in range(0, int(sample_rate / CHUNK * duration))]
    stream.stop_stream()
    stream.close()
    p.terminate()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
        wf = wave.open(temp_wav, "wb")
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))
        return temp_wav.name

def resample_audio(waveform, orig_sr, target_sr=8000):
    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
    return resampler(waveform)

def get_audio_bytes(tensor, sample_rate):
    audio_bytes = BytesIO()
    torchaudio.save(audio_bytes, tensor, sample_rate, format="wav")
    audio_bytes.seek(0)
    return audio_bytes

def normalize_audio(audio, norm):
    return audio * norm / torch.max(torch.abs(audio))

def denoise_audio(audio, sr):
    return nr.reduce_noise(y=audio, sr=sr)

def extract_feature(file_name, **kwargs):
    try:
        mfcc = kwargs.get("mfcc")
        chroma = kwargs.get("chroma")
        mel = kwargs.get("mel")
        contrast = kwargs.get("contrast")
        tonnetz = kwargs.get("tonnetz")
        X, sample_rate = librosa.load(file_name, sr=None)
        X = denoise_audio(X, sample_rate)
        result = np.array([])
        stft = np.abs(librosa.stft(X)) if chroma or contrast else None
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma_feature = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma_feature))
        if mel:
            mel_feature = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel_feature))
        if contrast:
            contrast_feature = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, contrast_feature))
        if tonnetz:
            tonnetz_feature = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
            result = np.hstack((result, tonnetz_feature))
        return result
    except Exception as e:
        st.error(f"❌ Lỗi khi trích xuất đặc trưng: {str(e)}")
        return None

def plot_waveform(file_path):
    y, sr = librosa.load(file_path, sr=None)
    fig, ax = plt.subplots(figsize=(8, 2))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set(title="Waveform")
    st.pyplot(fig)

def plot_spectrogram(file_path):
    y, sr = librosa.load(file_path)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(8, 3))
    img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    ax.set(title='Mel Spectrogram')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    st.pyplot(fig)

def predict_gender(file_path):
    features = extract_feature(file_path, mel=True)
    if features is not None:
        features = features.reshape(1, -1)
        expected_features = 128
        if features.shape[1] != expected_features:
            return "⚠️ Dữ liệu đầu vào không hợp lệ."
        else:
            model = st.session_state.gender_model
            male_prob = model.predict(features)[0][0]
            female_prob = 1 - male_prob
            gender = "🧑 Nam" if male_prob > female_prob else "👩 Nữ"
            return f"**Giới tính:** {gender}  \\ 🔵 Nam: {male_prob*100:.2f}% | 🔴 Nữ: {female_prob*100:.2f}%"
    return "❌ Không thể dự đoán."

# Input options
with st.sidebar:
    st.header("🎧 Chọn nguồn âm thanh")
    option = st.radio("Nguồn:", ["🎤 Ghi âm trực tiếp", "📂 Tải file WAV"])
    duration = st.slider("⏱️ Thời lượng ghi âm (giây):", min_value=1, max_value=15, value=5)
    uploaded_path = None
    if option == "🎤 Ghi âm trực tiếp":
        if st.button("🎙️ Bắt đầu ghi âm"):
            uploaded_path = record_audio(duration=duration)
            st.success("✅ Đã ghi âm xong!")
    elif option == "📂 Tải file WAV":
        uploaded_file = st.file_uploader("📤 Chọn file WAV", type=["wav"])
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
                temp_wav.write(uploaded_file.read())
                uploaded_path = temp_wav.name

# Process audio
if uploaded_path:
    st.subheader("🎧 Audio Gốc")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.audio(uploaded_path, format="audio/wav")
        plot_waveform(uploaded_path)
    with col2:
        plot_spectrogram(uploaded_path)

    mix, sr = read_wav(uploaded_path, return_rate=True)
    if mix.ndim > 1:
        mix = mix.flatten()
    mix = resample_audio(mix, sr, target_sr=8000)
    separated_audios = model.separate_audio(mix)
    norm = torch.norm(mix, float('inf'))

    for idx, spk in enumerate(separated_audios):
        spk = normalize_audio(spk, norm)
        audio_bytes = get_audio_bytes(spk.unsqueeze(0), 8000)
        st.markdown(f"### 🔊 Giọng tách {idx+1}")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.audio(audio_bytes, format="audio/wav")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
                torchaudio.save(temp.name, spk.unsqueeze(0), 8000)
                plot_waveform(temp.name)
        with col2:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
                torchaudio.save(temp.name, spk.unsqueeze(0), 8000)
                plot_spectrogram(temp.name)
                st.markdown(predict_gender(temp.name))

                
# streamlit run app.py