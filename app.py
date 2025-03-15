import streamlit as st
import numpy as np
import soundfile as sf
from io import BytesIO
from sound_sep import SoundSeparation

# Initialize the SoundSeparation class
sound_sep = SoundSeparation(config_path="config.yaml")

st.title("Audio Source Separation")

# Upload audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)

process_button = st.button("Start Processing")

if uploaded_file is not None and process_button:
    # Read the audio file
    mixture, original_sample_rate = sound_sep.read_audio_file(uploaded_file)

    # Separate the sound
    diarization, sources_hat = sound_sep.separate_sound(mixture, original_sample_rate)

    # Display the number of sources
    num_sources = sources_hat.getDimension()
    st.write(f"Number of sources extracted: {num_sources}")

    # Display each source
    for i in range(num_sources):
        source = sources_hat[:, i]
        st.write(f"Source {i+1}")
        
        # Create a BytesIO object to store the audio
        audio_buffer = BytesIO()
        sf.write(audio_buffer, source, sound_sep.default_sample_rate, format='wav')
        audio_buffer.seek(0)
        
        # Display the audio player
        st.audio(audio_buffer, format='audio/wav')
        # display probability of gender speaker




# Run the Streamlit app
# To run the app, use the command: streamlit run app.py