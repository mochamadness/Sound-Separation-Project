import torch
import torchaudio
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os

def remove_silence(audio_path, silence_thresh=-50, min_silence_len=1000, keep_silence=500):
    # Load the audio file
    audio = AudioSegment.from_file(audio_path)

    # Split the audio file on silences
    chunks = split_on_silence(audio, 
                              min_silence_len=min_silence_len, 
                              silence_thresh=silence_thresh, 
                              keep_silence=keep_silence)

    # Concatenate the chunks back together
    processed_audio = AudioSegment.empty()
    for chunk in chunks:
        processed_audio += chunk

    return processed_audio

def save_temp_audio(audio, file_path):
    audio.export(file_path, format="wav")

def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    return waveform, sample_rate

def save_audio(file_path, waveform, sample_rate):
    # Ensure waveform is 2D for saving
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    torchaudio.save(file_path, waveform, sample_rate)

def convert_to_1d(waveform):
    # If the waveform has more than one channel, average them to get a mono (1D) waveform
    if waveform.ndim == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0)
    return waveform

def resample_audio(waveform, original_sample_rate, target_sample_rate):
    resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
    return resampler(waveform)

def process_audio(input_audio_file, output_audio_file, target_sample_rate=8000):
    temp_file = "temp_processed_audio.wav"

    # Remove silence
    processed_audio = remove_silence(input_audio_file)

    # Save the processed audio temporarily
    save_temp_audio(processed_audio, temp_file)

    # Load the processed audio using torchaudio
    waveform, original_sample_rate = load_audio(temp_file)

    # Resample the audio to the target sample rate
    waveform_resampled = resample_audio(waveform, original_sample_rate, target_sample_rate)

    # Convert the audio to 1D (mono)
    waveform_1d = convert_to_1d(waveform_resampled)

    # Save the final processed audio
    save_audio(output_audio_file, waveform_1d, target_sample_rate)

    # Clean up the temporary file
    os.remove(temp_file)

print("Audio processing complete.")
input_audio_file = "dataset/LAM ĐI - VU TRONG PHUNG - VAN HOC VIET NAM - HEM RADIO - TRẠM DỪNG 1080.mp3"
output_audio_file = "dataset/cleaned/3.wav"
process_audio(input_audio_file, output_audio_file)

print("Audio processing complete.")
input_audio_file = "dataset/NHỮNG NGƯỜI KHỐN KHỔ - VICTOR HUGO - HẺM RADIO - TRẠM DỪNG 1080.mp3"
output_audio_file = "dataset/cleaned/4.wav"
process_audio(input_audio_file, output_audio_file)

print("Audio processing complete.")
input_audio_file = "dataset/TRUYỆN KIỀU (trọn bộ) - Nguyễn Du.mp3"
output_audio_file = "dataset/cleaned/4.wav"
process_audio(input_audio_file, output_audio_file)