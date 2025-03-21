import torch
import torchaudio
from pydub import AudioSegment

def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    return waveform, sample_rate

def save_audio(file_path, waveform, sample_rate):
    # Ensure waveform is 2D for saving
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    torchaudio.save(file_path, waveform, sample_rate)

def convert_to_1d(waveform):
    return waveform.flatten()

def resample_audio(waveform, original_sample_rate, target_sample_rate):
    resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
    return resampler(waveform)

def change_audio_length(waveform, target_length, sample_rate):
    # Calculate the number of samples needed for the target length
    num_samples = int(target_length * sample_rate)
    if waveform.size(1) < num_samples:
        # If the waveform is shorter, pad it with zeros
        padding = torch.zeros(1, num_samples - waveform.size(1))
        waveform = torch.cat((waveform, padding), dim=1)
    else:
        # If the waveform is longer, truncate it
        waveform = waveform[:, :num_samples]
    return waveform

# Example usage
input_audio_file = "test/cut_small_talk.wav"
output_audio_file = "test_in_1d/cut_small_talk.wav"
target_length_seconds = 10  # Desired length in seconds
target_sample_rate = 8000   # Desired sample rate in Hz

# Load the audio file
waveform, original_sample_rate = load_audio(input_audio_file)

# Convert the audio to 1D
waveform_1d = convert_to_1d(waveform)

# Resample the audio to the target sample rate
waveform_resampled = resample_audio(waveform_1d.unsqueeze(0), original_sample_rate, target_sample_rate).squeeze(0)

# Change the length of the audio
waveform_resampled_length = change_audio_length(waveform_resampled.unsqueeze(0), target_length_seconds, target_sample_rate).squeeze(0)

# Save the processed audio
save_audio(output_audio_file, waveform_resampled_length, target_sample_rate)

print(f"Original shape: {waveform.shape}, Original sample rate: {original_sample_rate}")
print(f"Processed shape: {waveform_resampled_length.shape}, New sample rate: {target_sample_rate}")