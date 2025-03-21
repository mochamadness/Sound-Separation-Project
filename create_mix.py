import torch
import torchaudio

def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    return waveform, sample_rate

def save_audio(file_path, waveform, sample_rate):
    # Ensure waveform is 2D for saving
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    torchaudio.save(file_path, waveform, sample_rate)

def convert_to_1defunct(waveform):
    return waveform.flatten()

def convert_to_1d(waveform):
    # If the waveform has more than one channel, average them to get a mono (1D) waveform
    if waveform.ndim == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0)
    return waveform

def resample_audio(waveform, original_sample_rate, target_sample_rate):
    resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
    return resampler(waveform)

def cut_audio(waveform, start_time, end_time, sample_rate):
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    return waveform[:, start_sample:end_sample]

def mix_audios(waveform1, waveform2):
    # Ensure both waveforms are 1D and have the same length
    min_length = min(len(waveform1), len(waveform2))
    waveform1 = waveform1[:min_length]
    waveform2 = waveform2[:min_length]
    # Mix the waveforms
    mixed_waveform = waveform1 + waveform2
    return mixed_waveform

# Example usage
input_audio_file1 = "test/1_slice_3.wav"
input_audio_file2 = "test/2_slice_4.wav"
output_audio_file = "test_in_1d/mix1_2.wav"
target_length_seconds = 10  # Desired length in seconds
target_sample_rate = 8000   # Desired sample rate in Hz

# Load the audio files
waveform1, original_sample_rate1 = load_audio(input_audio_file1)
waveform2, original_sample_rate2 = load_audio(input_audio_file2)

# Cut the first 5 seconds from each audio file
waveform1_cut = cut_audio(waveform1, 0, target_length_seconds, original_sample_rate1)
waveform2_cut = cut_audio(waveform2, 0, target_length_seconds, original_sample_rate2)

# Resample the audios to the target sample rate
waveform1_resampled = resample_audio(waveform1_cut, original_sample_rate1, target_sample_rate)
waveform2_resampled = resample_audio(waveform2_cut, original_sample_rate2, target_sample_rate)

# Convert the audios to 1D
waveform1_1d = convert_to_1d(waveform1_resampled)
waveform2_1d = convert_to_1d(waveform2_resampled)

# Mix the audios
mixed_waveform = mix_audios(waveform1_1d, waveform2_1d)

# Save the mixed audio
save_audio(output_audio_file, mixed_waveform, target_sample_rate)

print(f"Waveform 1 shape: {waveform1.shape}, Original sample rate: {original_sample_rate1}")
print(f"Waveform 2 shape: {waveform2.shape}, Original sample rate: {original_sample_rate2}")
print(f"Mixed waveform shape: {mixed_waveform.shape}, New sample rate: {target_sample_rate}")