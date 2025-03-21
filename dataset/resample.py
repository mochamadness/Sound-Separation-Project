from pydub import AudioSegment
import os

def resample_audio(input_audio_path, output_audio_path, target_sample_rate=8000):
    # Load the audio file
    audio = AudioSegment.from_file(input_audio_path)
    
    # Convert to mono (1D) if necessary
    if audio.channels > 1:
        audio = audio.set_channels(1)
    
    # Resample to target sample rate
    audio = audio.set_frame_rate(target_sample_rate)
    
    # Export the resampled audio
    audio.export(output_audio_path, format="wav")
    print(f"Exported {output_audio_path}")

def process_directory(input_dir, output_dir, target_sample_rate=8000):
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each file in the input directory
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".wav"):
                input_audio_path = os.path.join(root, file)
                output_audio_path = os.path.join(output_dir, file)
                
                # Resample the audio file
                resample_audio(input_audio_path, output_audio_path, target_sample_rate)

# Example usage
input_directory = "narrator_4_slices"
output_directory = "cleaned/4"
process_directory(input_directory, output_directory)

print("Audio processing complete.")