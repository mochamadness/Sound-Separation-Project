from pydub import AudioSegment
import os

def ensure_mono(audio_segment):
    if audio_segment.channels > 1:
        audio_segment = audio_segment.set_channels(1)
    return audio_segment

def mix_audio_chunks(voice1_dir, voice2_dir, output_dir, num_chunks=100):
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(1, num_chunks + 1):
        # Load the corresponding chunks from both directories
        voice1_chunk_path = os.path.join(voice1_dir, f"1_{i}.wav")
        voice2_chunk_path = os.path.join(voice2_dir, f"4_{i}.wav")
        
        if os.path.exists(voice1_chunk_path) and os.path.exists(voice2_chunk_path):
            voice1_chunk = AudioSegment.from_file(voice1_chunk_path)
            voice2_chunk = AudioSegment.from_file(voice2_chunk_path)
            
            # Ensure both chunks are mono (1D)
            voice1_chunk = ensure_mono(voice1_chunk)
            voice2_chunk = ensure_mono(voice2_chunk)
            
            # Mix the audio chunks
            mixed_chunk = voice1_chunk.overlay(voice2_chunk)
            
            # Save the mixed chunk
            output_chunk_path = os.path.join(output_dir, f"1n4_{i}.wav")
            mixed_chunk.export(output_chunk_path, format="wav")
            print(f"Exported {output_chunk_path}")
        else:
            print(f"Missing chunk: 1_{i}.wav or 4_{i}.wav")

# Example usage
voice1_directory = "cleaned/chunks/1"
voice2_directory = "cleaned/chunks/4"
output_directory = "cleaned/chunks/mix1_4"
num_chunks = 100
mix_audio_chunks(voice1_directory, voice2_directory, output_directory, num_chunks)

print("Audio mixing complete.")