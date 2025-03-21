from pydub import AudioSegment
import os

def split_audio_into_chunks(input_dir, output_dir, chunk_length_ms=10000, max_chunk=100):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    num_chunks = 0  # Initialize chunk counter
    
    # Process each file in the input directory
    for root, dirs, files in os.walk(input_dir):
        files.sort()
        for file in files:
            if file.endswith(".wav"):
                input_audio_path = os.path.join(root, file)
                
                # Load the audio file
                audio = AudioSegment.from_file(input_audio_path)
                
                # Calculate the total length of the audio in milliseconds
                total_length_ms = len(audio)
                
                # Split the audio into 10-second chunks
                for start_time in range(0, total_length_ms, chunk_length_ms):
                    end_time = min(start_time + chunk_length_ms, total_length_ms)
                    chunk = audio[start_time:end_time]
                    
                    # Increment the chunk counter
                    num_chunks += 1
                    
                    # Save the chunk with the naming convention 4_numchunks.wav
                    chunk_filename = f"4_{num_chunks}.wav"
                    chunk_filepath = os.path.join(output_dir, chunk_filename)
                    chunk.export(chunk_filepath, format="wav")
                    print(f"Exported {chunk_filepath}")
                    if(num_chunks>max_chunk): break

# Example usage
input_directory = "cleaned/4"  # Replace with the actual path to the input directory
output_directory = "cleaned/chunks/4"  # Replace with the actual path to the output directory
split_audio_into_chunks(input_directory, output_directory)

print("Audio splitting complete.")