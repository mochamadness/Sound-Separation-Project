from pydub import AudioSegment
import os

def split_audio(input_audio_path, output_dir, chunk_length_ms=10000, num_chunks=100):
    # Load the audio file
    audio = AudioSegment.from_file(input_audio_path)
    
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate the total length of the audio in milliseconds
    total_length_ms = len(audio)
    print(len(audio))
    # Calculate the number of chunks to create
    if total_length_ms < chunk_length_ms * num_chunks:
        num_chunks = total_length_ms // chunk_length_ms
        if total_length_ms % chunk_length_ms != 0:
            num_chunks += 1
    #audio = audio[30*1000:]
    # Split the audio into chunks and save them
    for i in range(num_chunks):
        start_time = i * chunk_length_ms
        end_time = min((i + 1) * chunk_length_ms, total_length_ms)
        chunk = audio[start_time:end_time]
        
        chunk_filename = f"{os.path.splitext(os.path.basename(input_audio_path))[0]}_{i + 1}.wav"
        chunk_filepath = os.path.join(output_dir, chunk_filename)
        #print(start_time, end_time)
        chunk.export(chunk_filepath, format="wav")
        print(f"Exported {chunk_filename}", start_time, end_time)


input_audio_file = "cleaned/2.wav"
output_directory = "cleaned/chunks/2"
num_chunks = 100
split_audio(input_audio_file, output_directory,10000, num_chunks)

print("Audio splitting complete.")