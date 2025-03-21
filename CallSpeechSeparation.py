from SpeechSeparation import SpeechSeparation
import os
import torch
from AudioReader import AudioReader, write_wav, read_wav

def main():
    # Define the paths and GPU IDs
    mix_scp = 'test_in_1d/1n4_1.wav'
    model_checkpoint = 'model/best.pt'
    gpuid = "0, 1, 2, 3, 4, 5, 6, 7"
    save_path = './checkpoint'
    yaml_path = 'options/train/train.yml'
    

    #missing normalize audio, might add later
    # Initialize the SpeechSeparation class
    separator = SpeechSeparation(model_checkpoint, yaml_path, gpuid)
    
    # Perform the separation
    mix = read_wav(mix_scp)
    spks = separator.separate_audio(mix)
    
    # Save separated signals
    for idx, spk in enumerate(spks):
        os.makedirs(save_path, exist_ok=True)
        filename = os.path.join(save_path, f'spk{idx+1}.wav')
        write_wav(filename, spk.unsqueeze(0), 8000)
        print(f'Speaker {idx+1} saved to {filename}')

if __name__ == "__main__":
    main()