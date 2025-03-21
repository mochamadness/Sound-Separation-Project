import os
import torch
import sys
sys.path.append('./options')
from AudioReader import read_wav, write_wav
from Conv_TasNet import ConvTasNet
from utils import get_logger
from option import parse

class SpeechSeparation():
    def __init__(self, model_checkpoint, yaml_path, gpuid):
        """
        Initialize the SpeechSeparation class.
        
        Args:
            model_checkpoint (str): Path to the model checkpoint file.
            yaml_path (str): Path to the YAML configuration file.
            gpuid (list of int): List of GPU IDs to use.
        """
        if torch.cuda.is_available():
            self.device = torch.device('cuda:{}'.format(gpuid[0]) if len(gpuid) > 0 else 'cpu')
            self.gpuid = tuple(gpuid)
        else:
            self.device = torch.device('cpu')
            self.gpuid = ()
        self.logger = get_logger(__name__)
        self.net = self.load_model(model_checkpoint, yaml_path)
        
    def load_model(self, model_checkpoint, yaml_path):
        """
        Load the ConvTasNet model from the checkpoint.
        
        Args:
            model_checkpoint (str): Path to the model checkpoint file.
            yaml_path (str): Path to the YAML configuration file.
        
        Returns:
            torch.nn.Module: Loaded ConvTasNet model.
        """
        dicts = torch.load(model_checkpoint, map_location='cpu')
        opt = parse(yaml_path, is_tain=False)
        net = ConvTasNet(**opt['net_conf'])
        net.load_state_dict(dicts["model_state_dict"])
        self.logger.info('Load checkpoint from {}, epoch {:d}'.format(model_checkpoint, dicts["epoch"]))
        return net.to(self.device)
    
    def separate(self, mix_path):
        """
        Separate the input mixture audio into two sources.
        
        Args:
            mix_path (str): Path to the input mixture audio file.
        
        Returns:
            tuple: Two separated audio signals as PyTorch tensors.
        """
        # Check if the file exists
        if not os.path.isfile(mix_path):
            raise FileNotFoundError(f"File {mix_path} does not exist")

        mix = read_wav(mix_path)
        mix = mix.to(self.device)
        
        with torch.no_grad():
            ests = self.net(mix)
            spks = [torch.squeeze(s.detach().cpu()) for s in ests]
        
        return spks
    
    def separate_audio(self, audio_tensor):
        """
        Separate the input audio tensor into two sources.
        
        Args:
            audio_tensor (torch.Tensor): Input audio tensor.
        
        Returns:
            tuple: Two separated audio signals as PyTorch tensors.
        """
        audio_tensor = audio_tensor.to(self.device)
        
        with torch.no_grad():
            ests = self.net(audio_tensor)
            spks = [torch.squeeze(s.detach().cpu()) for s in ests]
        
        return spks