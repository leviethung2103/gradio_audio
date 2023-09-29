import sys
import torch
import json
from cleanunet.network import CleanUNet
from cleanunet.config.cleanunet_constant import CLEANUNET_CHECKPOINT, CLEANUNET_CONFIG

class CleanUNetDenoise:
    def __init__(self, gpu=False):
        # self.net = None
        with open(CLEANUNET_CONFIG) as f:
            data = f.read()
        self.gpu = gpu
        self.config = json.loads(data)
        self.network_config = self.config["network_config"]
        
    def load_model(self,gpu=False):
        """ Load CleanUNet pretrained model """
        if gpu:
            self.net = CleanUNet(**self.network_config).cuda()
        else:
            self.net = CleanUNet(**self.network_config)
        checkpoint = torch.load(CLEANUNET_CHECKPOINT, map_location='cpu')
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.net.eval()
    
    def denoise_audio(self, audio):
        """ Fetch the audio input to denoise """
        if self.gpu:
            audio = audio.cuda()
            print("Using audio")
        print(audio)
        print(self.net)
        generated_audio = self.net(audio)
        print("Result")
        generated_audio = generated_audio[0].detach().cpu()
        return generated_audio