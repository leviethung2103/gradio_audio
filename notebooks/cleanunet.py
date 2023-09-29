import sys
sys.path.insert(0,'/home/jovyan/work/gradio')
from cleanunet.utils import CleanUNetDenoise
from cleanunet.config.cleanunet_constant import OUTPUT_PREDICTION_CLEANUNET_DIR
import torch
import torchaudio

# Load noise reduction for pin mic - load model CleanUNet
clean_unet = CleanUNetDenoise()
clean_unet.load_model()

output_audio = []
audio_path = '/home/jovyan/work/audio/Noise_TestCase1.wav'
audio = torchaudio.load(audio_path)[0]
output_audio.append(clean_unet.denoise_audio(audio))