import gradio as gr
from scipy.io import wavfile
import noisereduce as nr
import whisper
import librosa
import os 
from pydub import AudioSegment
import pandas as pd 
from cleanunet.utils import CleanUNetDenoise
from cleanunet.config.cleanunet_constant import OUTPUT_PREDICTION_CLEANUNET_DIR
import torchaudio
import torch

PORT = 5003

modelname = "base"
model = whisper.load_model(modelname)

clean_unet = CleanUNetDenoise()
clean_unet.load_model(gpu=False)

def speech2text(audio):
    """ Simple Way """
    if audio is None:
        return ""
    
    result = model.transcribe(audio)

    # print(result)

    # save transcript to file
    with open("transcription.txt","w", encoding='utf-8') as txt:
        for segment in result['segments']:
            txt.write(segment['text'].strip())


    # Create empty lists to store the values
    ids = []
    t1_values = []
    t2_values = []
    text_values = []

    print("Detected language: ", result['language'])

    display_text = ""

    for segment in result['segments']:
        segment_id = segment['id']
        t1 = segment['start']
        t2 = segment['end']
        text = segment['text']

        # Append values to the respective lists
        ids.append(segment_id)
        t1_values.append(t1)
        t2_values.append(t2)
        text_values.append(text)

        display_text += f"{t1} : {t2}\t{text}\n"

    # Create a Pandas DataFrame
    data = {
        'id': ids,
        't1': t1_values,
        't2': t2_values,
        'text': text_values
    }

    df = pd.DataFrame(data)
    df.to_csv("transcription.csv", index=False)
    return display_text
    # return result['text']

# def speech2text(audio):
#     """ More fine detail way """
#     if audio is None:
#         return ""

#     # -1. Preprocessing audio
#     audio = whisper.load_audio(audio)
#     audio = whisper.pad_or_trim(audio)

#     # 0. make log-mel spectrogram and move the same device as model
#     mel = whisper.log_mel_spectrogram(audio).to(model.device)

#     # 1. detect the max probability of language ? 
#     _, probs = model.detect_language(mel)
#     language = max(probs, key=probs.get)

#     # 2. Decode audio to text
#     options = whisper.DecodingOptions(fp16=False)
#     result = whisper.decode(model, mel, options)

#     print(result.text)

#     return (language, result.text)

def reduce_noise_by_noise_reducer(audio_path):
    """ Reduce noise by using the Noise Reducer
    https://github.com/timsainb/noisereduce
    """
    print("Run reduce noise by noise reducer")

    HYPERPARAMTER = {
        'stationary': False,
        'prop_decrease': 0.9
    }

    # read audio
    sample_rate, audio_data = wavfile.read(audio_path)

    # perform noise reduction
    reduced_noise = nr.reduce_noise(y=audio_data, sr=sample_rate, stationary=HYPERPARAMTER['stationary'], prop_decrease = HYPERPARAMTER['prop_decrease'])
    basename = os.path.basename(audio_path).split(".")[0]
    output_path = f"output/{basename}.wav"
    wavfile.write(output_path, sample_rate, reduced_noise)

    audio_data, sample_rate = librosa.load(output_path, sr=None)
    return sample_rate, audio_data

def reduce_noise_by_cleanunet(audio_path):
    output_audio = []
    # load audio
    audio = torchaudio.load(audio_path)[0]
    # run the denoise
    output_audio.append(clean_unet.denoise_audio(audio))

    output_audio = torch.cat(output_audio, dim=1)

    print(output_audio)

    audio_data, sample_rate = librosa.load(audio_path, sr=None)
    return sample_rate, audio_data

def reduce_noise_by_denoiser(audio_path):
    audio_data, sample_rate = librosa.load(audio_path, sr=None)
    return sample_rate, audio_data

def segment_audio(audio_path, start_time, end_time):
    # work in miliseconds
    t1 = start_time * 60 * 1000 
    t2 = end_time * 60 * 1000

    input_audio = AudioSegment.from_wav(audio_path)
    output_audio = input_audio[t1:t2]
    output_audio.export("output/segmented_audio.wav", format="wav")
    audio_data, sample_rate = librosa.load("output/segmented_audio.wav", sr=None)
    return sample_rate, audio_data

example_files = [] 
for path in os.listdir("/home/jovyan/work/audio"):
    if os.path.basename(path).split('.')[1] in ['mp3','wav']:
        example_files.append(os.path.join("/home/jovyan/work/audio", path))

with gr.Blocks() as demo:
    with gr.Tab("Whisper"):
        gr.Markdown("Transcribe audio into text")
        with gr.Row():
            raw_audio_path = gr.Audio(source="upload", type='filepath')
            transcribed_output = gr.Textbox(label='Transcript')
        transcribe_button = gr.Button('Transcribe')

    with gr.Tab('Denoise'):
        gr.Markdown("## 1. Reduce Noise with NoiseReducer")
        with gr.Row():
            noise_audio_path_1 = gr.Audio(source="upload", type='filepath')
            reduced_noise_output_1 = gr.Audio()
        examples = gr.Examples(example_files, inputs = noise_audio_path_1)
        noisereducer_button = gr.Button('Run')

        gr.Markdown("## 2. Reduce Noise with CleanUNet- NVIDIA")
        with gr.Row():
            noise_audio_path_2 = gr.Audio(source="upload", type='filepath')
            reduced_noise_output_2 = gr.Audio()
        examples = gr.Examples(example_files, inputs = noise_audio_path_2)
        cleanunet_button = gr.Button('Run')

        gr.Markdown("## 3. Reduce Noise with Denoiser - Facebook")
        with gr.Row():
            noise_audio_path_3 = gr.Audio(source="upload", type='filepath')
            reduced_noise_output_3 = gr.Audio()
        examples = gr.Examples(example_files, inputs = noise_audio_path_3)
        denoiser_button = gr.Button('Run')

    with gr.Tab('Pyannote'):
        gr.Markdown('Pyannote')

    with gr.Tab("Whisper and Pyannote"):
        gr.Markdown("Combination of Whisper and Pyannote")

    with gr.Tab("Utils"):
        gr.Markdown("""
        ## Segment Audio
        Cut video by minute
        """)
        with gr.Row():
            with gr.Column():
                input_segment_audio_path = gr.Audio(source="upload", type='filepath')
            with gr.Column():
                start_time = gr.Number(label="Enter start time in minute")
                end_time = gr.Number(label="Enter end time in minute")
            segemented_audio = gr.Audio()
        segment_audio_button = gr.Button("Run")

    transcribe_button.click(speech2text, inputs=raw_audio_path, outputs = transcribed_output)
    noisereducer_button.click(reduce_noise_by_noise_reducer, inputs=noise_audio_path_1, outputs = reduced_noise_output_1)
    cleanunet_button.click(reduce_noise_by_cleanunet, inputs=noise_audio_path_2, outputs = reduced_noise_output_2)
    denoiser_button.click(reduce_noise_by_denoiser, inputs=noise_audio_path_3, outputs = reduced_noise_output_3)
    segment_audio_button.click(segment_audio, inputs=[input_segment_audio_path, start_time, end_time], outputs= segemented_audio)


demo.launch(server_port=PORT)
