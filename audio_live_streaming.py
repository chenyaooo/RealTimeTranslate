import gradio as gr
from transformers import pipeline
import numpy as np

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

def stream_bytes(audio_file):
   return audio_file

def transcribe(audio):
    sr, y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    return transcriber({"sampling_rate": sr, "raw": y})["text"]


# use audio input streaming
# output live
demo = gr.Interface(
    fn=transcribe, 
    # inputs=gr.Audio(label="Input Audio", type="filepath", format="mp3"), 
    inputs=gr.Audio(sources=['microphone'], streaming=True), 
    outputs=[gr.Audio(streaming=True, autoplay=True), "text"],
    live=True
) 

demo.launch()