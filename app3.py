import gradio as gr
# from openai import OpenAI
import openai

from langchain.llms import OpenAI
import os
from dotenv import load_dotenv
import subprocess  # To run the whisper command
from gtts import gTTS
from io import BytesIO
import tempfile
load_dotenv()




def process(filepath):
    # set openai key
    client = openai.OpenAI(api_key="your-key")
    my_key = "your-key"

    # Store the voice input to a file
    audio = open(str(filepath), "rb")

    # openai speech-to-text
    transcript =client.audio.transcriptions.create(
        model="whisper-1",
        file=audio)

    # langchain
    llm = OpenAI(temperature=1, openai_api_key = my_key)

    # Construct the prompt for translation
    prompt_for_translation = f"Translate this to Japanese: {transcript}"


    # Get the translation response
    translation_response = llm(prompt_for_translation)
    # Convert translated text to speech
    tts = gTTS(translation_response, lang='ja')  # 'ja' is the language code for Japanese
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)  # Rewind the file

    # Convert translated text to speech
    tts = gTTS(translation_response, lang='ja')
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)  # Rewind the file

    # Convert BytesIO to bytes
    audio_bytes = mp3_fp.getvalue()
    tts = gTTS(translation_response, lang='ja')  # 'ja' is the language code for Japanese

    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio_file:
        tts.save(temp_audio_file.name)
        return temp_audio_file.name  # Return the path of the temp file

    return audio_bytes

    # Return the audio file
    #return mp3_fp
    #return translation_response  
    # 

# Update the outputs in your Gradio Interface to return audio
demo = gr.Interface(
    fn=process, 
    inputs=gr.Audio(sources="microphone", type="filepath"), 
    outputs=gr.Audio()
) 
# demo = gr.Interface(
#     fn=process, 
#     inputs=gr.Audio(sources="microphone", type="filepath"), 
#     outputs="text"
#)

demo.launch()
