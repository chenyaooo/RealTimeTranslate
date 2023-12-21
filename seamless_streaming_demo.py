import torch
from fairseq2.data.audio import AudioDecoder, WaveformToFbankConverter
from fairseq2.memory import MemoryBlock
from fairseq2.nn.padding import get_seqs_and_padding_mask
from pathlib import Path
from seamless_communication.models.conformer_shaw import load_conformer_shaw_model
import gradio as gr
import openai
from langchain.llms import OpenAI
import os
from dotenv import load_dotenv
from gtts import gTTS
import tempfile
from module_name import collater


# 加载环境变量
load_dotenv()

def process_audio(audio_wav_path, device='cpu', dtype=torch.float32):
    # 音频解码和特征提取
    audio_decoder = AudioDecoder()
    fbank_converter = WaveformToFbankConverter()
    # 加载模型
    model = load_conformer_shaw_model("conformer_shaw", device=device, dtype=dtype)
    model.eval()

    with Path(audio_wav_path).open("rb") as fb:
        block = MemoryBlock(fb.read())

    decoded_audio = audio_decoder(block)
    src = collater(fbank_converter(decoded_audio))["fbank"]
    seqs, padding_mask = get_seqs_and_padding_mask(src)

    with torch.inference_mode():
        seqs, padding_mask = model.encoder_frontend(seqs, padding_mask)
        seqs, padding_mask = model.encoder(seqs, padding_mask)
    

    return seqs, padding_mask


os.environ["OPENAI_API_KEY"] = "sk-3HZZISYiQTpCDTqg2XuuT3BlbkFJA6KViB5KV57gANO"


def process(filepath):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OpenAI API key is not set in environment variables.")

    
    text = process_audio(filepath)

    llm = OpenAI(temperature=1, openai_api_key=openai_api_key)

    prompt_for_translation = f"Translate this to Japanese: {text}"
    translation_response = llm(prompt_for_translation)

    tts = gTTS(translation_response, lang='ja')
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio_file:
        tts.save(temp_audio_file.name)
        return temp_audio_file.name

demo = gr.Interface(
    fn=process,
    inputs=gr.Audio(sources="microphone", type="filepath"),
    outputs=gr.Audio()
)

demo.launch(share=True)