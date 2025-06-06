import sys, audioop
sys.modules['pyaudioop'] = audioop

from pathlib import Path
from datetime import datetime
import time
import queue

import tempfile
import os
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

import openai
import pydub
import whisper  # Adicionado
from moviepy.video.io.VideoFileClip import VideoFileClip
from dotenv import load_dotenv, find_dotenv
from openai import RateLimitError

_ = load_dotenv(find_dotenv())

# Diretórios
PASTA_TEMP = Path(__file__).parent / 'temp'
PASTA_TEMP.mkdir(exist_ok=True)
PASTA_TRANSCRICOES = Path(__file__).parent / 'TRANSCRICOES'
PASTA_TRANSCRICOES.mkdir(exist_ok=True)

# Arquivos temporários
ARQUIVO_AUDIO_TEMP = PASTA_TEMP / 'audio.wav'
ARQUIVO_VIDEO_TEMP = PASTA_TEMP / 'video.mp4'
ARQUIVO_MIC_TEMP = PASTA_TEMP / 'mic.wav'

# Cliente OpenAI
client = openai.OpenAI()

# Whisper local
local_model = None
def get_local_whisper():
    global local_model
    if local_model is None:
        local_model = whisper.load_model("base")
    return local_model

# Retry
MAX_RETRIES = 3
RETRY_DELAY = 2

def handle_openai_error(func):
    def wrapper(*args, **kwargs):
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except RateLimitError:
                if attempt == MAX_RETRIES - 1:
                    st.warning("Limite de taxa da API OpenAI atingido. Usando serviço local.")
                    return use_fallback_service(*args, **kwargs)
                else:
                    time.sleep(RETRY_DELAY * (attempt + 1))
            except Exception as e:
                st.error(f"Erro: {str(e)}")
                return None
    return wrapper

@handle_openai_error
def processa_transcricao_chatgpt(texto: str) -> str:
    ASSISTANT_ID = "asst_IIeBxLET5NSbEzVpFs4xbCrP"
    thread = client.beta.threads.create()
    client.beta.threads.messages.create(thread_id=thread.id, role="user", content=texto)
    run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=ASSISTANT_ID)
    while True:
        run_status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if run_status.status in ['completed', 'failed', 'cancelled']:
            break
        time.sleep(1)
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    for message in reversed(messages.data):
        if message.role == "assistant":
            return "\n\n".join([part.text.value for part in message.content])
    return "Nenhuma resposta do assistente foi encontrada."

def use_fallback_service(caminho_audio=None, prompt=None, texto=None):
    try:
        if caminho_audio:
            model = get_local_whisper()
            result = model.transcribe(caminho_audio, language="pt")
            return result["text"], "Análise não disponível (serviço local)"
        elif texto:
            return texto, "Análise não disponível (serviço local)"
    except Exception as e:
        st.error(f"Erro no fallback: {str(e)}")
        return "", ""

def converter_para_wav(caminho_entrada: str) -> str:
    audio = pydub.AudioSegment.from_file(caminho_entrada)
    fd, caminho_wav = tempfile.mkstemp(suffix=".wav", prefix="audio_")
    os.close(fd)
    audio.export(caminho_wav, format="wav")
    return caminho_wav

@handle_openai_error
def transcreve_audio(caminho_audio: str, prompt: str) -> tuple[str, str]:
    prompt = prompt or "Transcrição de atendimento jurídico"
    with open(caminho_audio, 'rb') as arquivo:
        try:
            resp = client.audio.transcriptions.create(
                model='whisper-1', language='pt', response_format='text',
                file=arquivo, prompt=prompt,
            )
            analise = processa_transcricao_chatgpt(resp.text)
            return resp.text, analise
        except Exception as e:
            st.warning(f"Erro API OpenAI: {str(e)}. Usando fallback.")
            return use_fallback_service(caminho_audio, prompt)

# Estado inicial para transcrição do microfone
if 'transcricao_mic' not in st.session_state:
    st.session_state['transcricao_mic'] = ''
if 'analise_mic' not in st.session_state:
    st.session_state['analise_mic'] = ''
if 'gravando_audio' not in st.session_state:
    st.session_state['gravando_audio'] = False
if 'audio_completo' not in st.session_state:
    st.session_state['audio_completo'] = pydub.AudioSegment.empty()

# Cache para configuração de ICE servers
@st.cache_data
def get_ice_servers():
    return [{'urls': ['stun:stun.l.google.com:19302']}]

# Concatena frames em AudioSegment
def adiciona_chunck_de_audio(frames, chunk_audio):
    for frame in frames:
        seg = pydub.AudioSegment(
            data=frame.to_ndarray().tobytes(),
            sample_width=frame.format.bytes,
            frame_rate=frame.sample_rate,
            channels=len(frame.layout.channels)
        )
        chunk_audio += seg
    return chunk_audio

def salva_transcricao(texto: str, analise: str, origem: str = ""):
    agora = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    prefixo = f"{agora}_{origem}" if origem else agora
    arquivo_transcricao = PASTA_TRANSCRICOES / f"{prefixo}_transcricao.txt"
    with open(arquivo_transcricao, 'w', encoding='utf-8') as f:
        f.write(texto)
    arquivo_analise = PASTA_TRANSCRICOES / f"{prefixo}_analise.txt"
    with open(arquivo_analise, 'w', encoding='utf-8') as f:
        f.write(analise)
    return arquivo_transcricao, arquivo_analise
