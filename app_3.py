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
from moviepy.video.io.VideoFileClip import VideoFileClip
from dotenv import load_dotenv, find_dotenv
from openai import RateLimitError

_ = load_dotenv(find_dotenv())

# Diretório temporário
PASTA_TEMP = Path(__file__).parent / 'temp'
PASTA_TEMP.mkdir(exist_ok=True)

# Diretório de transcrições
PASTA_TRANSCRICOES = Path(__file__).parent / 'TRANSCRICOES'
PASTA_TRANSCRICOES.mkdir(exist_ok=True)

# Arquivos temporários
ARQUIVO_AUDIO_TEMP = PASTA_TEMP / 'audio.wav'
ARQUIVO_VIDEO_TEMP = PASTA_TEMP / 'video.mp4'
ARQUIVO_MIC_TEMP = PASTA_TEMP / 'mic.wav'

# Cliente OpenAI
client = openai.OpenAI()

# Modelo Whisper local para fallback
local_model = None

def get_local_whisper():
    global local_model
    if local_model is None:
        import whisper
        local_model = whisper.load_model("base")
    return local_model

# Configurações de retry
MAX_RETRIES = 3
RETRY_DELAY = 2  # segundos

def handle_openai_error(func):
    def wrapper(*args, **kwargs):
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except RateLimitError:
                if attempt == MAX_RETRIES - 1:
                    st.warning("OpenAI API rate limit atingido. Usando serviço local de fallback.")
                    return use_fallback_service(*args, **kwargs)
                else:
                    time.sleep(RETRY_DELAY * (attempt + 1))
            except Exception as e:
                st.error(f"Erro ao processar: {str(e)}")
                return None
    return wrapper

def use_fallback_service(caminho_audio=None, prompt=None, texto=None):
    try:
        if caminho_audio:
            model = get_local_whisper()
            result = model.transcribe(caminho_audio, language="pt")
            return result["text"], "Análise não disponível (usando serviço local)"
        elif texto:
            return texto, "Análise não disponível (usando serviço local)"
    except Exception as e:
        st.error(f"Erro no serviço de fallback: {str(e)}")
        return "", ""

PROMPT_PSICOLOGICO = ''' ... '''
PROMPT_JURIDICO = ''' ... '''
PROMPT_SERVICO_SOCIAL = ''' ... '''

PROMPTS = {
    "Psicológico": PROMPT_PSICOLOGICO,
    "Jurídico": PROMPT_JURIDICO,
    "Serviço Social": PROMPT_SERVICO_SOCIAL
}

@handle_openai_error
def processa_transcricao_chatgpt(texto: str) -> str:
    resposta = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": texto}
        ]
    )
    return resposta.choices[0].message.content

def converter_para_wav(caminho_entrada: str) -> str:
    audio = pydub.AudioSegment.from_file(caminho_entrada)
    fd, caminho_wav = tempfile.mkstemp(suffix=".wav", prefix="audio_")
    os.close(fd)
    audio.export(caminho_wav, format="wav")
    return caminho_wav

@handle_openai_error
def transcreve_audio(caminho_audio: str, prompt: str) -> tuple[str, str]:
    with open(caminho_audio, 'rb') as arquivo:
        try:
            resp = client.audio.transcriptions.create(
                model='whisper-1',
                language='pt',
                response_format='text',
                file=arquivo,
                prompt=prompt,
            )
            analise = processa_transcricao_chatgpt(resp)
            return resp, analise
        except Exception as e:
            st.warning(f"Erro na API OpenAI: {str(e)}. Usando serviço local.")
            return use_fallback_service(caminho_audio, prompt)

def salva_transcricao(texto: str, analise: str, origem: str = ""):
    agora = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    prefixo = f"{agora}_{origem}" if origem else agora
    with open(PASTA_TRANSCRICOES / f"{prefixo}_transcricao.txt", 'w', encoding='utf-8') as f:
        f.write(texto)
    with open(PASTA_TRANSCRICOES / f"{prefixo}_analise.txt", 'w', encoding='utf-8') as f:
        f.write(analise)
    return

def _salva_audio_do_video(file_bytes):
    with open(ARQUIVO_VIDEO_TEMP, 'wb') as f:
        f.write(file_bytes.read())
    clip = VideoFileClip(str(ARQUIVO_VIDEO_TEMP))
    clip.audio.write_audiofile(str(ARQUIVO_AUDIO_TEMP), logger=None)

def transcreve_tab_video():
    tipo_atendimento = st.radio('Tipo de Atendimento:', list(PROMPTS.keys()), horizontal=True, key='tipo_video')
    prompt = PROMPTS[tipo_atendimento]
    st.text_area("Prompt Selecionado:", prompt[:800] + '...', height=300)

    video = st.file_uploader('Adicione um vídeo', type=['mp4','mov','avi','mkv','webm'])
    if video:
        _salva_audio_do_video(video)
        wav = converter_para_wav(str(ARQUIVO_AUDIO_TEMP))
        texto, analise = transcreve_audio(wav, prompt)
        st.write("**Transcrição:**")
        st.write(texto)
        st.write("**Análise:**")
        st.write(analise)
        salva_transcricao(texto, analise, f'video_{video.name}')

def main():
    st.header('🎙️ Assistente de Organização 🎙️')
    st.markdown('Gravação, Transcrição e Organização.')
    st.markdown('Reuniões, Palestras, Atendimentos e Outros.')
    abas = st.tabs(['Microfone', 'Vídeo', 'Áudio', 'Texto'])
    with abas[0]:
        transcreve_tab_mic()
    with abas[1]:
        transcreve_tab_video()
    with abas[2]:
        pass
    with abas[3]:
        pass

if __name__ == '__main__':
    main()
