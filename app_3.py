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
            except RateLimitError as e:
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

def transcreve_tab_mic():
    for key, default in {
        "transcricao_mic": "",
        "analise_mic": "",
        "gravando_audio": False,
        "audio_completo": pydub.AudioSegment.empty()
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    tipo_atendimento = st.radio('Tipo de Atendimento:', list(PROMPTS.keys()), horizontal=True)
    prompt_mic = PROMPTS[tipo_atendimento]
    st.text_area("Prompt Selecionado:", prompt_mic[:800] + '...', height=300)

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button('🔴 Gravar Áudio' if not st.session_state['gravando_audio'] else '⏹️ Parar Gravação'):
            st.session_state['gravando_audio'] = not st.session_state['gravando_audio']
            if not st.session_state['gravando_audio'] and len(st.session_state['audio_completo']) > 0:
                st.session_state['audio_completo'].export(
                    PASTA_TRANSCRICOES / f"audio_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.wav",
                    format='wav'
                )
                st.session_state['audio_completo'] = pydub.AudioSegment.empty()

    ctx = webrtc_streamer(
        key='mic', mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        media_stream_constraints={'video': False, 'audio': True},
        rtc_configuration={"iceServers": [{'urls': ['stun:stun.l.google.com:19302']}]} )

@handle_openai_error
def processa_transcricao_chatgpt(texto: str) -> str:
    resposta = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": PROMPT_PSICOLOGICO.format(texto)}
        ]
    )
    return resposta.choices[0].message.content

def converter_para_wav(caminho_entrada: str) -> str:
    audio = pydub.AudioSegment.from_file(caminho_entrada)
    fd, caminho_wav = tempfile.mkstemp(suffix=".wav", prefix="audio_")
    os.close(fd)
    audio.export(caminho_wav, format="wav")
    return caminho_wav

@st.cache_data
def get_ice_servers():
    return [{'urls': ['stun:stun.l.google.com:19302']}]

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
    with open(PASTA_TRANSCRICOES / f"{prefixo}_transcricao.txt", 'w', encoding='utf-8') as f:
        f.write(texto)
    with open(PASTA_TRANSCRICOES / f"{prefixo}_analise.txt", 'w', encoding='utf-8') as f:
        f.write(analise)
    return

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

def transcreve_tab_mic():
    tipo = st.radio("Tipo de Atendimento:", list(PROMPTS.keys()), horizontal=True)
    prompt = PROMPTS[tipo]
    st.session_state['prompt_mic'] = prompt

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button('🔴 Gravar Áudio' if not st.session_state['gravando_audio'] else '⏹️ Parar Gravação'):
            st.session_state['gravando_audio'] = not st.session_state['gravando_audio']
            if not st.session_state['gravando_audio'] and len(st.session_state['audio_completo']) > 0:
                st.session_state['audio_completo'].export(
                    PASTA_TRANSCRICOES / f"audio_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.wav",
                    format='wav')
                st.session_state['audio_completo'] = pydub.AudioSegment.empty()

    ctx = webrtc_streamer(key='mic', mode=WebRtcMode.SENDONLY,
                          audio_receiver_size=1024,
                          media_stream_constraints={'video': False, 'audio': True},
                          rtc_configuration={"iceServers": get_ice_servers()})

    # Captura e processa áudio como antes [...]
    # (mesma lógica da versão anterior)

def transcreve_tab_video():
    tipo = st.radio("Tipo de Atendimento:", list(PROMPTS.keys()), horizontal=True, key="video_radio")
    prompt = PROMPTS[tipo]
    video = st.file_uploader("Adicione um vídeo", type=['mp4','mov','avi','mkv','webm'])
    if video:
        with open(ARQUIVO_VIDEO_TEMP, 'wb') as f:
            f.write(video.read())
        clip = VideoFileClip(str(ARQUIVO_VIDEO_TEMP))
        clip.audio.write_audiofile(str(ARQUIVO_AUDIO_TEMP), logger=None)
        wav = converter_para_wav(str(ARQUIVO_AUDIO_TEMP))
        texto, analise = transcreve_audio(wav, prompt)
        st.write("**Transcrição:**")
        st.write(texto)
        st.write("**Análise:**")
        st.write(analise)
        salva_transcricao(texto, analise, f'video_{video.name}')

def transcreve_tab_audio():
    tipo = st.radio("Tipo de Atendimento:", list(PROMPTS.keys()), horizontal=True, key="audio_radio")
    prompt = PROMPTS[tipo]
    audio = st.file_uploader("Adicione um áudio", type=['opus','mp4','mpeg','wav','mp3','m4a'])
    if audio:
        caminho = PASTA_TEMP / audio.name
        with open(caminho, 'wb') as f:
            f.write(audio.read())
        wav = converter_para_wav(str(caminho))
        texto, analise = transcreve_audio(wav, prompt)
        st.write("**Transcrição:**")
        st.write(texto)
        st.write("**Análise:**")
        st.write(analise)
        salva_transcricao(texto, analise, f'audio_{audio.name}')

def transcreve_tab_texto():
    tipo = st.radio("Tipo de Atendimento:", list(PROMPTS.keys()), horizontal=True, key="texto_radio")
    prompt = PROMPTS[tipo]
    arquivo_texto = st.file_uploader("Adicione um arquivo de texto", type=['txt', 'doc', 'docx'])
    if arquivo_texto:
        try:
            if arquivo_texto.type == 'text/plain':
                texto = arquivo_texto.getvalue().decode('utf-8')
            else:
                import docx2txt
                texto = docx2txt.process(arquivo_texto)
            analise = processa_transcricao_chatgpt(texto)
            st.write("**Texto Original:**")
            st.write(texto)
            st.write("**Análise:**")
            st.write(analise)
            salva_transcricao(texto, analise, f'texto_{arquivo_texto.name}')
        except Exception as e:
            st.error(f"Erro ao processar o arquivo: {str(e)}")

def main():
    st.header('🎙️ Assistente de Organização 🎙️')
    st.markdown('Gravação, Transcrição e Organização.')
    st.markdown('Reuniões, Palestras, Atendimentos e Outros.')
    abas = st.tabs(['Microfone', 'Vídeo', 'Áudio', 'Texto'])
    with abas[0]: transcreve_tab_mic()
    with abas[1]: transcreve_tab_video()
    with abas[2]: transcreve_tab_audio()
    with abas[3]: transcreve_tab_texto()

if __name__ == '__main__':
    main()
