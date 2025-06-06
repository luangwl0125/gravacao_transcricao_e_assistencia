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
                if attempt == MAX_RETRIES - 1:  # Se for a última tentativa
                    st.warning("OpenAI API rate limit atingido. Usando serviço local de fallback.")
                    return use_fallback_service(*args, **kwargs)
                else:
                    time.sleep(RETRY_DELAY * (attempt + 1))  # Espera exponencial
            except Exception as e:
                st.error(f"Erro ao processar: {str(e)}")
                return None
    return wrapper

def use_fallback_service(caminho_audio=None, prompt=None, texto=None):
    """Serviço de fallback usando Whisper local"""
    try:
        if caminho_audio:  # Para transcrição
            model = get_local_whisper()
            result = model.transcribe(caminho_audio, language="pt")
            return result["text"], "Análise não disponível (usando serviço local)"
        elif texto:  # Para análise
            return texto, "Análise não disponível (usando serviço local)"
    except Exception as e:
        st.error(f"Erro no serviço de fallback: {str(e)}")
        return "", ""

PROMPT_JURIDICO = ''' 
Você é um Defensor Público Supervisor, especialista em Direito Público, com ênfase em Direitos Humanos, Direito da Pessoa Idosa e Direito Previdenciário. Sua função é analisar tecnicamente a transcrição de um atendimento jurídico prestado no Núcleo de Atendimento a Idosos da Defensoria Pública do Estado de Alagoas.

Sua análise servirá de base para:
- Orientar estagiários de Direito;
- Elaborar documentos jurídicos e administrativos;
- Produzir minutas de petições iniciais, manifestações, ofícios, notificações, requerimentos e recursos, conforme aplicável ao caso concreto.

Utilize linguagem jurídica formal, clara e objetiva, com base nas normas vigentes e boa técnica argumentativa.

**Fundamentação jurídica obrigatória (a aplicar conforme o caso):**

📘 **Constituição Federal de 1988**  
- Art. 1º, III; Art. 3º, IV; Art. 5º; Art. 6º; Art. 230

📕 **Estatuto da Pessoa Idosa (Lei nº 10.741/2003)**  
- Direitos fundamentais (arts. 2º a 21)  
- Previdência, Assistência, Saúde, Trabalho e Justiça (arts. 22 a 46)  
- Penalizações (arts. 49 a 108)

📗 **Lei Orgânica da Assistência Social (Lei nº 8.742/1993)**  
📘 **Lei nº 8.213/1991 – Benefícios Previdenciários**  
📘 **Lei nº 13.146/2015 – Estatuto da Pessoa com Deficiência (quando aplicável)**  
📘 **Código Civil** (Alimentos, Interdição, Curatela)  
📘 **Código de Processo Civil** (Tutela Provisória, Interdição, Alimentos)

**Tipos de documentos/petições que você pode elaborar a partir da análise:**

- Petição Inicial (Alimentos, Curatela, Interdição, Benefício Assistencial, Tutela Antecipada)  
- Requerimento administrativo à rede pública (CRAS, CAPS, UBS, INSS etc.)  
- Notificação Extrajudicial  
- Ofício Institucional para encaminhamentos ou articulações intersetoriais  
- Declaração ou termo de comparecimento  
- Requisição de documentos ou exames  
- Minuta de manifestação, réplica ou apelação conforme o andamento processual

**Diretrizes obrigatórias:**

1. Evite inferências sem base na transcrição. Use “não informado” quando necessário.  
2. Mantenha fidelidade aos dados, sigilo e ética profissional.  
3. Fundamente toda recomendação com base legal adequada.  
4. Estruture a resposta conforme os tópicos abaixo. Se alguma seção não for aplicável, indique como "não se aplica".

**Seções Obrigatórias (em letras maiúsculas):**

- DADOS DO ATENDIMENTO  
- QUALIFICAÇÃO E CONTEXTO DO ASSISTIDO  
- PROBLEMA JURÍDICO APRESENTADO  
- ELEMENTOS DE FATO RELEVANTES  
- ELEMENTOS DE DIREITO IDENTIFICADOS (com leis e artigos)  
- AÇÕES REALIZADAS NO ATENDIMENTO  
- ANÁLISE CRÍTICA DO PROCEDIMENTO  
- ORIENTAÇÕES PARA O ESTAGIÁRIO  
- RECOMENDAÇÕES JURÍDICAS  
- MINUTA DE DOCUMENTO OU PETIÇÃO (se aplicável)

O conteúdo da transcrição a ser analisado está delimitado entre:

#### TRANSCRIÇÃO ####
{}
#### TRANSCRIÇÃO ####
'''

@handle_openai_error
def processa_transcricao_chatgpt(texto: str) -> str:
    """Processa o texto usando o ChatGPT para gerar uma análise estruturada"""
    resposta = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": PROMPT_ANALISE.format(texto)}
        ]
    )
    return resposta.choices[0].message.content

# Converte qualquer formato suportado para WAV
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
    """Salva a transcrição original e a análise em arquivos separados"""
    agora = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    prefixo = f"{agora}_{origem}" if origem else agora
    
    # Salva transcrição original
    arquivo_transcricao = PASTA_TRANSCRICOES / f"{prefixo}_transcricao.txt"
    with open(arquivo_transcricao, 'w', encoding='utf-8') as f:
        f.write(texto)
    
    # Salva análise estruturada
    arquivo_analise = PASTA_TRANSCRICOES / f"{prefixo}_analise.txt"
    with open(arquivo_analise, 'w', encoding='utf-8') as f:
        f.write(analise)
    
    return arquivo_transcricao, arquivo_analise

# Aba Microfone
def transcreve_tab_mic():
    prompt_mic = st.text_input('Tipo de Atendimento (opcional)', key='input_mic')
    
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
        rtc_configuration={"iceServers": get_ice_servers()}
    )

    # Quando parar a gravação
    if not ctx.state.playing:
        # Reseta estado de gravação de áudio
        if st.session_state['gravando_audio']:
            st.session_state['gravando_audio'] = False
            if len(st.session_state['audio_completo']) > 0:
                st.session_state['audio_completo'].export(
                    PASTA_TRANSCRICOES / f"audio_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.wav",
                    format='wav'
                )
                st.session_state['audio_completo'] = pydub.AudioSegment.empty()
        
        if st.session_state['transcricao_mic']:  # Se há transcrição
            if not st.session_state['analise_mic']:  # Se ainda não gerou análise
                st.write("Gerando análise...")
                st.session_state['analise_mic'] = processa_transcricao_chatgpt(st.session_state['transcricao_mic'])
            
            st.write("**Transcrição:**")
            st.write(st.session_state['transcricao_mic'])
            st.write("**Análise:**")
            st.write(st.session_state['analise_mic'])
            
            salva_transcricao(
                st.session_state['transcricao_mic'],
                st.session_state['analise_mic'],
                'microfone'
            )
        return

    status_col1, status_col2 = st.columns(2)
    with status_col1:
        st.markdown('**🎙️ Transcrevendo...**')
    with status_col2:
        if st.session_state['gravando_audio']:
            st.markdown('**🔴 Gravando áudio...**')

    placeholder = st.empty()
    chunk_audio = pydub.AudioSegment.empty()
    ultimo = time.time()
    st.session_state['transcricao_mic'] = ''
    st.session_state['analise_mic'] = ''

    while ctx.audio_receiver:
        try:
            frames = ctx.audio_receiver.get_frames(timeout=1)
        except queue.Empty:
            time.sleep(0.1)
            continue
        
        # Processa os frames de áudio
        chunk_atual = pydub.AudioSegment.empty()
        chunk_atual = adiciona_chunck_de_audio(frames, chunk_atual)
        chunk_audio += chunk_atual
        
        # Se estiver gravando, adiciona ao áudio completo
        if st.session_state['gravando_audio']:
            st.session_state['audio_completo'] += chunk_atual
        
        agora = time.time()
        # A cada 10s, transcreve
        if len(chunk_audio) > 0 and agora - ultimo > 10:
            ultimo = agora
            # exporta temporariamente para transcrição
            chunk_audio.export(ARQUIVO_MIC_TEMP, format='wav')
            texto, _ = transcreve_audio(str(ARQUIVO_MIC_TEMP), prompt_mic)
            st.session_state['transcricao_mic'] += texto
            placeholder.write(st.session_state['transcricao_mic'])
            chunk_audio = pydub.AudioSegment.empty()

# Extrai áudio de vídeo
def _salva_audio_do_video(file_bytes):
    with open(ARQUIVO_VIDEO_TEMP, 'wb') as f:
        f.write(file_bytes.read())
    clip = VideoFileClip(str(ARQUIVO_VIDEO_TEMP))
    clip.audio.write_audiofile(str(ARQUIVO_AUDIO_TEMP), logger=None)

# Aba Vídeo
def transcreve_tab_video():
    prompt = st.text_input('Prompt (opcional)', key='input_video')
    video = st.file_uploader('Adicione um vídeo', type=['mp4','mov','avi','mkv','webm'])
    if video:
        # salva vídeo e extrai áudio WAV
        _salva_audio_do_video(video)
        wav = converter_para_wav(str(ARQUIVO_AUDIO_TEMP))
        texto, analise = transcreve_audio(wav, prompt)
        st.write("**Transcrição:**")
        st.write(texto)
        st.write("**Análise:**")
        st.write(analise)
        # Salva os arquivos
        salva_transcricao(texto, analise, f'video_{video.name}')

# Aba Áudio
def transcreve_tab_audio():
    prompt = st.text_input('Prompt (opcional)', key='input_audio')
    audio = st.file_uploader('Adicione um áudio', type=['opus','mp4','mpeg','wav','mp3','m4a'])
    if audio:
        # salva e converte para WAV
        caminho = PASTA_TEMP / audio.name
        with open(caminho, 'wb') as f:
            f.write(audio.read())
        wav = converter_para_wav(str(caminho))
        texto, analise = transcreve_audio(wav, prompt)
        st.write("**Transcrição:**")
        st.write(texto)
        st.write("**Análise:**")
        st.write(analise)
        # Salva os arquivos
        salva_transcricao(texto, analise, f'audio_{audio.name}')

# Aba Texto
def transcreve_tab_texto():
    st.write("Envie um arquivo de texto com a transcrição para análise")
    arquivo_texto = st.file_uploader('Adicione um arquivo de texto', type=['txt', 'doc', 'docx'])
    if arquivo_texto:
        try:
            if arquivo_texto.type == 'text/plain':
                # Para arquivos .txt
                texto = arquivo_texto.getvalue().decode('utf-8')
            else:
                # Para arquivos Word (.doc, .docx)
                import docx2txt
                texto = docx2txt.process(arquivo_texto)
            
            analise = processa_transcricao_chatgpt(texto)
            st.write("**Texto Original:**")
            st.write(texto)
            st.write("**Análise:**")
            st.write(analise)
            # Salva os arquivos
            salva_transcricao(texto, analise, f'texto_{arquivo_texto.name}')
        except Exception as e:
            st.error(f"Erro ao processar o arquivo: {str(e)}")

# Função principal
def main():
    st.header('Núcleo de Atendimento ao Idoso - DPE/AL')
    st.markdown('Transcrição e/ou Gravação e Organização de Atendimentos.')
    st.markdown('Orientações e Recomendações Jurídicas')
    abas = st.tabs(['Microfone', 'Vídeo', 'Áudio', 'Texto'])
    with abas[0]:
        transcreve_tab_mic()
    with abas[1]:
        transcreve_tab_video()
    with abas[2]:
        transcreve_tab_audio()
    with abas[3]:
        transcreve_tab_texto()

if __name__ == '__main__':
    main()

