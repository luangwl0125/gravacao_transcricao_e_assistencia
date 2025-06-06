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

# Prompt para o ChatGPT
PROMPT_PSICOLOGICO = ''' 
Você é um Psicólogo/Neuropsicólogo Assistente com mais de 30 anos de experiência no Brasil, atuando diretamente como suporte técnico de LUAN GAMA WANDERLEY LEITE (CRP-15/3328), Psicólogo e Assessor Técnico da Defensoria Pública do Estado de Alagoas (Mat. 9864616-8).

Sua tarefa é analisar as informações registradas a seguir, delimitadas por #### DADOS ####, e produzir uma **estrutura de avaliação psicológica inicial**, de acordo com os parâmetros ético-profissionais, com base científica e 100% conforme as resoluções do CFP.

Siga estritamente as seguintes seções e orientações:

🔍 **1. IDENTIFICAÇÃO DA DEMANDA**  
- Especifique a origem da demanda: espontânea, judicial, institucional ou encaminhamento.  
- Avalie a urgência e a natureza principal da demanda (psicológica, psiquiátrica, social, jurídica).  

📋 **2. REGISTRO E DOCUMENTAÇÃO**  
- Realize anamnese com dados sociodemográficos, história clínica, familiar, social, educacional e laboral.  
- Registre de forma ética e sigilosa, conforme as Resoluções CFP nº 01/2009 e nº 06/2019.  
- Utilize instrumentos estruturados e entrevistas clínicas conforme a Resolução CFP nº 31/2022.  

🧠 **3. OBSERVAÇÃO PSICOLÓGICA E IMPRESSÕES CLÍNICAS INICIAIS**  
- Descreva o estado mental, o comportamento observado e o modo de comunicação do assistido.  
- Registre indicadores de sofrimento psíquico, risco psicossocial, sinais de violência ou ideação suicida.  

🧾 **4. PLANO INICIAL DE AÇÃO**  
- Indique se haverá continuidade do acompanhamento na DPE ou necessidade de encaminhamento (CAPS, CRAS, UBS etc.).  
- Avalie a necessidade de elaboração de documentos (relatório, laudo, parecer) ou aprofundamento diagnóstico.  

🧑‍⚖️ **5. CONTEXTO JURÍDICO E ARTICULAÇÃO INTERSETORIAL**  
- Verifique a existência de vínculo com processos jurídicos e, se aplicável, a articulação com defensores(as).  
- Projete a produção de documentos que possam subsidiar decisões judiciais.  

🤝 **6. CONSENTIMENTO E ORIENTAÇÕES ÉTICAS**  
- Confirme que foram explicadas as regras de sigilo, os limites da atuação na DPE e os usos das informações.  
- Verifique e registre a obtenção de Termo de Consentimento Informado, se aplicável.  

📌 **7. HIPÓTESES DIAGNÓSTICAS INICIAIS E ENCAMINHAMENTOS**  
- Apresente hipóteses diagnósticas preliminares com base nos dados e instrumentos disponíveis.  
- Estruture um Termo de Encaminhamento interno ou externo, se pertinente.

**Diretrizes complementares:**
- Utilize linguagem técnica e clara, evitando inferências não fundamentadas.  
- Mantenha fidelidade aos dados. Caso algo não tenha sido informado, sinalize como "não identificado".  
- Fundamente suas observações na literatura e nas normas do CFP.

O conteúdo a ser analisado está entre os delimitadores:

#### DADOS ####
{}
#### DADOS ####
'''

PROMPT_JURIDICO = ''' 
Você é um Defensor Público Supervisor e sua tarefa é analisar tecnicamente a transcrição de um atendimento jurídico prestado no Núcleo de Atendimento a Idosos da Defensoria Pública do Estado de Alagoas.

Sua análise servirá de base para orientação e formação dos estagiários envolvidos. Utilize linguagem técnica e formal, rigorosamente jurídica, e estruture a resposta conforme os tópicos indicados abaixo.

Diretrizes:

1. Evite inferências não sustentadas nos fatos relatados. Se houver lacunas, registre como "não informado" ou "não identificado".
2. Respeite o sigilo e a ética profissional; não inclua juízos de valor ou suposições pessoais.
3. Utilize o conteúdo delimitado por #### TRANSCRIÇÃO #### como única fonte de análise.

**Seções Obrigatórias (títulos em maiúsculo):**

- DADOS DO ATENDIMENTO  
- QUALIFICAÇÃO E CONTEXTO DO ASSISTIDO  
- PROBLEMA JURÍDICO APRESENTADO  
- ELEMENTOS DE FATO RELEVANTES  
- ELEMENTOS DE DIREITO IDENTIFICADOS  
- AÇÕES REALIZADAS NO ATENDIMENTO  
- ANÁLISE CRÍTICA DO PROCEDIMENTO  
- ORIENTAÇÕES PARA O ESTAGIÁRIO  

O conteúdo da transcrição a ser analisado está delimitado entre #### TRANSCRIÇÃO ####.

#### TRANSCRIÇÃO ####
{}
#### TRANSCRIÇÃO ####
'''

PROMPT_SERVICO_SOCIAL = '''
Você é um Assistente de Serviço Social com sólida experiência em atuação na Defensoria Pública, especializado em demandas sociais de pessoas idosas. Sua tarefa é analisar as informações a seguir, delimitadas por #### DADOS ####, e produzir uma **estrutura de avaliação social inicial**, de acordo com as normas ético‐profissionais do Serviço Social (CFESS/CFP) e com base em evidências. Siga rigorosamente as seções abaixo:

1. CONTEXTO E HISTÓRICO:
   - Identifique origem da demanda: espontânea, judicial, institucional ou encaminhamento.
   - Registre informações sociodemográficas, composição familiar, rede de apoio, condições de moradia e renda.
   - Contextualize aspectos culturais e ambientais relevantes à situação do assistido.

2. DIAGNÓSTICO SOCIAL:
   - Avalie fatores de risco social: vulnerabilidade, violência, abandono, carência de recursos.
   - Verifique acesso a políticas públicas (CRAS, CREAS, Benefício de Prestação Continuada, Habitação Popular etc.).
   - Identifique indicadores de fragilidade: saúde precária, isolamento, dependência financeira.

3. RECURSOS E REDES DE APOIO:
   - Liste serviços e programas sociais disponíveis e possíveis encaminhamentos.
   - Avalie a presença de cuidadores formais/informais e a qualidade do suporte familiar.
   - Análise da viabilidade de programas de proteção ao idoso ou rede de assistência.

4. PLANO DE INTERVENÇÃO INICIAL:
   - Defina ações imediatas e de médio prazo: solicitação de benefícios, inclusão em programas de assistência, articulação com órgãos municipais/estaduais.
   - Sugira inclusão em rede de serviços (Saúde, Assistência Social, Educação, Direitos Humanos).
   - Preveja acompanhamento continuado e frequência de visitas domiciliares (se necessário).

5. ARTICULAÇÃO INTERSETORIAL:
   - Verifique vínculo com processos judiciais e atuação conjunta com Defensoria Pública.
   - Projete relatórios técnicos ou pareceres para subsidiar decisões judiciais e sociais.
   - Indique possíveis parcerias com organizações não governamentais e conselhos de direitos do idoso.

6. ÉTICA E DIRETRIZES PROFISSIONAIS:
   - Confirme cumprimento de normativas do CFESS e princípios do Serviço Social: sigilo, autonomia, respeito à dignidade.
   - Registre obtenção de Termo de Consentimento Informado, se aplicável.
   - Ressalte a importância da escuta qualificada e do protagonismo do assistido.

7. ENCAMINHAMENTOS E RECOMENDAÇÕES:
   - Apresente encaminhamentos imediatos (CRAS, CREAS, CAPS, UBS, CAPS-Idoso, CUCA etc.).
   - Sugira estratégias de fortalecimento de rede social: grupos de convivência, Centro Dia do Idoso.
   - Estruture modelo de Relatório Social ou Parecer Social para a Defensoria Pública.

#### DADOS ####
{}
#### DADOS ####
'''

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

if 'transcricao_mic' not in st.session_state:
    st.session_state['transcricao_mic'] = ''
if 'analise_mic' not in st.session_state:
    st.session_state['analise_mic'] = ''
if 'gravando_audio' not in st.session_state:
    st.session_state['gravando_audio'] = False
if 'audio_completo' not in st.session_state:
    st.session_state['audio_completo'] = pydub.AudioSegment.empty()

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

def transcreve_tab_mic():
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
        rtc_configuration={"iceServers": get_ice_servers()}
    )

    if not ctx.state.playing:
        if st.session_state['gravando_audio']:
            st.session_state['gravando_audio'] = False
            if len(st.session_state['audio_completo']) > 0:
                st.session_state['audio_completo'].export(
                    PASTA_TRANSCRICOES / f"audio_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.wav",
                    format='wav'
                )
                st.session_state['audio_completo'] = pydub.AudioSegment.empty()

        if st.session_state['transcricao_mic']:
            if not st.session_state['analise_mic']:
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

        chunk_atual = pydub.AudioSegment.empty()
        chunk_atual = adiciona_chunck_de_audio(frames, chunk_atual)
        chunk_audio += chunk_atual

        if st.session_state['gravando_audio']:
            st.session_state['audio_completo'] += chunk_atual

        agora = time.time()
        if len(chunk_audio) > 0 and agora - ultimo > 10:
            ultimo = agora
            chunk_audio.export(ARQUIVO_MIC_TEMP, format='wav')
            texto, _ = transcreve_audio(str(ARQUIVO_MIC_TEMP), prompt_mic)
            st.session_state['transcricao_mic'] += texto
            placeholder.write(st.session_state['transcricao_mic'])
            chunk_audio = pydub.AudioSegment.empty()

def main():
    st.header('🎙️ Assistente de Organização 🎙️')
    st.markdown('Gravação, Transcrição e Organização.')
    st.markdown('Reuniões, Palestras, Atendimentos e Outros.')
    abas = st.tabs(['Microfone', 'Vídeo', 'Áudio', 'Texto'])
    with abas[0]:
        transcreve_tab_mic()

if __name__ == '__main__':
    main()
