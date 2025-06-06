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

# --- PROMPTS ---
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


# --- Seleção de Prompt ---
st.sidebar.title("Tipo de Atendimento")
tipo_atendimento = st.sidebar.radio("Selecione o tipo:", ["Psicológico", "Jurídico"])
prompt_analise = PROMPT_PSICOLOGICO if tipo_atendimento == "Psicológico" else PROMPT_JURIDICO

# --- Diretórios ---
PASTA_TEMP = Path("temp")
PASTA_TRANSCRICOES = Path("TRANSCRICOES")
PASTA_TEMP.mkdir(exist_ok=True)
PASTA_TRANSCRICOES.mkdir(exist_ok=True)
ARQUIVO_AUDIO_TEMP = PASTA_TEMP / "audio.wav"
ARQUIVO_VIDEO_TEMP = PASTA_TEMP / "video.mp4"
ARQUIVO_MIC_TEMP = PASTA_TEMP / "mic.wav"

client = openai.OpenAI()

# --- Processamento ---
def processa_transcricao_chatgpt(texto: str) -> str:
    resposta = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt_analise.format(texto)}]
    )
    return resposta.choices[0].message.content

def transcreve_audio(caminho_audio: str, prompt: str) -> tuple[str, str]:
    with open(caminho_audio, 'rb') as arquivo:
        resp = client.audio.transcriptions.create(
            model='whisper-1',
            language='pt',
            response_format='text',
            file=arquivo,
            prompt=prompt,
        )
        analise = processa_transcricao_chatgpt(resp)
        return resp, analise

def converter_para_wav(caminho_entrada: str) -> str:
    audio = pydub.AudioSegment.from_file(caminho_entrada)
    fd, caminho_wav = tempfile.mkstemp(suffix=".wav", prefix="audio_")
    os.close(fd)
    audio.export(caminho_wav, format="wav")
    return caminho_wav

def salva_transcricao(texto: str, analise: str, origem: str = ""):
    agora = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    prefixo = f"{agora}_{origem}" if origem else agora
    with open(PASTA_TRANSCRICOES / f"{prefixo}_transcricao.txt", 'w', encoding='utf-8') as f:
        f.write(texto)
    with open(PASTA_TRANSCRICOES / f"{prefixo}_analise.txt", 'w', encoding='utf-8') as f:
        f.write(analise)

# --- Abas ---
def transcreve_tab_audio():
    st.subheader("Análise de Áudio")
    audio = st.file_uploader("Envie um arquivo de áudio", type=["mp3", "wav", "m4a", "ogg"])
    if audio:
        caminho = PASTA_TEMP / audio.name
        with open(caminho, 'wb') as f:
            f.write(audio.read())
        wav = converter_para_wav(str(caminho))
        texto, analise = transcreve_audio(wav, prompt_analise)
        st.write("**Transcrição:**")
        st.write(texto)
        st.write("**Análise:**")
        st.write(analise)
        salva_transcricao(texto, analise, f"audio_{audio.name}")

def transcreve_tab_video():
    st.subheader("Análise de Vídeo")
    video = st.file_uploader("Envie um vídeo", type=["mp4", "mov", "avi", "mkv"])
    if video:
        with open(ARQUIVO_VIDEO_TEMP, 'wb') as f:
            f.write(video.read())
        clip = VideoFileClip(str(ARQUIVO_VIDEO_TEMP))
        clip.audio.write_audiofile(str(ARQUIVO_AUDIO_TEMP), logger=None)
        wav = converter_para_wav(str(ARQUIVO_AUDIO_TEMP))
        texto, analise = transcreve_audio(wav, prompt_analise)
        st.write("**Transcrição:**")
        st.write(texto)
        st.write("**Análise:**")
        st.write(analise)
        salva_transcricao(texto, analise, f"video_{video.name}")

def transcreve_tab_texto():
    st.subheader("Análise de Texto")
    texto_arquivo = st.file_uploader("Envie um arquivo de texto (.txt, .docx)", type=["txt", "doc", "docx"])
    if texto_arquivo:
        try:
            if texto_arquivo.type == "text/plain":
                texto = texto_arquivo.getvalue().decode('utf-8')
            else:
                import docx2txt
                texto = docx2txt.process(texto_arquivo)
            analise = processa_transcricao_chatgpt(texto)
            st.write("**Texto Original:**")
            st.write(texto)
            st.write("**Análise:**")
            st.write(analise)
            salva_transcricao(texto, analise, f"texto_{texto_arquivo.name}")
        except Exception as e:
            st.error(f"Erro ao processar: {str(e)}")

def main():
    st.title("🎙️ Assistente de Organização")
    st.markdown("Gravação, Transcrição e Organização.")
    abas = st.tabs(["Áudio", "Vídeo", "Texto"])
    with abas[0]:
        transcreve_tab_audio()
    with abas[1]:
        transcreve_tab_video()
    with abas[2]:
        transcreve_tab_texto()

if __name__ == '__main__':
    main()
