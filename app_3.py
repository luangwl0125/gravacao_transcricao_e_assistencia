import sys
import audioop
sys.modules['pyaudioop'] = audioop

from pathlib import Path
from datetime import datetime
import queue
import os

import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

import openai
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

PASTA_TEMP = Path(__file__).parent / 'temp'
PASTA_TEMP.mkdir(exist_ok=True)
PASTA_TRANSCRICOES = Path(__file__).parent / 'TRANSCRICOES'
PASTA_TRANSCRICOES.mkdir(exist_ok=True)

ARQUIVO_MIC_TEMP = PASTA_TEMP / 'mic.wav'

@st.cache_resource
def get_local_whisper():
    import whisper
    return whisper.load_model("base")

# Prompts
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

def st_webrtc_audio_recorder():
    webrtc_ctx = webrtc_streamer(
        key="mic",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
    )

    audio_bytes = None
    if webrtc_ctx.audio_receiver:
        try:
            frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            audio_bytes = b"".join([f.to_ndarray().tobytes() for f in frames])
        except queue.Empty:
            pass

    return audio_bytes

def transcreve_upload(tipo, extensoes, formato_visualizacao):
    arquivo = st.file_uploader(f"Envie um arquivo de {tipo}", type=extensoes)
    if arquivo:
        caminho_temp = PASTA_TEMP / arquivo.name
        with open(caminho_temp, "wb") as f:
            f.write(arquivo.getbuffer())

        if tipo in ["áudio", "vídeo"]:
            st.audio(str(caminho_temp), format=formato_visualizacao) if tipo == "áudio" else st.video(str(caminho_temp))

        try:
            resultado = get_local_whisper().transcribe(str(caminho_temp))
            texto = resultado["text"]
            st.subheader("Transcrição")
            st.text_area("Texto transcrito:", texto, height=300)
            salvar_transcricao(texto)
        except Exception as e:
            st.error(f"Erro ao transcrever {tipo}: {e}")

def transcreve_texto():
    arquivo = st.file_uploader("Envie um arquivo de texto (.txt, .docx)", type=["txt", "docx"])
    if arquivo:
        try:
            if arquivo.type == "text/plain":
                conteudo = arquivo.read().decode("utf-8")
            else:
                from docx import Document
                doc = Document(arquivo)
                conteudo = "\n".join([p.text for p in doc.paragraphs])
            st.subheader("Conteúdo do Texto")
            st.text_area("Texto extraído:", conteudo, height=300)
            salvar_transcricao(conteudo)
        except Exception as e:
            st.error(f"Erro ao ler texto: {e}")

def transcreve_mic():
    st.markdown("Pressione o botão abaixo para iniciar a gravação de voz.")
    audio_buffer = st_webrtc_audio_recorder()
    if audio_buffer:
        st.audio(audio_buffer, format="audio/wav")
        with open(ARQUIVO_MIC_TEMP, "wb") as f:
            f.write(audio_buffer)

        resultado = get_local_whisper().transcribe(str(ARQUIVO_MIC_TEMP))
        texto = resultado["text"]
        st.subheader("Transcrição")
        st.text_area("Texto transcrito:", texto, height=300)
        salvar_transcricao(texto)

def salvar_transcricao(texto):
    if st.button("Salvar transcrição"):
        agora = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        tipo = st.session_state.get('tipo_prompt', 'generico').lower().replace(' ', '_')
        caminho = PASTA_TRANSCRICOES / f"transcricao_{tipo}_{agora}.txt"
        caminho.write_text(texto, encoding='utf-8')
        st.success(f"Transcrição salva em: {caminho}")

def main():
    st.sidebar.title("Selecione o tipo de atendimento")
    tipo_prompt = st.sidebar.radio("Tipo de Análise:", list(PROMPTS.keys()))
    st.session_state['tipo_prompt'] = tipo_prompt
    st.session_state['prompt_escolhido'] = PROMPTS[tipo_prompt]

    st.header('Assistente de Organização')
    st.markdown('Gravação, Transcrição e Organização.')

    abas = st.tabs(['Microfone', 'Vídeo', 'Áudio', 'Texto'])
    with abas[0]:
        transcreve_mic()
    with abas[1]:
        transcreve_upload("vídeo", ["mp4", "mov"], "video/mp4")
    with abas[2]:
        transcreve_upload("áudio", ["wav", "mp3"], "audio/wav")
    with abas[3]:
        transcreve_texto()

if __name__ == '__main__':
    main()
