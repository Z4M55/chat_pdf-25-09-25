# -*- coding: utf-8 -*-
import os
import platform
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

# =========================
# Configuración de la página
# =========================
st.set_page_config(
    page_title="🤖 RAG | Tech Analyzer",
    page_icon="💾",
    layout="wide"
)

# =========================
# Estilos Tech (oscuro + neón)
# =========================
st.markdown("""
<style>
  :root {
    --bg:#0b1220;
    --panel:#0f182b;
    --text:#e6f7ff;
    --muted:#9fb3c8;
    --accent:#00e5ff;
    --accent2:#00ffa3;
  }
  html, body, .stApp {
    background: radial-gradient(1000px 600px at 10% 0%, #0f1a30 0%, var(--bg) 60%);
    color: var(--text) !important;
  }
  [data-testid="stSidebar"], section[data-testid="stSidebar"] > div {
    background: linear-gradient(180deg, #0e1628 0%, #0b1220 100%) !important;
    color: var(--text) !important;
    border-right: 1px solid rgba(0,229,255,.15);
  }
  h1, h2, h3, h4, h5, h6 {
    color: var(--accent);
    font-family: "JetBrains Mono", monospace;
  }
  p, label, span, div, .stMarkdown {
    color: var(--text) !important;
    font-family: "Inter", sans-serif;
  }
  .stButton>button {
    background: linear-gradient(90deg, var(--accent) 0%, var(--accent2) 100%) !important;
    color: #00121a !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    transition: transform .08s ease-in-out, box-shadow .2s ease-in-out;
    box-shadow: 0 0 12px rgba(0,229,255,.5);
  }
  .stButton>button:hover {
    transform: translateY(-1px);
    box-shadow: 0 0 18px rgba(0,229,255,.75);
  }
  textarea, input {
    background: #0f182b !important;
    color: var(--text) !important;
    border: 1px solid rgba(0,229,255,.3) !important;
    border-radius: 10px !important;
  }
  .stExpander {
    background: var(--panel) !important;
    border: 1px solid rgba(0,229,255,.2);
    border-radius: 10px;
  }
</style>
""", unsafe_allow_html=True)

# =========================
# Título principal
# =========================
st.title("💾 Generación Aumentada por Recuperación (RAG) | Tech Mode")
st.markdown("""
**RAG (Retrieval-Augmented Generation)** combina el poder de la búsqueda semántica con modelos de lenguaje avanzados.  
Sube un **PDF** 📄 y haz preguntas; el sistema leerá, comprenderá y responderá con precisión usando **IA de OpenAI**.  
""")
st.caption(f"🧠 Python versión: `{platform.python_version()}`")

# =========================
# Cargar imagen (opcional)
# =========================
try:
    image = Image.open("Chat_pdf.png")
    st.image(image, width=320, caption="⚙️ IA + Document Intelligence")
except Exception as e:
    st.warning(f"⚠️ Imagen no cargada: {e}")

# =========================
# Barra lateral (info)
# =========================
with st.sidebar:
    st.subheader("🧩 Agente de Análisis Documental")
    st.markdown("""
    Este asistente utiliza **LangChain + OpenAI + FAISS** para:  
    - 🔍 Extraer conocimiento de documentos PDF  
    - 🧠 Generar respuestas basadas en contexto  
    - ⚙️ Crear un flujo RAG (Recuperación + Generación)  
    """)

# =========================
# Ingreso de clave API
# =========================
api_key = st.text_input("🔑 Ingresa tu Clave de OpenAI:", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
else:
    st.warning("⚠️ Ingresa tu clave de API de OpenAI para continuar.")

# =========================
# Carga de PDF
# =========================
pdf = st.file_uploader("📂 Carga un archivo PDF", type="pdf")

# =========================
# Procesamiento del PDF
# =========================
if pdf is not None and api_key:
    try:
        with st.spinner("🧠 Extrayendo texto del PDF..."):
            reader = PdfReader(pdf)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
        
        st.info(f"📄 Texto extraído: `{len(text)} caracteres`")

        # Dividir texto
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.success(f"✅ Documento dividido en `{len(chunks)}` fragmentos")

        # Crear embeddings y base de conocimiento
        with st.spinner("⚙️ Generando embeddings y base semántica..."):
            embeddings = OpenAIEmbeddings()
            knowledge_base = FAISS.from_texts(chunks, embeddings)

        # Interfaz de pregunta
        st.subheader("💬 Haz una pregunta sobre el documento")
        user_question = st.text_area("Escribe tu pregunta aquí...", placeholder="Ejemplo: ¿Cuál es el objetivo principal del texto?")

        # =========================
        # Procesar la pregunta
        # =========================
        if user_question:
            with st.spinner("🤖 Procesando tu consulta con IA..."):
                docs = knowledge_base.similarity_search(user_question)

                llm = OpenAI(temperature=0, model_name="gpt-4o")
                chain = load_qa_chain(llm, chain_type="stuff")

                response = chain.run(input_documents=docs, question=user_question)

            st.markdown("### 🧾 Respuesta:")
            st.success(response)

            # Información adicional
            st.caption("⚙️ Motor: `GPT-4o` | Embeddings: `OpenAIEmbeddings` | VectorStore: `FAISS`")

    except Exception as e:
        st.error(f"❌ Error al procesar el PDF: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

elif pdf is not None and not api_key:
    st.warning("⚠️ Debes ingresar tu clave de API de OpenAI para continuar.")
else:
    st.info("💡 Sube un archivo PDF para comenzar el análisis.")

# =========================
# Pie de página
# =========================
st.markdown("---")
st.markdown("""
**RAG Tech Analyzer 🤖**  
Sistema de **Generación Aumentada por Recuperación** (LangChain + OpenAI).  
Analiza documentos, busca contexto y genera conocimiento 💾  
> “Knowledge is power. Augmented knowledge is evolution.” ⚡
""")
st.caption("© 2025 | Modo Tecnológico 🧠 | Desarrollado con Streamlit + LangChain")
