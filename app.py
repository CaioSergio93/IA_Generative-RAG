import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI, ChatOpenAI
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import tempfile

# Configuração inicial
load_dotenv()
st.set_page_config(page_title="ChatRAG Inteligente", page_icon="🧠")

# Funções do sistema RAG
def processar_pdf(uploaded_file):
    """Processa um arquivo PDF carregado"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        return documents
    finally:
        os.unlink(tmp_path)

def processar_documentos(documents):
    """Divide os documentos em chunks para processamento"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False
    )
    texts = text_splitter.split_documents(documents)
    return texts

def criar_vetorstore(texts):
    """Cria o banco de dados vetorial"""
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    db = FAISS.from_documents(texts, embedding_model)
    return db, embedding_model

def carregar_modelo_llm():
    """Carrega o modelo da OpenAI"""
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("Chave API da OpenAI não configurada")
    
    return OpenAI(temperature=0.1, max_tokens=2000)

def criar_sistema_rag(db, embedding_model, llm):
    """Configura a cadeia RAG completa"""
    retriever = db.as_retriever(search_kwargs={"k": 3})
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

def gerar_sugestoes_contextuais(documentos, llm):
    """Gera sugestões de perguntas baseadas no conteúdo real dos documentos"""
    # Primeiro extraímos trechos relevantes para análise
    trechos_analise = []
    for doc in documentos[:5]:  # Analisamos apenas os primeiros para performance
        content = doc.page_content[:1000]  # Pegamos apenas o início para análise
        trechos_analise.append(f"---\n{content}\n---")
    
    texto_analise = "\n".join(trechos_analise)[:5000]  # Limita o tamanho
    
    # Template para geração de perguntas
    template = """
    Com base nos seguintes trechos de documentos, gere 5 perguntas relevantes que
    um usuário poderia fazer sobre este conteúdo. As perguntas devem ser específicas
    e demonstrar compreensão do material.
    
    Trechos:
    {texto}
    
    Perguntas sugeridas (uma por linha, em português):
    1. 
    """
    
    prompt = PromptTemplate(template=template, input_variables=["texto"])
    chain = LLMChain(llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3), prompt=prompt)
    
    try:
        resultado = chain.run(texto=texto_analise)
        perguntas = [p.strip() for p in resultado.split("\n") if p.strip() and p.strip()[0].isdigit()]
        return [p.split(". ", 1)[1] for p in perguntas if ". " in p][:5]
    except Exception:
        # Fallback para perguntas genéricas se houver erro
        return [
            "Quais são os pontos principais deste documento?",
            "Pode resumir as informações mais relevantes?",
            "Quais dados ou estatísticas são apresentados?",
            "Existem recomendações ou conclusões importantes?",
            "Quem são as partes ou entidades mencionadas?"
        ]

# Interface Streamlit
def main():
    st.title("🧠 ChatRAG Inteligente")
    st.caption("Faça upload de documentos e obtenha respostas inteligentes")
    
    # Sidebar
    with st.sidebar:
        st.header("Configurações")
        if "OPENAI_API_KEY" not in os.environ:
            st.error("Configure a OPENAI_API_KEY no ambiente")
            return
        
        st.markdown("""
        ### Como usar:
        1. Carregue seus documentos
        2. Aguarde o processamento
        3. Faça perguntas ou use as sugestões
        """)
    
    # Estado da sessão
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
        st.session_state.documentos = []
        st.session_state.sugestoes = []
    
    # Upload de documentos
    if not st.session_state.qa_chain:
        st.subheader("📤 Carregar Documentos")
        uploaded_files = st.file_uploader(
            "Selecione arquivos PDF", 
            type="pdf",
            accept_multiple_files=True
        )
        
        if uploaded_files and st.button("Processar Documentos"):
            with st.status("Processando...", expanded=True) as status:
                try:
                    st.write("Lendo arquivos PDF...")
                    documentos = []
                    for file in uploaded_files:
                        st.write(f"Processando: {file.name}")
                        documentos.extend(processar_pdf(file))
                    
                    if not documentos:
                        st.error("Nenhum conteúdo válido encontrado")
                        return
                    
                    st.session_state.documentos = documentos
                    
                    st.write("Preparando documentos...")
                    textos = processar_documentos(documentos)
                    
                    st.write("Criando banco de dados vetorial...")
                    db, embeddings = criar_vetorstore(textos)
                    
                    st.write("Configurando modelo de IA...")
                    llm = carregar_modelo_llm()
                    
                    st.write("Gerando sugestões de perguntas...")
                    st.session_state.sugestoes = gerar_sugestoes_contextuais(documentos, llm)
                    
                    st.write("Criando sistema RAG...")
                    st.session_state.qa_chain = criar_sistema_rag(db, embeddings, llm)
                    
                    status.update(label="Processamento completo!", state="complete")
                    st.success("Pronto para perguntas!")
                    st.balloons()
                except Exception as e:
                    st.error(f"Erro: {str(e)}")
    
    # Seção de perguntas e respostas
    if st.session_state.qa_chain:
        st.divider()
        st.subheader("💬 Chat com os Documentos")
        
        # Mostrar sugestões
        if st.session_state.sugestoes:
            st.markdown("**Sugestões de perguntas:**")
            cols = st.columns(2)
            for i, pergunta in enumerate(st.session_state.sugestoes):
                with cols[i % 2]:
                    if st.button(pergunta, key=f"sug_{i}", use_container_width=True):
                        st.session_state.pergunta = pergunta
        
        # Histórico de chat
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "sources" in msg:
                    with st.expander("Fontes"):
                        for src in msg["sources"]:
                            st.caption(f"📄 {src['doc']} (página {src['page']})")
                            st.markdown(f"> {src['content']}")
        
        # Input de pergunta
        if prompt := st.chat_input("Digite sua pergunta...") or st.session_state.get("pergunta"):
            if "pergunta" in st.session_state:
                prompt = st.session_state.pergunta
                del st.session_state.pergunta
            
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Pensando..."):
                    try:
                        result = st.session_state.qa_chain({"query": prompt})
                        resposta = result["result"]
                        
                        st.markdown(resposta)
                        
                        fontes = []
                        for doc in result["source_documents"][:3]:
                            fontes.append({
                                "doc": os.path.basename(doc.metadata.get("source", "Documento")),
                                "page": doc.metadata.get("page", "N/A"),
                                "content": doc.page_content[:200] + "..."
                            })
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": resposta,
                            "sources": fontes
                        })
                    except Exception as e:
                        st.error(f"Erro: {str(e)}")
        
        if st.button("🔄 Carregar Novos Documentos"):
            st.session_state.clear()
            st.rerun()

if __name__ == "__main__":
    main()