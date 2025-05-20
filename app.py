import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI, ChatOpenAI
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
import os
import tempfile

# Configuração inicial
st.set_page_config(page_title="ChatRAG Inteligente", page_icon="🧠")

# Verificação de segurança
def verificar_ambiente():
    """Verifica se as configurações necessárias estão presentes"""
    if 'openai' not in st.secrets or 'api_key' not in st.secrets.openai:
        st.error("🔒 Configuração de API não encontrada. Verifique os segredos do aplicativo.")
        st.stop()

# Funções do sistema RAG
def processar_pdf(uploaded_file):
    """Processa um arquivo PDF carregado de forma segura"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        return documents
    except Exception as e:
        st.error(f"Erro ao processar PDF: {str(e)}")
        return None
    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)

def processar_documentos(documents):
    """Divide os documentos em chunks otimizados"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(documents)

def criar_vetorstore(texts):
    """Cria o banco de dados vetorial com embeddings otimizados"""
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return FAISS.from_documents(texts, embedding_model), embedding_model

def carregar_modelo_llm():
    """Carrega o modelo da OpenAI com configurações otimizadas"""
    return OpenAI(
        temperature=0.1,
        max_tokens=2000,
        api_key=st.secrets.openai.api_key  # Acessa a chave dos segredos
    )

def criar_sistema_rag(db, embedding_model, llm):
    """Configura a cadeia RAG completa com busca semântica"""
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10}
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        verbose=False
    )

def gerar_sugestoes_contextuais(documentos, llm):
    """Gera perguntas contextualizadas usando análise de conteúdo"""
    try:
        # Análise dos 3 primeiros documentos para performance
        contexto = "\n".join(doc.page_content[:500] for doc in documentos[:3])
        
        prompt = PromptTemplate.from_template("""
        Com base nestes trechos de documentos, gere 3 perguntas específicas:
        {contexto}

        Perguntas relevantes (em português, numeradas):
        1. """)
        
        chain = LLMChain(
            llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3),
            prompt=prompt
        )
        resultado = chain.run(contexto=contexto)
        return [p.split(". ", 1)[1] for p in resultado.split("\n") if ". " in p][:3]
    except Exception:
        return [
            "Quais são os pontos principais?",
            "Há dados estatísticos relevantes?",
            "Quais recomendações são apresentadas?"
        ]

# Interface Streamlit
def main():
    st.title("🧠 ChatRAG Inteligente")
    st.caption("Faça upload de documentos e converse com seu conteúdo")
    
    # Verificação inicial
    verificar_ambiente()

    # Gerenciamento de estado
    if 'qa_chain' not in st.session_state:
        st.session_state.update({
            'qa_chain': None,
            'documentos': [],
            'sugestoes': [],
            'messages': []
        })

    # Seção de upload de documentos
    if not st.session_state.qa_chain:
        st.subheader("📤 Carregar Documentos")
        uploaded_files = st.file_uploader(
            "Selecione arquivos PDF", 
            type="pdf",
            accept_multiple_files=True,
            help="Você pode selecionar vários arquivos de uma vez"
        )

        if uploaded_files and st.button("Processar Documentos", type="primary"):
            with st.status("Processando...", expanded=True) as status:
                try:
                    # Processamento em etapas com feedback
                    st.write("🔍 Lendo documentos...")
                    documentos = []
                    for file in uploaded_files:
                        docs = processar_pdf(file)
                        if docs:
                            documentos.extend(docs)
                            st.write(f"✓ {file.name} processado")
                    
                    if not documentos:
                        st.error("Nenhum conteúdo válido encontrado")
                        return

                    st.session_state.documentos = documentos
                    
                    st.write("✂️ Dividindo conteúdo...")
                    textos = processar_documentos(documentos)
                    
                    st.write("🧠 Criando índice de busca...")
                    db, embeddings = criar_vetorstore(textos)
                    
                    st.write("⚙️ Configurando IA...")
                    llm = carregar_modelo_llm()
                    
                    st.write("💡 Gerando sugestões...")
                    st.session_state.sugestoes = gerar_sugestoes_contextuais(documentos, llm)
                    
                    st.write("🔗 Conectando sistema RAG...")
                    st.session_state.qa_chain = criar_sistema_rag(db, embeddings, llm)
                    
                    status.update(label="✅ Processamento completo!", state="complete")
                    st.balloons()
                except Exception as e:
                    st.error(f"⚠️ Erro no processamento: {str(e)}")

    # Seção de interação
    if st.session_state.qa_chain:
        st.divider()
        st.subheader("💬 Chat com os Documentos")

        # Sugestões de perguntas
        if st.session_state.sugestoes:
            st.markdown("**Sugestões:**")
            cols = st.columns(3)
            for i, pergunta in enumerate(st.session_state.sugestoes):
                with cols[i % 3]:
                    if st.button(pergunta, key=f"sug_{i}", help="Clique para usar esta pergunta"):
                        st.session_state.pergunta = pergunta

        # Histórico de conversa
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "sources" in msg:
                    with st.expander("📌 Fontes"):
                        for src in msg["sources"]:
                            st.caption(f"📄 {src['doc']} (página {src['page']})")
                            st.markdown(f"> {src['content']}")

        # Input de pergunta
        if prompt := st.chat_input("Digite sua pergunta...") or st.session_state.get("pergunta"):
            if "pergunta" in st.session_state:
                prompt = st.session_state.pergunta
                del st.session_state.pergunta
            
            # Adiciona pergunta ao histórico
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Processa resposta
            with st.chat_message("assistant"):
                with st.spinner("Analisando..."):
                    try:
                        result = st.session_state.qa_chain({"query": prompt})
                        resposta = result["result"]
                        
                        st.markdown(resposta)
                        
                        # Extrai fontes
                        fontes = [{
                            "doc": os.path.basename(doc.metadata.get("source", "Documento")),
                            "page": doc.metadata.get("page", "N/A"),
                            "content": doc.page_content[:250] + "..."
                        } for doc in result["source_documents"][:3]]
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": resposta,
                            "sources": fontes
                        })
                    except Exception as e:
                        st.error(f"❌ Erro ao gerar resposta: {str(e)}")

        # Controles
        st.divider()
        if st.button("🔄 Reiniciar com Novos Documentos", type="secondary"):
            st.session_state.clear()
            st.rerun()

if __name__ == "__main__":
    main()
