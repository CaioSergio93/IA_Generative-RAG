from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
import warnings
import os

# Configuração de warnings
warnings.filterwarnings('ignore')

def carregar_documentos(pasta):
    if not os.path.exists(pasta):
        raise ValueError(f"Pasta '{pasta}' não encontrada")
    
    loader = DirectoryLoader(
        path=pasta,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    
    documents = loader.load()
    total_paginas = sum([len(doc.metadata.get('page', [])) if isinstance(doc.metadata.get('page', []), list) else 1 for doc in documents])
    print(f"\n✅ Total de documentos carregados: {len(documents)} arquivos PDF")
    print(f"📄 Total de páginas processadas: {total_paginas}")
    return documents

def processar_documentos(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=128,
        length_function=len
    )
    texts = text_splitter.split_documents(documents)
    print(f"\n🔪 Documentos divididos em {len(texts)} chunks de texto")
    return texts

def criar_vetorstore(texts, db_name="meu_banco_vetorial"):
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    db = FAISS.from_documents(texts, embedding_model)
    db.save_local(db_name)
    print(f"\n🛢️ Banco de dados vetorial '{db_name}' criado com {db.index.ntotal} vetores")
    return db, embedding_model

def carregar_modelo_llm(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado em {model_path}")
    
    llm = LlamaCpp(
        model_path=model_path,
        temperature=0.1,
        max_tokens=2000,
        n_ctx=2048,
        n_gpu_layers=0,
        verbose=False
    )
    print("\n🧠 Modelo LLM carregado com sucesso")
    return llm

def criar_sistema_rag(db, embedding_model, llm):
    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

def ask_local_rag(qa_chain, question):
    print(f"\n🔍 Buscando resposta para: '{question}'")
    result = qa_chain({"query": question})
    
    print("\n🔵 Resposta:")
    print(result["result"])
    
    print("\n📄 Fontes relacionadas:")
    for i, doc in enumerate(result["source_documents"], 1):
        source = os.path.basename(doc.metadata.get('source', 'desconhecido'))
        page = doc.metadata.get('page', 'N/A')
        print(f"\n📌 Fonte {i}: {source} (Página {page})")
        print(doc.page_content[:300].replace('\n', ' ') + "...")

def main():
    # Configurações
    pasta_pdf = "pdf"
    model_path = "models/llama-2-7b-chat.Q4_K_M.gguf"
    
    try:
        print("🚀 Iniciando o sistema RAG...")
        
        documents = carregar_documentos(pasta_pdf)
        texts = processar_documentos(documents)
        db, embedding_model = criar_vetorstore(texts)
        llm = carregar_modelo_llm(model_path)
        qa_chain = criar_sistema_rag(db, embedding_model, llm)
        
        print("\n💡 Exemplos de perguntas para testar:")
        print("1. Qual foi a receita total em Janeiro?")
        print("2. Compare as despesas entre os meses")
        print("3. Mostre os dados de folha de pagamento")
        
        print("\n🤖 Agente RAG Pronto - Digite 'sair' para encerrar")
        while True:
            query = input("\n💬 Sua pergunta: ").strip()
            if query.lower() in ['sair', 'exit', 'quit']:
                break
            if query:
                ask_local_rag(qa_chain, query)
            else:
                print("⚠️ Por favor, digite uma pergunta válida")
                
    except Exception as e:
        print(f"\n❌ Erro crítico: {str(e)}")
    finally:
        print("\n🔴 Sistema encerrado")

if __name__ == "__main__":
    main()