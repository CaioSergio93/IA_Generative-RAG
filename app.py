from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI  # Alterado para OpenAI
from langchain.chains import RetrievalQA
import warnings
import os
from dotenv import load_dotenv
load_dotenv()

# Configuração de warnings
warnings.filterwarnings('ignore')

def carregar_documentos(pasta):
    """Carrega todos os PDFs da pasta especificada"""
    if not os.path.exists(pasta):
        print(f"\n❌ Pasta '{pasta}' não encontrada")
        print("ℹ️ Crie uma pasta 'pdf' e coloque seus documentos lá")
        raise FileNotFoundError(f"Pasta '{pasta}' não existe")
    
    print(f"\n🔍 Carregando documentos da pasta '{pasta}'...")
    try:
        loader = DirectoryLoader(
            path=pasta,
            glob="*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        documents = loader.load()
        total_paginas = sum(
            len(pages) if isinstance(pages := doc.metadata.get('page', []), list) else 1 
            for doc in documents
        )
        print(f"✅ {len(documents)} arquivos PDF carregados ({total_paginas} páginas)")
        return documents
    except Exception as e:
        print(f"\n❌ Falha ao carregar documentos: {str(e)}")
        raise

def processar_documentos(documents):
    """Divide os documentos em chunks para processamento"""
    print("\n✂️ Processando documentos...")
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=128,
            length_function=len,
            is_separator_regex=False
        )
        texts = text_splitter.split_documents(documents)
        print(f"📚 {len(texts)} chunks criados para análise")
        return texts
    except Exception as e:
        print(f"\n❌ Falha ao processar documentos: {str(e)}")
        raise

def criar_vetorstore(texts, db_name="banco_vetorial"):
    """Cria e salva o banco de dados vetorial"""
    print("\n🧠 Configurando embeddings...")
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        print("🛢️ Criando banco de dados vetorial...")
        db = FAISS.from_documents(texts, embedding_model)
        db.save_local(db_name)
        print(f"✅ Banco vetorial '{db_name}' criado ({db.index.ntotal} vetores)")
        return db, embedding_model
    except Exception as e:
        print(f"\n❌ Falha ao criar vetorstore: {str(e)}")
        raise

def carregar_modelo_llm():
    """Carrega o modelo da OpenAI"""
    print("\n⚙️ Configurando modelo OpenAI...")
    try:
        # Configure sua API key como variável de ambiente
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("Chave API da OpenAI não encontrada. Configure OPENAI_API_KEY")
        
        llm = OpenAI(
            model_name="gpt-3.5-turbo-instruct",  # Modelo mais rápido e econômico
            temperature=0.1,
            max_tokens=2000
        )
        print("✅ Modelo OpenAI carregado com sucesso!")
        return llm
    except Exception as e:
        print(f"\n❌ Falha ao carregar modelo OpenAI: {str(e)}")
        print("⚠️ Possíveis soluções:")
        print("- Verifique se a variável OPENAI_API_KEY está configurada")
        print("- Confira seu saldo/créditos na OpenAI")
        print("- Verifique sua conexão com a internet")
        raise

def criar_sistema_rag(db, embedding_model, llm):
    """Configura a cadeia RAG completa"""
    print("\n🔗 Configurando sistema RAG...")
    try:
        retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 10}
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            verbose=False
        )
        print("✅ Sistema RAG pronto para uso!")
        return qa_chain
    except Exception as e:
        print(f"\n❌ Falha ao configurar sistema RAG: {str(e)}")
        raise

def ask_local_rag(qa_chain, question):
    """Processa perguntas e mostra respostas formatadas"""
    print(f"\n🔎 Processando: '{question}'")
    try:
        result = qa_chain.invoke({"query": question})
        
        print("\n💡 Resposta:")
        print(result["result"].strip())
        
        if result["source_documents"]:
            print("\n📚 Fontes utilizadas:")
            for i, doc in enumerate(result["source_documents"][:3], 1):
                source = os.path.basename(doc.metadata.get('source', 'documento'))
                page = doc.metadata.get('page', 'N/A')
                print(f"\n📌 Fonte {i}: {source} (Página {page})")
                print("   " + doc.page_content[:250].replace('\n', ' ').strip() + "...")
        else:
            print("\nℹ️ Nenhuma fonte específica foi utilizada para esta resposta")
            
    except Exception as e:
        print(f"\n⚠️ Erro ao processar pergunta: {str(e)}")

def main():
    """Função principal"""
    # Configurações
    PASTA_PDF = "pdf"
    
    try:
        print("\n" + "="*50)
        print("🚀 SISTEMA RAG COM OPENAI - INICIANDO")
        print("="*50)
        
        # 1. Carregar e processar documentos
        documentos = carregar_documentos(PASTA_PDF)
        textos = processar_documentos(documentos)
        
        # 2. Criar banco vetorial
        banco_vetorial, embeddings = criar_vetorstore(textos)
        
        # 3. Carregar modelo LLM
        modelo_llm = carregar_modelo_llm()
        
        # 4. Configurar sistema RAG
        sistema_rag = criar_sistema_rag(banco_vetorial, embeddings, modelo_llm)
        
        # 5. Interface de usuário
        print("\n" + "="*50)
        print("💡 Dicas: Pergunte sobre os documentos carregados")
        print("📋 Exemplos:")
        print("- 'Resuma os principais pontos do documento'")
        print("- 'Quais são as despesas mais relevantes?'")
        print("- 'Compare informações entre diferentes páginas'")
        print("Digite 'sair' para encerrar")
        print("="*50)
        
        while True:
            try:
                pergunta = input("\n❓ Sua pergunta: ").strip()
                if pergunta.lower() in ['sair', 'exit', 'quit']:
                    break
                if pergunta:
                    ask_local_rag(sistema_rag, pergunta)
                else:
                    print("⚠️ Por favor, digite uma pergunta válida")
            except KeyboardInterrupt:
                print("\n⚠️ Operação interrompida pelo usuário")
                break
                
    except Exception as e:
        print(f"\n❌ ERRO CRÍTICO: {str(e)}")
    finally:
        print("\n🔴 Sistema encerrado")

if __name__ == "__main__":
    # Configure sua API key antes de executar
    os.environ["OPENAI_API_KEY"] = "sua-chave-aqui"
    #os.environ["OPENAI_API_KEY"] = ""
    main()