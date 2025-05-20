ChatRAG Inteligente - Sistema de Perguntas e Respostas sobre Documentos

📌 Visão Geral
O ChatRAG Inteligente é um sistema avançado de Perguntas e Respostas sobre Documentos (QA) que utiliza IA Generativa (RAG - Retrieval-Augmented Generation) para extrair informações precisas de arquivos PDF.

📂 Carregue seus documentos (PDF)

🤖 Faça perguntas em linguagem natural

🔍 Obtenha respostas com referências diretas aos documentos

💡 Receba sugestões de perguntas baseadas no conteúdo

✨ Principais Funcionalidades
✅ Upload de múltiplos PDFs – Interface simples para carregar documentos
✅ Processamento automático – Extração e indexação inteligente do conteúdo
✅ Sugestões de perguntas contextualizadas – IA analisa o conteúdo e sugere perguntas relevantes
✅ Respostas baseadas em evidências – Sistema RAG busca nos documentos antes de responder
✅ Referência às fontes – Mostra exatamente de qual página/arquivo veio a informação

🛠️ Tecnologias Utilizadas
Tecnologia	Descrição
Python	Linguagem principal
Streamlit	Interface web interativa
LangChain	Framework para sistemas RAG
FAISS (Facebook AI Similarity Search)	Banco de dados vetorial para buscas semânticas
OpenAI API (GPT-3.5-turbo)	Modelo de linguagem para respostas
Hugging Face (all-MiniLM-L6-v2)	Modelo de embeddings para análise de texto
🚀 Como Usar
1️⃣ Executando Localmente
Pré-requisitos
Python 3.8+

Conta na OpenAI (para API Key)

Passos
Clone o repositório
git clone https://github.com/CaioSergio93/IA_Generative-RAG.git
cd IA_Generative-RAG

Instale as dependências:
pip install -r requirements.txt
Crie um arquivo .env e adicione sua OpenAI API Key:

env
OPENAI_API_KEY=sua_chave_aqui

Execute o app:
streamlit run app.py

Acesse no navegador:
→ http://localhost:8501

2️⃣ Hospedagem na Nuvem (Streamlit Sharing / Hugging Face Spaces)
Faça upload do projeto para um repositório Git (GitHub, GitLab, etc.)

Configure a variável OPENAI_API_KEY nas configurações do serviço

Deploy automático no Streamlit Cloud ou Hugging Face

📂 Estrutura do Projeto
IA_Generative-RAG/
├── app.py                # Código principal (Streamlit)
├── requirements.txt      # Dependências Python
├── .env.example          # Modelo para variáveis de ambiente
├── README.md             # Documentação
└── pdf/                  # Pasta para upload de documentos (opcional)
📞 Contato
💌 E-mail: caio.dev.system@gmail.com
📸 Instagram: @caiosergiom

📜 Licença
Este projeto está sob a licença MIT. Consulte o arquivo LICENSE para mais detalhes.

Feito com ❤️ por Caio Sérgio 🚀
