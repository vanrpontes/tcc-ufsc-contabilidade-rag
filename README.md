**Sispetro AI — Chatbot RAG para Manuais Técnicos**

Protótipo de chatbot baseado em Retrieval-Augmented Generation (RAG), desenvolvido como Trabalho de Conclusão de Curso (TCC) em Ciências Contábeis pela Universidade Federal de Santa Catarina (UFSC).

O Potencial da Inteligência Artificial pelo Método RAG para a Gestão da Informação Contábil
Autor: Vanclércio da Rocha Pontes
Orientador: Prof. Dr. Valmir Emil Hoffmann
UFSC — Florianópolis, 2026

🔗 Aplicação em produção: https://chatbotsispetro.com.br

---

📌 Sobre o projeto
O Sispetro AI é um assistente conversacional que utiliza a arquitetura RAG para consultar, de forma precisa e contextualizada, a base de tutoriais técnicos do sistema de gestão Sispetro — um ERP voltado a postos de combustíveis e distribuidoras, desenvolvido pela Futura Tecnologia.
Em vez de depender exclusivamente do conhecimento generalista de um modelo de linguagem, o sistema recupera trechos relevantes de uma base documental própria — extraída do Atlassian Confluence — antes de gerar cada resposta, reduzindo o risco de respostas fabricadas (alucinações) e ancorando o conteúdo gerado em fontes verificáveis.
Este repositório contém o código-fonte completo utilizado na pesquisa, incluindo os scripts de extração e pré-processamento da base de conhecimento, a configuração do pipeline RAG e a implementação do backend e do frontend da aplicação. O código é disponibilizado com o objetivo de assegurar a transparência e a reprodutibilidade dos resultados apresentados no TCC.

---

🏗️ Arquitetura
O sistema é estruturado em três camadas desacopladas:
Frontend (Streamlit)  ⇄  Backend (FastAPI)  ⇄  Pinecone (Base Vetorial)
                                │
                                ▼
                          OpenAI API (GPT-3.5 Turbo)

**Pipeline de ingestão dos dados:**
1. Extração — script Python que coleta as páginas do espaço técnico do Sispetro no Atlassian Confluence via API, salvando o conteúdo bruto em HTML.
2. Pré-processamento — limpeza dos arquivos HTML com Beautiful Soup, convertendo o conteúdo para texto puro (.txt, UTF-8), preservando título e URL de origem de cada página.
3. Fragmentação (chunking) — segmentação dos textos com o RecursiveCharacterTextSplitter do LangChain.
4. Vetorização (embeddings) — geração de vetores semânticos com o modelo text-embedding-3-small da OpenAI.
5. Indexação — armazenamento dos vetores no Pinecone, banco de dados vetorial em nuvem.

**Pipeline de consulta (em tempo de execução):**
1. O usuário envia uma pergunta pela interface Streamlit.
2. O backend FastAPI recupera, via similaridade semântica, os trechos mais relevantes da base vetorial.
3. O modelo GPT-3.5 Turbo (OpenAI) gera a resposta com base no contexto recuperado.
4. A resposta é exibida ao usuário, mantendo o histórico da sessão.

---

**📁 Estrutura do repositório**

├── backend/                        # API FastAPI: lógica do RAG, embeddings e integração com OpenAI/Pinecone
│   ├── app/
│   │   ├── main.py                 # Entrypoint da API (porta 8080)
│   │   └── chat.py                 # Motor do RAG (retriever + geração)
│   └── data/                       # Base de tutoriais extraída (não versionada — ver .gitignore)
├── frontend/                       # Interface Streamlit (porta 8501)
│   └── app_web.py
├── research/                       # Artefatos acadêmicos e de validação do TCC
│   ├── academic/                   # Documentos acadêmicos oficiais
│   │   ├── TCC_Vanclercio_Pontes_2026.pdf
│   │   └── slides_apresentacao_TCC_2026.pdf
│   ├── validation/                 # Evidências da validação empírica do protótipo
│   │   ├── evidencias-perguntas-teste/   # Prints das respostas geradas pelo chatbot
│   │   ├── historico_24_prompts_sispetro_ai.pdf
│   │   └── instrumento_validacao_sispetro.xlsx
│   └── knowledge-base/             # Inventário da base de conhecimento
│       ├── extrair_arquivos.py     # Script para listar arquivos da base
│       └── inventario_base_sispetro.xlsx
├── docker-compose.yml              # Orquestração dos serviços (backend + frontend)
├── Dockerfile
└── .gitignore

---

**🚀 Como executar localmente**

**Pré-requisitos:**
- Python 3.10+
- Docker e Docker Compose
- Conta na OpenAI com chave de API ativa
- Conta no Pinecone com um índice vetorial criado

**Passo a passo**
1. Clone o repositório:
bash
   git clone https://github.com/vanrpontes/tcc-ufsc-contabilidade-rag.git
   cd tcc-ufsc-contabilidade-rag

2. Crie um arquivo .env na raiz do projeto com suas próprias credenciais:
env
   OPENAI_API_KEY=sua_chave_aqui
   PINECONE_API_KEY=sua_chave_aqui
   PINECONE_INDEX_NAME=nome_do_seu_indice

3. (Opcional) Execute o pipeline de extração e ingestão da base de conhecimento, caso deseje popular o índice vetorial com seus próprios dados:
bash
   python backend/scripts/extract.py
   python backend/scripts/ingest.py

4. Suba os containers:
bash
   docker compose up --build

5. Acesse a aplicação em http://localhost:8501.

⚠️ Sobre credenciais: por motivos de segurança, o arquivo .env não está incluído neste repositório (ver .gitignore). É necessário configurar suas próprias chaves de API da OpenAI e do Pinecone para executar o projeto. Nenhuma credencial real foi exposta neste repositório público.

---

**🎓 Contexto acadêmico e validação**
A validação empírica do protótipo, incluindo o instrumento de avaliação utilizado, os resultados obtidos e a discussão crítica das limitações identificadas, está detalhada no Capítulo 4 do TCC. O conjunto completo de perguntas-teste, respostas geradas pelo sistema e trechos correspondentes da documentação técnica utilizados na checagem documental está disponível no Apêndice do trabalho.

---

**📄 Licença**
Este projeto é disponibilizado para fins acadêmicos e de pesquisa. Sinta-se à vontade para estudar, adaptar e reproduzir o experimento, mantendo a devida citação ao trabalho original.

---

**✉️ Contato**
Para dúvidas sobre o projeto ou sobre o TCC, entre em contato:
 - E-mail: vrpontes@outlook.com
 - LinkedIn: linkedin.com/in/vanrpontes

---
**UFSC — Florianópolis, 2026**
---
