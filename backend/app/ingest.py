import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

# Carrega as chaves do .env que está na pasta /backend
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

def run_ingestion():
    print("⏳ Iniciando o processo de LOAD (Ingestão) para o Pinecone...")
    
    # 1. Localizar a pasta de dados
    data_path = os.path.join(BASE_DIR, "data")
    
    if not os.path.exists(data_path):
        print(f"❌ Erro: A pasta {data_path} não foi encontrada!")
        return

    # 2. Carregar os documentos .txt com correção de Encoding
    print(f"📂 Lendo arquivos em: {data_path}")
    
    # O segredo está no loader_kwargs passando o encoding utf-8
    loader = DirectoryLoader(
        data_path, 
        glob="./*.txt", 
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'}
    )
    
    try:
        raw_documents = loader.load()
        print(f"📄 {len(raw_documents)} documentos carregados com sucesso.")
    except Exception as e:
        print(f"⚠️ Falha no UTF-8, tentando carregar com Latin-1...")
        # Fallback para arquivos salvos em formato Windows antigo (ANSI/Latin-1)
        loader = DirectoryLoader(
            data_path, 
            glob="./*.txt", 
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'latin-1'}
        )
        try:
            raw_documents = loader.load()
            print(f"📄 {len(raw_documents)} documentos carregados (via Latin-1).")
        except Exception as e2:
            print(f"❌ Erro fatal ao carregar documentos: {e2}")
            return

    # 3. Fragmentação (Chunking) conforme Metodologia do TCC
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100
    )
    docs = text_splitter.split_documents(raw_documents)
    print(f"✂️  Documentos fatiados em {len(docs)} chunks.")

    # 4. Configurar Embeddings e Pinecone
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    if not index_name:
        print("❌ Erro: PINECONE_INDEX_NAME não definido no .env")
        return

    print(f"🚀 Enviando vetores para o índice '{index_name}' no Pinecone...")
    
    try:
        PineconeVectorStore.from_documents(
            docs, 
            embeddings, 
            index_name=index_name
        )
        print("✅ SUCESSO! O 'cérebro' do seu TCC já está na nuvem.")
    except Exception as e:
        print(f"❌ Erro na subida para o Pinecone: {e}")

if __name__ == "__main__":
    run_ingestion()