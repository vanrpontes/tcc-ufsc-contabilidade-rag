import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Iniciar o FastAPI (É isso que o Uvicorn procura)
app = FastAPI(title="Consultor Especialista SISPETRO - TCC UFSC")

# Modelo para receber a pergunta via JSON
class QuestionRequest(BaseModel):
    question: str

# 2. Rota de Teste (Health Check)
# Se você acessar o link da AWS no navegador, vai aparecer essa mensagem
@app.get("/")
def home():
    return {
        "status": "online", 
        "projeto": "TCC UFSC - RAG Contábil Sispetro",
        "autor": "Vanclércio Rocha Pontes"
    }

# 3. Rota Principal do Chat
@app.post("/ask")
def ask_sispetro(request: QuestionRequest):
    question = request.question
    
    # Configuração de componentes (Pega direto das variáveis de ambiente da AWS)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Seu Template Personalizado
    template = """Você é o Consultor Especialista do SISPETRO, um assistente virtual baseado em Inteligência Artificial desenvolvido para o TCC de Ciências Contábeis da UFSC.

    DIRETRIZES DE IDENTIDADE:
    1. QUEM É VOCÊ: Se perguntarem "quem é você", responda: "Sou um Consultor Especialista baseado em IA, treinado para auxiliar na gestão contábil e operacional do software Sispetro."
    2. O QUE É O SISPETRO: Se perguntarem "o que é o Sispetro", responda: "O Sispetro é um ERP especializado para distribuidoras de combustíveis, desenvolvido pela Futura. Ele automatiza rotinas fiscais, estoque, financeiro e integra consultas à ANP e SEFAZ."
    3. NÃO CONFUNDA: Você é o instrutor; o Sispetro é a ferramenta. Nunca diga "Eu sou um ERP". 

    DIRETRIZES TÉCNICAS:
    - APRESENTAÇÃO: Use sua identificação apenas na primeira saudação.
    - SIGLAS: Explique siglas como OC (Ordem de Carregamento) e ICMS ST (Substituição Tributária) na primeira menção.
    - CONTEXTO: Use os dados abaixo para fundamentar suas respostas técnicas.

    CONTEXTO:
    {context}

    PERGUNTA: 
    {question}

    RESPOSTA:"""

    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain LCEL
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Execução
    response = rag_chain.invoke(question)
    
    return {"result": response}