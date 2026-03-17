import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# 1. Configuração de ambiente
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

def ask_sispetro(question):
    # Setup dos componentes base
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # 2. O Prompt (Exatamente como pede o Tutorial de RAG da doc que você mandou)
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

    # 3. A "Chain" Moderna (LCEL) - Isso substitui o RetrievalQA que sumiu
    # O format_docs junta os pedaços de texto encontrados no Pinecone
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 4. Execução
    response = rag_chain.invoke(question)
    
    return {"result": response}

if __name__ == "__main__":
    while True:
        pergunta = input("\nO que deseja realizar? ")
        if pergunta.lower() in ['sair', 'exit']: break
        print("🔍 Analisando manuais...")
        resposta = ask_sispetro(pergunta)
        print(f"\n💡 RESPOSTA:\n{resposta['result']}\n")