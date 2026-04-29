import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

def ask_sispetro(question):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    index_name = os.getenv("PINECONE_INDEX_NAME")

    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # TEMPLATE REFINADO: Foco em identidade e objetividade
    template = """
    Você é o Assistente Técnico Especialista do SISPETRO. Sua missão é fornecer suporte preciso baseado nos manuais do sistema.

    DIRETRIZES DE COMUNICAÇÃO:
    1. IDENTIDADE: Se o usuário apenas cumprimentar (ex: "Oi", "Olá", "Bom dia"), responda: "Olá! Sou o assistente técnico do Sispetro. Como posso te ajudar com suas dúvidas contábeis ou técnicas hoje?"
    
    2. OBJETIVIDADE TÉCNICA: Se o usuário fizer uma pergunta técnica direta, NÃO gaste tempo com saudações ou apresentações. Vá direto ao ponto.
    
    3. FLUIDEZ: Use parágrafos explicativos e naturais. Evite listas numeradas a menos que seja um passo a passo estrito. Integre caminhos de menu como: "No Sispetro, navegue em Relatórios > Vendas..."
    
    4. NÃO INVENTE: Se a resposta não estiver no contexto, diga: "Desculpe, não localizei esse procedimento específico nos manuais técnicos do Sispetro disponíveis. Posso ajudar com outro tema?"

    CONTEXTO:
    {context}

    PERGUNTA: 
    {question}

    RESPOSTA PROFISSIONAL:"""

    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = rag_chain.invoke(question)
    return {"result": response, "source_documents": []}