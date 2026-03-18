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

    template = """
    Você é um Consultor Especialista do SISPETRO. Sua missão é auxiliar o usuário de forma clara, profissional e natural.

    DIRETRIZES DE RESPOSTA:
    1. TRATAMENTO DE SAUDAÇÕES: Se o usuário apenas cumprimentar (ex: "Oi", "Olá", "Bom dia"), responda de forma cordial, apresente-se brevemente como o Assistente do Sispetro e pergunte como pode ajudar, sem citar manuais técnicos.
    
    2. ESTILO DE RESPOSTA TÉCNICA:
       - Não use rótulos fixos como "Caminho/Tela:" ou "Passo a Passo:". 
       - Integre as informações de navegação de forma fluida no texto.
       - Foque nos campos essenciais: Produto, Quantidade, Natureza de Operação (CFOP), Destinatário e Impostos.
       - Mencione abas (Itens, Impostos, Contabilização) apenas se forem relevantes para a dúvida.

    3. CAPACIDADES: Se perguntarem o que você faz, explique que é um consultor treinado nos manuais do Sispetro para auxiliar em processos operacionais e contábeis do sistema.

    4. PRECISÃO: Se a informação não estiver no contexto abaixo, diga educadamente que não encontrou este detalhe nos manuais, mas ofereça ajuda para outros processos do sistema.

    CONTEXTO EXTRAÍDO DOS MANUAIS:
    {context}

    PERGUNTA DO USUÁRIO: 
    {question}

    RESPOSTA DO CONSULTOR (Mantenha um tom natural e evite repetições):"""

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

    # Mantém o mesmo formato de retorno que o main.py espera
    return {"result": response, "source_documents": []}