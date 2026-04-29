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

    # REFINAMENTO DO TEMPLATE: Foco em fluidez e evitar redundância
    template = """
    Você é um Consultor Especialista do SISPETRO. Sua missão é auxiliar o usuário de forma clara, profissional e técnica.

    REGRAS CRÍTICAS DE COMPORTAMENTO:
    1. EVITE REPETIÇÕES: Não se apresente em todas as respostas. Se o usuário fizer uma pergunta técnica direta, responda diretamente sem frases de introdução como "Olá, sou o consultor...".
    
    2. TRATAMENTO DE SAUDAÇÕES: 
       - Se for a PRIMEIRA interação (ex: apenas "Oi", "Olá"), apresente-se brevemente.
       - Se for uma pergunta técnica após um cumprimento, IGNORE a saudação e foque na solução técnica.

    3. ESTILO DE RESPOSTA:
       - Responda de forma fluida. Em vez de listas rígidas, use parágrafos explicativos.
       - Integre caminhos de menus naturalmente: "Acesse o menu Relatórios > Controladoria..." em vez de usar rótulos como "Caminho:".
       - Dê prioridade aos campos: Produto, Quantidade, CFOP, Destinatário e Impostos.

    4. PRECISÃO: Se o contexto abaixo não contiver a resposta, informe que não localizou este procedimento específico nos manuais atuais e coloque-se à disposição para outras dúvidas do sistema.

    CONTEXTO DOS MANUAIS (Utilize estritamente para responder):
    {context}

    PERGUNTA DO USUÁRIO: 
    {question}

    RESPOSTA (Direta, técnica e sem repetições de identidade):"""

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

    # Retorno compatível com main.py
    return {"result": response, "source_documents": []}