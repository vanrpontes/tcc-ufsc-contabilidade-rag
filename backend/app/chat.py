import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# 1. Configuração de ambiente
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

def ask_sispetro(question):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    # Usamos o GPT-3.5 mas com um ajuste de "foco" no prompt
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # AJUSTE DO MEIO-TERMO: Foco no Fluxo Principal do Sispetro
    template = """
    Você é um Consultor Sênior do SISPETRO. Sua tarefa é fornecer um guia prático e equilibrado.
    Não seja genérico demais, mas também não se prenda a integrações externas específicas (como CODIF) a menos que seja perguntado.

    OBJETIVO: Descrever o fluxo padrão de operação dentro das telas do SISPETRO.

    DIRETRIZES:
    1. Foque nos campos essenciais: Produto, Quantidade, Natureza de Operação (CFOP), Destinatário e Impostos.
    2. Mencione as abas do sistema (Itens, Impostos, Contabilização, Dados Adicionais) se estiverem no contexto.
    3. Separe a resposta em: "Caminho/Tela" e "Passo a Passo".
    4. Se o contexto trouxer informações sobre configurações e sobre operações, PRIORIZE a operação manual.
    5. Se houver mais de uma forma de fazer (ex: manual vs automática), mencione brevemente.

    CONTEXTO EXTRAÍDO:
    {context}

    PERGUNTA: 
    {question}

    RESPOSTA DO CONSULTOR:"""

    PROMPT = PromptTemplate(
        template=template, 
        input_variables=["context", "question"]
    )

    # Reduzi o K para 4 para evitar "poluição" de documentos irrelevantes
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa.invoke(question)

if __name__ == "__main__":
    print("\n" + "—"*50)
    print("🚀 CONSULTORIA SISPETRO - VERSÃO CALIBRADA")
    print("—"*50)
    
    while True:
        pergunta = input("\nO que deseja realizar? ")
        if pergunta.lower() in ['sair', 'exit']: break
        
        print("🔍 Analisando manuais...")
        resposta = ask_sispetro(pergunta)
        print(f"\n💡 RESPOSTA:\n{resposta['result']}\n")