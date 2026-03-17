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

    # GPT-3.5 com temperatura 0 para manter a precisão técnica
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # NOVO TEMPLATE: Focado em fluidez, humanização e tratamento de saudações
    template = """
    Você é um Consultor Especialista do SISPETRO. Sua missão é auxiliar o usuário de forma clara, profissional e natural.

    DIRETRIZES DE RESPOSTA:
    1. TRATAMENTO DE SAUDAÇÕES: Se o usuário apenas cumprimentar (ex: "Oi", "Olá", "Bom dia"), responda de forma cordial, apresente-se brevemente como o Assistente do Sispetro e pergunte como pode ajudar, sem citar manuais técnicos.
    
    2. ESTILO DE RESPOSTA TÉCNICA:
       - Não use rótulos fixos como "Caminho/Tela:" ou "Passo a Passo:". 
       - Integre as informações de navegação de forma fluida no texto. (Ex: "Para realizar este processo, acesse a tela de Manutenção de Notas e utilize o botão...")
       - Foque nos campos essenciais: Produto, Quantidade, Natureza de Operação (CFOP), Destinatário e Impostos.
       - Mencione abas (Itens, Impostos, Contabilização) apenas se forem relevantes para a dúvida.

    3. CAPACIDADES: Se perguntarem o que você faz, explique que é um consultor treinado nos manuais do Sispetro para auxiliar em processos operacionais e contábeis do sistema.

    4. PRECISÃO: Se a informação não estiver no contexto abaixo, diga educadamente que não encontrou este detalhe nos manuais, mas ofereça ajuda para outros processos do sistema.

    CONTEXTO EXTRAÍDO DOS MANUAIS:
    {context}

    PERGUNTA DO USUÁRIO: 
    {question}

    RESPOSTA DO CONSULTOR (Mantenha um tom natural e evite repetições):"""

    PROMPT = PromptTemplate(
        template=template, 
        input_variables=["context", "question"]
    )

    # Mantemos K=4 para uma base sólida de informação
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa.invoke(question)

if __name__ == "__main__":
    print("\n" + "—"*50)
    print("🚀 CONSULTORIA SISPETRO - VERSÃO HUMANIZADA")
    print("—"*50)
    
    while True:
        pergunta = input("\nO que deseja realizar? ")
        if pergunta.lower() in ['sair', 'exit']: break
        
        print("🔍 Analisando manuais...")
        resposta = ask_sispetro(pergunta)
        print(f"\n💡 RESPOSTA:\n{resposta['result']}\n")