import uvicorn
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# Importa a função do arquivo chat.py que está dentro da pasta 'app'
from app.chat import ask_sispetro 

app = FastAPI(
    title="API Consultor Sispetro - RAG",
    description="Interface de integração para consulta de tutoriais técnicos via IA.",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def read_root():
    # Rota para o Health Check do App Runner (Protocolo HTTP, Caminho /)
    return {"status": "Online", "service": "Sispetro RAG API", "autor": "Vanclércio Rocha Pontes"}

@app.post("/ask")
def handle_ask(request: QueryRequest):
    try:
        # Chama o motor RAG do chat.py
        result = ask_sispetro(request.question)
        
        # Retorna o resultado limpando os metadados para o JSON
        return {
            "answer": result.get('result'),
            "source_documents": [doc.metadata for doc in result.get('source_documents', [])]
        }
    except Exception as e:
        # Log de erro para depuração no console AWS
        print(f"Erro na execução: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Roda na 8080 para casar com o painel do App Runner
    uvicorn.run(app, host="0.0.0.0", port=8080)