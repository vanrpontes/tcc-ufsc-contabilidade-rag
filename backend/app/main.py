from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.chat import ask_sispetro # Importa a sua função calibrada

app = FastAPI(
    title="API Consultor Sispetro - RAG",
    description="Interface de integração para consulta de tutoriais técnicos via IA.",
    version="1.0.0"
)

# Definimos o formato de entrada da pergunta
class QueryRequest(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {"status": "Online", "service": "Sispetro RAG API"}

@app.post("/ask")
def handle_ask(request: QueryRequest):
    try:
        # Chama a lógica de busca e geração que já testamos no chat.py
        result = ask_sispetro(request.question)
        
        return {
            "answer": result['result'],
            "source_documents": [doc.metadata for doc in result.get('source_documents', [])]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno no servidor: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)