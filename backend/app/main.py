from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.chat import ask_sispetro # Importa a lógica que já funciona

app = FastAPI(title="API do Consultor Sispetro")

# Definimos o formato do dado que a API vai receber
class Pergunta(BaseModel):
    texto: str

@app.get("/")
def home():
    return {"status": "Servidor Sispetro Ativo"}

@app.post("/perguntar")
def realizar_pergunta(pergunta: Pergunta):
    try:
        # Chamamos a função que você já testou e aprovou
        resposta = ask_sispetro(pergunta.texto)
        return {"resposta": resposta['result']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))