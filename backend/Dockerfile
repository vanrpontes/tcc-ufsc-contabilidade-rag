# Usa uma imagem leve do Python
FROM python:3.10-slim

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia os arquivos de requisitos e instala
# Certifique-se de ter o requirements.txt dentro da pasta backend/
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o conteúdo da pasta backend para o container
COPY . .

# Expõe a porta que o FastAPI usa
EXPOSE 8000

# Comando para rodar a API (ajustado para o seu main.py dentro de app/)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]