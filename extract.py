import requests
import os
import time
from bs4 import BeautifulSoup
import urllib3
from dotenv import load_dotenv

# Desativa avisos de SSL (comum em ambientes corporativos/Atlassian)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- CONFIGURAÇÕES ---
BASE_URL = "https://futura.atlassian.net/wiki"
SPACE_KEY = "SPT"
# Ajustado para a nova estrutura modular
OUTPUT_FOLDER = os.path.join("backend", "data")

def clean_html(html_content):
    if not html_content:
        return ""
    soup = BeautifulSoup(html_content, 'html.parser')
    # Remove scripts e estilos para limpar o ruído
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()
    return soup.get_text(separator='\n')

def fetch_public_pages():
    # Garante que a pasta backend/data exista
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # Tenta carregar credenciais do .env se existirem
    load_dotenv()
    EMAIL = os.getenv('CONFLUENCE_EMAIL')
    TOKEN = os.getenv('CONFLUENCE_API_TOKEN')
    auth = (EMAIL, TOKEN) if EMAIL and TOKEN else None

    api_url = f"{BASE_URL}/rest/api/content"
    params = {
        'spaceKey': SPACE_KEY,
        'limit': 100,
        'expand': 'body.storage',
        'start': 0
    }

    count = 0
    print(f"🚀 Iniciando captura do espaço: {SPACE_KEY}...")
    print(f"📂 Destino: {os.path.abspath(OUTPUT_FOLDER)}")

    while True:
        try:
            # Tenta com autenticação, se falhar ou não houver, tenta público
            response = requests.get(api_url, params=params, auth=auth, verify=False)
            
            if response.status_code != 200:
                print(f"⚠️ Status {response.status_code}. Tentando acesso sem autenticação...")
                response = requests.get(api_url, params=params, verify=False)
            
            response.raise_for_status()
            data = response.json()
            results = data.get('results', [])
            
        except Exception as e:
            print(f"❌ Falha na conexão: {e}")
            break

        if not results:
            print("🏁 Nenhuma página adicional encontrada.")
            break

        for page in results:
            count += 1
            title = page['title']
            body = page.get('body', {}).get('storage', {}).get('value', '')
            
            clean_text = clean_html(body)
            # Limpa caracteres inválidos para nomes de arquivos no Windows/Linux
            safe_title = "".join(x for x in title if x.isalnum() or x in " -_").strip()
            
            # Define o caminho do arquivo .txt
            file_path = os.path.join(OUTPUT_FOLDER, f"{safe_title}.txt")
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"TITULO: {title}\n")
                f.write(f"URL: {BASE_URL}/spaces/{SPACE_KEY}/pages/{page['id']}\n")
                f.write("-" * 50 + "\n")
                f.write(clean_text)
            
            print(f"[{count}] Extraído: {title}")

        # Paginação: avança para o próximo bloco de 100 páginas
        params['start'] += params['limit']
        
        if len(results) < params['limit']:
            break
            
        time.sleep(0.2) # Evita sobrecarga no servidor (Rate Limit)

    print(f"\n✅ Concluído! {count} arquivos salvos em '{OUTPUT_FOLDER}'.")

if __name__ == "__main__":
    fetch_public_pages()