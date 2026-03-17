import streamlit as st
import requests

# 1. Configuração da página e Identidade Visual
st.set_page_config(
    page_title="Sispetro AI - TCC UFSC", 
    page_icon="🛡️", 
    layout="centered"
)

# Estilo CSS personalizado
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stTitle {
        font-weight: 800;
        color: #1c2e4a;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}

    /* AJUSTE DA LOGO NA BARRA LATERAL */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        padding-top: 1rem; 
    }
    .sidebar-logo-container {
        display: flex;
        justify-content: center; /* Centraliza horizontalmente */
        align-items: center;     /* Centraliza verticalmente */
        width: 100%;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Cabeçalho Estruturado (Foco no TCC)
st.title("🛡️ Chatbot Técnico Sispetro")
st.markdown("""
**Projeto de Graduação em Ciências Contábeis – UFSC** *O Potencial da Inteligência Artificial pelo Método RAG para a Gestão da Informação Contábil*
""")
st.caption("Baseado em documentos técnicos e manuais operacionais do software Sispetro.")
st.divider()

# 3. Gerenciamento do Histórico de Mensagens
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Campo de Entrada e Lógica da API
if prompt := st.chat_input("Como posso ajudar na sua consulta contábil/técnica hoje?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analisando base de conhecimento técnica..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/ask",
                    json={"question": prompt},
                    timeout=30 
                )
                
                if response.status_code == 200:
                    answer = response.json()["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"Erro na API: Código {response.status_code}")
            except Exception as e:
                st.error(f"Erro de conexão com o servidor Backend. Detalhe: {e}")

# 5. Barra Lateral (Sidebar) com Logo Centralizada e Reduzida
with st.sidebar:
    # Div de centralização total
    st.markdown('<div class="sidebar-logo-container">', unsafe_allow_html=True)
    st.image("https://identidade.ufsc.br/files/2017/10/brasao_UFSC_vertical_sigla_sombreado.png", width=60)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.title("Sobre o Projeto")
    st.info("""
    Este chatbot utiliza a arquitetura **RAG (Retrieval-Augmented Generation)** para consultar manuais técnicos e fornecer respostas precisas sobre o Sispetro.
    """)
    st.markdown("---")
    st.write("**Autor:** Vanclércio da Rocha Pontes")
    st.write("**E-mail:** vrpontes@outlook.com")
    
    if st.button("Limpar Histórico de Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("© 2026 - Tecnologia e Contabilidade")