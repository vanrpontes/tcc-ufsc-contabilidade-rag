import streamlit as st
import requests

# 1. Configuração da página
st.set_page_config(
    page_title="Sispetro AI - TCC UFSC", 
    page_icon="🛡️", 
    layout="centered"
)

# Estilo CSS
st.markdown("""
    <style>
    .stTitle { font-weight: 800; color: #1c2e4a; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .sidebar-logo-container {
        display: flex; justify-content: center; align-items: center; width: 100%; margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Chatbot Sispetro")
st.markdown("""
**Projeto de Graduação em Ciências Contábeis da UFSC** *O Potencial da Inteligência Artificial pelo Método RAG para a Gestão da Informação Contábil*
""")
st.caption("Baseado em documentos técnicos, tutoriais e manuais operacionais do software Sispetro.")
st.divider()

# 3. Gerenciamento do Histórico com Saudação Inicial
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Olá! Sou o assistente inteligente do Sispetro. Estou pronto para tirar suas dúvidas técnicas e contábeis sobre o sistema. Como posso ajudar?"
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Campo de Entrada
if prompt := st.chat_input("Dúvida técnica ou contábil..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando manuais técnicos..."):
            try:
                # Nota: Use o protocolo HTTPS agora que você configurou o SSL!
                response = requests.post(
                    "https://chatbotsispetro.com.br/ask", 
                    json={"question": prompt},
                    timeout=30 
                )
                
                if response.status_code == 200:
                    answer = response.json()["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error("Erro na comunicação com o assistente.")
            except Exception as e:
                st.error(f"Erro de conexão: {e}")

# 5. Sidebar
with st.sidebar:
    # Centralização do Brasão usando colunas
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://identidade.ufsc.br/files/2017/10/brasao_UFSC_vertical_sigla_sombreado.png", width=80)
    
    st.title("Sobre o Projeto")
    st.info("Este chatbot utiliza RAG para consultar manuais técnicos do Sispetro.")
    st.markdown("---")
    
    st.markdown("""
        <div style="font-size: 0.9rem; line-height: 1.4;">
            <strong>Autor:</strong> Vanclércio da Rocha Pontes<br>
            <strong>Orientador:</strong> Prof. Dr. Valmir Emil Hoffmann
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    
    if st.button("Limpar Histórico", use_container_width=True):
        st.session_state.messages = [st.session_state.messages[0]] # Mantém apenas a saudação
        st.rerun()

# 6. Rodapé da Página Principal
st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0e1117;
        color: #999999; /* Um cinza um pouco mais claro para facilitar a leitura */
        text-align: center;
        font-size: 1rem; /* Aumentado de 0.75 para 0.9 */
        font-weight: 500;
        padding: 8px 0; /* Aumentado para dar mais destaque */
        z-index: 9999;
    }
    </style>
    <div class="footer">
        UFSC | Florianópolis — 2026
    </div>
""", unsafe_allow_html=True)