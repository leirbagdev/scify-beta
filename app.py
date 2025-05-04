import streamlit as st
from graph import graph
from schemas import ReportState
import time
import traceback

st.title("Sci-Fi v2 - Pesquisa Inteligente")

st.sidebar.header("Configurações")
search_source = st.sidebar.radio(
    "Fonte de pesquisa:",
    ["PubMed (Artigos Científicos)", "Tavily (Web geral)"],
    index=0
)

st.sidebar.info("✨ As buscas ocorrem em paralelo para maior eficiência!")

user_input = st.text_input("Digite sua pergunta:", "Make a research for me about building an LLM from scratch")
if st.button("Pesquisar"):
    try:
        st.write("Gerando resposta...")
        status = st.empty()
        
        # Criar log para acompanhar o progresso
        status.text("Iniciando processamento...")
        
        # Inicializar o estado com a fonte de pesquisa selecionada
        state = ReportState(
            user_input=user_input, 
            source=search_source.split()[0].lower()  # Extrai "pubmed" ou "tavily"
        )
        status.text(f"Realizando buscas paralelas em {search_source}...")
        
        # Tenta invocar o grafo
        result = None
        try:
            result = graph.invoke(state)
            status.text("Processamento concluído!")
        except Exception as e:
            st.error(f"Erro durante o processamento: {str(e)}")
            st.code(traceback.format_exc())
            # Cria uma resposta padrão
            result = {"final_response": f"Desculpe, ocorreu um erro ao processar sua solicitação: {str(e)}"}
        
        # Mostra o resultado
        st.write("### Resposta Final")
        if result and "final_response" in result:
            st.write(result["final_response"])
        else:
            st.write("Não foi possível obter uma resposta. Por favor, tente novamente.")
            
    except Exception as e:
        st.error(f"Erro geral: {str(e)}")
        st.code(traceback.format_exc())