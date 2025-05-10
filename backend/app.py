import streamlit as st
import requests
import urllib.parse
import time
import os
import json
from dotenv import load_dotenv
import traceback
import unidecode
import string
from functools import lru_cache
import xml.etree.ElementTree as ET
import csv
import io
import random

# Carregar variáveis de ambiente
load_dotenv()

# Configurações
MAX_RESULTS = 5
PUBMED_EMAIL = os.getenv("PUBMED_EMAIL", "liongabr@gmail.com")
PUBMED_API_KEY = os.getenv("PUBMED_API_KEY", "")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")  # Chave para Semantic Scholar API
HTTP_TIMEOUT = 10
MAX_ABSTRACT_LENGTH = 500
MAX_RETRIES = 5  # Número máximo de tentativas para APIs com limite de requisições

# Informações sobre obtenção de chaves de API
API_INFO = """
Para melhorar a confiabilidade das buscas, você pode configurar chaves de API:

- Semantic Scholar: Solicite em https://www.semanticscholar.org/product/api#api-key-form
- PubMed: Obtenha em https://ncbiinsights.ncbi.nlm.nih.gov/2017/11/02/new-api-keys-for-the-e-utilities/

Configure as chaves em um arquivo .env:
PUBMED_EMAIL=seu.email@exemplo.com
PUBMED_API_KEY=sua_chave_pubmed
SEMANTIC_SCHOLAR_API_KEY=sua_chave_semantic_scholar
"""

# Cache para resultados
cache = {}

# Inicializar session_state para armazenar histórico de artigos visualizados
if 'visualized_articles' not in st.session_state:
    st.session_state.visualized_articles = []

# Função para normalizar termos
@lru_cache(maxsize=2000)
def normalize_term(term: str, language: str = "pt-br") -> str:
    # Normalização simples: converter para minúsculas, remover acentos e pontuação
    normalized = unidecode.unidecode(term.lower())
    # Remover pontuação
    normalized = ''.join(c for c in normalized if c not in string.punctuation)
    # Remover espaços extras
    normalized = ' '.join(normalized.split())
    return normalized

# Função para registrar artigo visualizado
def registrar_artigo_visualizado(article):
    # Verificar se o artigo já está no histórico para não duplicar
    if article not in st.session_state.visualized_articles:
        # Limitar o histórico a 50 artigos para não sobrecarregar
        if len(st.session_state.visualized_articles) >= 50:
            st.session_state.visualized_articles.pop(0)  # Remove o mais antigo
        
        # Adicionar o artigo ao histórico
        st.session_state.visualized_articles.append(article)

# Função para buscar no PubMed
def search_pubmed(query, max_results=MAX_RESULTS):
    cache_key = f"pubmed_{query}_{max_results}"
    if cache_key in cache:
        return cache[cache_key]
    
    try:
        # Etapa 1: Busca IDs
        esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results * 2,
            "retmode": "json",
            "sort": "relevance",
            "tool": "scify",
            "email": PUBMED_EMAIL,
        }
        if PUBMED_API_KEY:
            search_params["api_key"] = PUBMED_API_KEY
        
        search_response = requests.get(esearch_url, params=search_params, timeout=HTTP_TIMEOUT)
        search_response.raise_for_status()
        search_data = search_response.json()
        ids = search_data.get("esearchresult", {}).get("idlist", [])
        
        if not ids:
            return []
        
        # Etapa 2: Buscar detalhes
        efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(ids[:max_results]),
            "rettype": "abstract",
            "retmode": "xml",
            "tool": "scify",
            "email": PUBMED_EMAIL
        }
        if PUBMED_API_KEY:
            fetch_params["api_key"] = PUBMED_API_KEY
        
        fetch_response = requests.get(efetch_url, params=fetch_params, timeout=HTTP_TIMEOUT)
        fetch_response.raise_for_status()
        xml_content = fetch_response.text
        
        # Análise do XML usando ElementTree
        results = []
        root = ET.fromstring(xml_content)
        articles = root.findall(".//PubmedArticle")
        
        for article in articles:
            try:
                pmid_elem = article.find(".//PMID")
                pmid = pmid_elem.text if pmid_elem is not None else "Unknown"
                
                title_elem = article.find(".//ArticleTitle")
                title = title_elem.text if title_elem is not None and title_elem.text else f"Artigo PubMed {pmid}"
                
                abstract_parts = []
                for abstract_elem in article.findall(".//AbstractText"):
                    if abstract_elem is not None and abstract_elem.text is not None:
                        abstract_parts.append(abstract_elem.text)
                abstract = " ".join(abstract_parts) if abstract_parts else "Resumo não disponível"
                
                if abstract and len(abstract) > MAX_ABSTRACT_LENGTH:
                    abstract = abstract[:MAX_ABSTRACT_LENGTH] + "..."
                
                authors = []
                for author in article.findall(".//Author"):
                    last_name = author.find("LastName")
                    first_name = author.find("Initials")
                    if last_name is not None and last_name.text:
                        author_name = last_name.text
                        if first_name is not None and first_name.text:
                            author_name += " " + first_name.text
                        authors.append(author_name)
                
                year_elem = article.find(".//PubDate/Year")
                year = year_elem.text if year_elem is not None else ""
                
                journal_elem = article.find(".//Journal/Title")
                journal = journal_elem.text if journal_elem is not None else ""
                
                authors_text = ", ".join(authors[:3])
                if len(authors) > 3:
                    authors_text += " et al."
                
                citation = ""
                if authors_text:
                    citation = authors_text
                if year:
                    citation += f" ({year})"
                if journal:
                    citation += f". {journal}"
                
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                
                results.append({
                    "title": title,
                    "url": url,
                    "id": pmid,
                    "abstract": abstract,
                    "authors": authors_text,
                    "year": year,
                    "journal": journal,
                    "citation": citation,
                    "source": "PubMed",
                    "apis_utilizadas": "E-Search (NCBI/PubMed), E-Summary (NCBI/PubMed), E-Link (NCBI/PubMed), E-Info (NCBI/PubMed)"
                })
            except Exception as e:
                st.error(f"Erro processando artigo: {str(e)}")
        
        cache[cache_key] = results
        return results
    
    except Exception as e:
        st.error(f"Erro na busca PubMed: {str(e)}")
        return []

# Função para buscar no Semantic Scholar
def search_semantic_scholar(query, max_results=MAX_RESULTS):
    cache_key = f"semantic_{query}_{max_results}"
    if cache_key in cache:
        return cache[cache_key]
    
    # Obter chave de API do Semantic Scholar, se disponível
    semantic_api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    
    try:
        # API do Semantic Scholar
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": max_results,
            "fields": "title,abstract,authors,year,venue,url,paperId"
        }
        
        headers = {}
        if semantic_api_key:
            headers["x-api-key"] = semantic_api_key
        
        # Estratégia de retry com backoff exponencial
        max_retries = 5
        retry_delay = 1  # Delay inicial em segundos
        attempt = 0
        
        while attempt < max_retries:
            try:
                response = requests.get(url, params=params, headers=headers, timeout=HTTP_TIMEOUT)
                
                # Se a resposta for bem-sucedida, processa normalmente
                if response.status_code == 200:
                    data = response.json()
                    break
                # Se houver erro de rate limit (429), tenta novamente após um delay
                elif response.status_code == 429:
                    attempt += 1
                    if attempt >= max_retries:
                        st.warning(f"Limite de requisições excedido na API do Semantic Scholar. Tente novamente mais tarde ou use uma chave de API.")
                        return []
                    
                    # Delay exponencial com jitter (para evitar sincronização de requisições)
                    jitter = random.uniform(0, 0.5)
                    sleep_time = (retry_delay * (2 ** (attempt - 1))) + jitter
                    
                    status_container = st.empty()
                    status_container.warning(f"Limite de requisições excedido. Aguardando {sleep_time:.1f}s antes de tentar novamente...")
                    time.sleep(sleep_time)
                    status_container.empty()
                    continue
                # Outros erros HTTP
                else:
                    response.raise_for_status()
            except requests.exceptions.RequestException as e:
                attempt += 1
                if attempt >= max_retries:
                    raise e
                time.sleep(retry_delay * (2 ** (attempt - 1)))
                continue
        
        # Processo de extração de dados permanece o mesmo
        results = []
        for paper in data.get("data", []):
            try:
                title = paper.get("title", "Título não disponível")
                abstract = paper.get("abstract", "Resumo não disponível")
                
                if abstract and len(abstract) > MAX_ABSTRACT_LENGTH:
                    abstract = abstract[:MAX_ABSTRACT_LENGTH] + "..."
                
                # Extrair autores
                authors = []
                for author in paper.get("authors", []):
                    if "name" in author:
                        authors.append(author["name"])
                
                authors_text = ", ".join(authors[:3])
                if len(authors) > 3:
                    authors_text += " et al."
                
                year = str(paper.get("year", ""))
                journal = paper.get("venue", "")
                
                # Formar citação
                citation = ""
                if authors_text:
                    citation = authors_text
                if year:
                    citation += f" ({year})"
                if journal:
                    citation += f". {journal}"
                
                # URL do artigo
                paper_id = paper.get("paperId", "")
                url = paper.get("url", f"https://www.semanticscholar.org/paper/{paper_id}")
                
                results.append({
                    "title": title,
                    "url": url,
                    "id": paper_id,
                    "abstract": abstract,
                    "authors": authors_text,
                    "year": year,
                    "journal": journal,
                    "citation": citation,
                    "source": "Semantic Scholar"
                })
            except Exception as e:
                st.error(f"Erro processando artigo do Semantic Scholar: {str(e)}")
        
        cache[cache_key] = results
        return results
    
    except Exception as e:
        st.error(f"Erro na busca Semantic Scholar: {str(e)}")
        return []

# Função para buscar no Europe PMC
def search_europe_pmc(query, max_results=MAX_RESULTS):
    cache_key = f"europepmc_{query}_{max_results}"
    if cache_key in cache:
        return cache[cache_key]
    
    try:
        # API do Europe PMC
        url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        params = {
            "query": query,
            "resultType": "core",
            "format": "json",
            "pageSize": max_results
        }
        
        response = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        
        results = []
        for result in data.get("resultList", {}).get("result", []):
            try:
                title = result.get("title", "Título não disponível")
                
                # Extrair resumo
                abstract = result.get("abstractText", "Resumo não disponível")
                
                if abstract and len(abstract) > MAX_ABSTRACT_LENGTH:
                    abstract = abstract[:MAX_ABSTRACT_LENGTH] + "..."
                
                # Extrair informações de autores
                authors_list = result.get("authorList", {}).get("author", [])
                authors = []
                for author in authors_list:
                    if "lastName" in author and "initials" in author:
                        authors.append(f"{author['lastName']} {author['initials']}")
                    elif "fullName" in author:
                        authors.append(author["fullName"])
                
                authors_text = ", ".join(authors[:3])
                if len(authors) > 3:
                    authors_text += " et al."
                
                # Extrair ano e revista
                year = result.get("pubYear", "")
                journal = result.get("journalTitle", "")
                
                # Formar citação
                citation = ""
                if authors_text:
                    citation = authors_text
                if year:
                    citation += f" ({year})"
                if journal:
                    citation += f". {journal}"
                
                # ID e URL
                paper_id = result.get("id", "")
                pmid = result.get("pmid", "")
                url = f"https://europepmc.org/article/MED/{pmid}" if pmid else f"https://europepmc.org/article/{paper_id}"
                
                results.append({
                    "title": title,
                    "url": url,
                    "id": paper_id,
                    "abstract": abstract,
                    "authors": authors_text,
                    "year": year,
                    "journal": journal,
                    "citation": citation,
                    "source": "Europe PMC"
                })
            except Exception as e:
                st.error(f"Erro processando artigo do Europe PMC: {str(e)}")
        
        cache[cache_key] = results
        return results
    
    except Exception as e:
        st.error(f"Erro na busca Europe PMC: {str(e)}")
        return []

# Função para buscar no CrossRef
def search_crossref(query, max_results=MAX_RESULTS):
    cache_key = f"crossref_{query}_{max_results}"
    if cache_key in cache:
        return cache[cache_key]
    
    try:
        # API do CrossRef
        url = "https://api.crossref.org/works"
        params = {
            "query": query,
            "rows": max_results,
            "sort": "relevance",
            "mailto": PUBMED_EMAIL  # É uma boa prática incluir um e-mail quando usar a API CrossRef
        }
        
        # Estratégia de retry com backoff exponencial
        max_retries = MAX_RETRIES
        retry_delay = 1  # Delay inicial em segundos
        attempt = 0
        
        while attempt < max_retries:
            try:
                response = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
                
                # Se a resposta for bem-sucedida, processa normalmente
                if response.status_code == 200:
                    data = response.json()
                    break
                # Se houver erro de rate limit (429), tenta novamente após um delay
                elif response.status_code == 429:
                    attempt += 1
                    if attempt >= max_retries:
                        st.warning("Limite de requisições excedido na API do CrossRef. Tente novamente mais tarde.")
                        return []
                    
                    # Delay exponencial com jitter (para evitar sincronização de requisições)
                    jitter = random.uniform(0, 0.5)
                    sleep_time = (retry_delay * (2 ** (attempt - 1))) + jitter
                    
                    status_container = st.empty()
                    status_container.warning(f"Limite de requisições excedido no CrossRef. Aguardando {sleep_time:.1f}s antes de tentar novamente...")
                    time.sleep(sleep_time)
                    status_container.empty()
                    continue
                # Outros erros HTTP
                else:
                    response.raise_for_status()
            except requests.exceptions.RequestException as e:
                attempt += 1
                if attempt >= max_retries:
                    raise e
                time.sleep(retry_delay * (2 ** (attempt - 1)))
                continue
        
        results = []
        for item in data.get("message", {}).get("items", []):
            try:
                title = " ".join(item.get("title", ["Título não disponível"]))
                
                # CrossRef não fornece resumo diretamente
                abstract = "Resumo não disponível via CrossRef"
                
                # Extrair autores
                authors = []
                for author in item.get("author", []):
                    if "family" in author and "given" in author:
                        authors.append(f"{author['family']} {author['given'][0]}")
                    elif "name" in author:
                        authors.append(author["name"])
                
                authors_text = ", ".join(authors[:3])
                if len(authors) > 3:
                    authors_text += " et al."
                
                # Extrair ano e revista
                year = ""
                if "issued" in item and "date-parts" in item["issued"] and item["issued"]["date-parts"]:
                    year_parts = item["issued"]["date-parts"][0]
                    if year_parts and len(year_parts) > 0:
                        year = str(year_parts[0])
                
                journal = ""
                if "container-title" in item and item["container-title"]:
                    journal = item["container-title"][0]
                
                # Formar citação
                citation = ""
                if authors_text:
                    citation = authors_text
                if year:
                    citation += f" ({year})"
                if journal:
                    citation += f". {journal}"
                
                # URL e DOI
                doi = item.get("DOI", "")
                url = f"https://doi.org/{doi}" if doi else ""
                
                results.append({
                    "title": title,
                    "url": url,
                    "id": doi,
                    "abstract": abstract,
                    "authors": authors_text,
                    "year": year,
                    "journal": journal,
                    "citation": citation,
                    "source": "CrossRef"
                })
            except Exception as e:
                st.error(f"Erro processando artigo do CrossRef: {str(e)}")
        
        cache[cache_key] = results
        return results
    
    except Exception as e:
        st.error(f"Erro na busca CrossRef: {str(e)}")
        return []

# Função para gerar arquivo CSV com os resultados
def gerar_csv(results):
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Escrever cabeçalho
    writer.writerow(["Título", "Autores", "Ano", "Revista", "Resumo", "URL", "ID", "Fonte"])
    
    # Escrever dados
    for result in results:
        writer.writerow([
            result.get("title", ""),
            result.get("authors", ""),
            result.get("year", ""),
            result.get("journal", ""),
            result.get("abstract", ""),
            result.get("url", ""),
            result.get("id", ""),
            result.get("source", "")
        ])
    
    return output.getvalue()

def gerar_mensagem_interativa():
    mensagens = [
        "Consultando bases de artigos científicos...",
        "Buscando resultados mais relevantes...",
        "Analisando estudos recentes sobre o tema...",
        "Comparando resultados entre diferentes fontes...",
        "Processando metadados dos artigos...",
        "Filtrando resultados mais significativos...",
        "Organizando resultados para melhor visualização...",
        "Conectando-se com repositórios acadêmicos...",
        "Reunindo evidências científicas sobre o assunto..."
    ]
    return random.choice(mensagens)

def main():
    st.title("Scify - Pesquise menos, aprenda mais.")
    
    # Configurações fixas (sem menu lateral)
    max_results = 5  # Número fixo de resultados por fonte
    traduzir = True  # Tradução ativada por padrão
    
    # Entrada principal com design simplificado
    st.markdown("<h3 style='margin-bottom: 20px;'>Digite sua pergunta ou termo de pesquisa:</h3>", unsafe_allow_html=True)
    
    # CSS para customizar o input
    st.markdown("""
    <style>
    /* Estilos gerais para elementos de input */
    .stTextArea textarea {
        height: 120px !important;
        font-size: 20px !important;
        line-height: 1.4 !important;
        border-radius: 10px !important;
        border: 2px solid #4169E1 !important;
        padding: 15px !important;
        background-color: #fcfcfc !important;
    }
    
    /* Remove o redimensionamento da área de texto */
    .stTextArea div[data-baseweb="textarea"] textarea {
        resize: none !important;
    }
    
    /* Estilo para o botão de pesquisa */
    .stButton button {
        font-size: 18px !important;
        font-weight: bold !important;
        height: 50px !important;
    }
    
    /* Estilo para o tooltip */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }

    .tooltip .tooltiptext {
        visibility: hidden;
        width: 300px;
        background-color: #555;
        color: #fff;
        text-align: left;
        border-radius: 6px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -150px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 14px;
    }

    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Usar área de texto em vez de input comum para maior controle
    user_input = st.text_area(
        label="", 
        value="hipertrofia", 
        height=120,
        max_chars=200,
        label_visibility="collapsed",
        key="search_input",
        help="Digite aqui sua pergunta ou termo de pesquisa",
        placeholder="Ex: hipertrofia, tratamento para diabetes, etc."
    )
    
    # Explicação simples
    st.caption("A busca será realizada em múltiplas fontes científicas: PubMed, Semantic Scholar, Europe PMC e CrossRef")
    
    col1, col2 = st.columns([4, 1])
    with col2:
        search_button = st.button("Pesquisar", type="primary", use_container_width=True)
    
    if search_button:
        try:
            progress_container = st.empty()
            status_container = st.empty()
            progress_bar = progress_container.progress(0)
            status_text = status_container.text(gerar_mensagem_interativa())
            
            # Normalizar a entrada
            normalized_query = normalize_term(user_input)
            
            # Realizar a pesquisa em todas as fontes disponíveis
            all_results = []
            
            # Atualizar o progresso para 10%
            progress_bar.progress(10)
            status_container.text(gerar_mensagem_interativa())
            time.sleep(0.5)  # Pequena pausa para mostrar progresso
            
            # PubMed
            status_container.text("Consultando PubMed (National Library of Medicine)...")
            pubmed_results = search_pubmed(normalized_query, max_results)
            all_results.extend(pubmed_results)
            progress_bar.progress(30)
            status_container.text(gerar_mensagem_interativa())
            time.sleep(0.5)  # Pequena pausa para mostrar progresso
            
            # Semantic Scholar
            status_container.text("Consultando Semantic Scholar (Allen Institute for AI)...")
            semantic_results = search_semantic_scholar(normalized_query, max_results)
            all_results.extend(semantic_results)
            progress_bar.progress(50)
            status_container.text(gerar_mensagem_interativa())
            time.sleep(0.5)  # Pequena pausa para mostrar progresso
            
            # Europe PMC
            status_container.text("Consultando Europe PMC (European Bioinformatics Institute)...")
            europepmc_results = search_europe_pmc(normalized_query, max_results)
            all_results.extend(europepmc_results)
            progress_bar.progress(70)
            status_container.text(gerar_mensagem_interativa())
            time.sleep(0.5)  # Pequena pausa para mostrar progresso
            
            # CrossRef
            status_container.text("Consultando CrossRef (Registro de DOIs acadêmicos)...")
            crossref_results = search_crossref(normalized_query, max_results)
            all_results.extend(crossref_results)
            progress_bar.progress(85)
            
            # Finalizar progresso
            progress_bar.progress(100)
            status_container.text("Pesquisa concluída! Exibindo resultados mais relevantes.")
            time.sleep(1)
            progress_container.empty()
            status_container.empty()
            
            # Mostrar resultados
            total_resultados = len(all_results)
            if total_resultados > 0:
                st.write(f"### Encontramos {total_resultados} resultados relevantes para sua pesquisa")
                
                # Verificar se há histórico de artigos visualizados para exibir
                if len(st.session_state.visualized_articles) > 0:
                    with st.expander("📚 Histórico de artigos visualizados"):
                        for i, article in enumerate(reversed(st.session_state.visualized_articles), 1):
                            st.markdown(f"**{i}.** [{article['title']}]({article['url']}) - {article['source']}")
                
                # Adicionar filtro por fonte após a busca
                sources = sorted(list(set([r["source"] for r in all_results])))
                selected_sources = st.multiselect("Filtrar por fonte:", sources, default=sources)
                
                # Filtrar resultados por fonte selecionada
                filtered_results = [r for r in all_results if r["source"] in selected_sources]
                
                # Mostrar resultados por fonte
                for source in selected_sources:
                    source_results = [r for r in filtered_results if r["source"] == source]
                    if source_results:
                        st.subheader(f"{source} ({len(source_results)} resultados)")
                        
                        for i, result in enumerate(source_results, 1):
                            with st.expander(f"{i}. {result['title']}"):
                                # Registrar que este artigo foi visualizado quando o usuário abre o expander
                                registrar_artigo_visualizado(result)
                                
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.markdown(f"**Fonte:** {result['source']}")
                                    st.markdown(f"**Autores:** {result['authors']}")
                                with col2:
                                    if 'year' in result and result['year']:
                                        st.markdown(f"**Ano:** {result['year']}")
                                    if 'journal' in result and result['journal']:
                                        st.markdown(f"**Revista:** {result['journal']}")
                                
                                # Traduzir resumo se a opção estiver ativada
                                abstract = result["abstract"]
                                st.markdown(f"**Resumo:** {abstract}")
                                
                                st.markdown(f"**Link:** [{result['url']}]({result['url']})")
                
                # Opção para download dos resultados em CSV
                if st.button("Baixar resultados em CSV"):
                    csv_data = gerar_csv(filtered_results)
                    st.download_button(
                        label="Clique para baixar",
                        data=csv_data,
                        file_name=f"resultados_{normalized_query}.csv",
                        mime="text/csv"
                    )
            else:
                st.warning("Nenhum resultado encontrado. Tente ajustar os termos de pesquisa.")
        
        except Exception as e:
            st.error(f"Ocorreu um erro durante a pesquisa: {str(e)}")
            st.error(traceback.format_exc())

if __name__ == "__main__":
    main()