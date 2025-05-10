from langgraph.graph import StateGraph
from langgraph.constants import START, END
from schemas import ReportState, SearchResult
from prompts import planner_prompt, search_summary_prompt, reviewer_prompt
from langchain_ollama import ChatOllama
import requests
import urllib.parse
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple, TypedDict, Optional, Union, cast
import os
import json
import re
import xml.etree.ElementTree as ET
import time
import concurrent.futures
import functools
import asyncio
import aiohttp
import unidecode
import string
from collections import defaultdict
from functools import lru_cache

# Carregar variáveis de ambiente
load_dotenv()

# Configuração das LLMs
model_name = "mistral:7b-instruct-v0.2-q4_0"
planner_llm = ChatOllama(model=model_name, base_url=os.getenv("OLLAMA_HOST"), temperature=0.1)
summary_llm = ChatOllama(model=model_name, base_url=os.getenv("OLLAMA_HOST"), temperature=0.1)
writer_llm = ChatOllama(model=model_name, base_url=os.getenv("OLLAMA_HOST"), temperature=0.0)

# Configurações para Ollama
CPU_THREADS = min(os.cpu_count() or 4, 8)
MAX_WORKERS = min(CPU_THREADS - 1, 4)

# Detalhes para APIs
PUBMED_EMAIL = os.getenv("PUBMED_EMAIL", "liongabr@gmail.com")
PUBMED_API_KEY = os.getenv("PUBMED_API_KEY", "")
PUBMED_RATE_LIMIT = 0.34
CROSSREF_EMAIL = os.getenv("CROSSREF_EMAIL", "liongabr@gmail.com")

# Timeouts
PLANNER_TIMEOUT = 3
SEARCH_TIMEOUT = 7
SUMMARY_TIMEOUT = 2
REVIEW_TIMEOUT = 15
HTTP_TIMEOUT = 5

# Configurações de performance
MAX_RESULTS_DEFAULT = 5
MAX_ABSTRACT_LENGTH = 500
CACHE_ENABLED = True
SKIP_SUMMARY = False

# Caches
pubmed_cache = {}
crossref_cache = {}
arxiv_cache = {}
semantic_cache = {}
api_response_times = {"pubmed": [], "crossref": [], "arxiv": [], "semantic": []}

# Idiomas suportados
SUPPORTED_LANGUAGES = ["pt-br", "en-us", "es"]

# Função para normalizar termos com cache - Simplificada para remover dependência do SpaCy
@lru_cache(maxsize=2000)
def normalize_term(term: str, language: str = "pt-br") -> str:
    if language not in SUPPORTED_LANGUAGES:
        language = "pt-br"
    # Normalização simples: converter para minúsculas, remover acentos e pontuação
    normalized = unidecode.unidecode(term.lower())
    # Remover pontuação
    normalized = ''.join(c for c in normalized if c not in string.punctuation)
    # Remover espaços extras
    normalized = ' '.join(normalized.split())
    return normalized

# Dicionário hierárquico
SCIENTIFIC_TERMS = {
    "concepts": {
        "hipertrofia": {
            "variants": [
                "hipertrofias", "hipertrofia muscular", "hipertrofya",
                "hypertrophy", "hypertrophies", "muscle growth",
                "hipertrofia", "crecimiento muscular", "hipertrofía"
            ],
            "translations": {
                "pt-br": ["hipertrofia"],
                "en-us": ["hypertrophy", "muscle growth"],
                "es": ["hipertrofia", "crecimiento muscular"]
            },
            "related_terms": [
                {
                    "term": "aumento da área de secção transversa",
                    "translations": {
                        "pt-br": ["aumento da área de secção transversa"],
                        "en-us": ["cross-sectional area", "CSA"],
                        "es": ["aumento del área de sección transversal"]
                    },
                    "weight": 0.9,
                    "context": ["sports science", "physiology"]
                },
                {
                    "term": "crescimento muscular",
                    "translations": {
                        "pt-br": ["crescimento muscular"],
                        "en-us": ["muscle growth"],
                        "es": ["crecimiento muscular"]
                    },
                    "weight": 0.8,
                    "context": ["sports science", "physiology"]
                },
                {
                    "term": "dano muscular",
                    "translations": {
                        "pt-br": ["dano muscular"],
                        "en-us": ["muscle damage"],
                        "es": ["daño muscular"]
                    },
                    "weight": 0.7,
                    "context": ["physiology", "exercise physiology"]
                },
                {
                    "term": "célula-satélite",
                    "translations": {
                        "pt-br": ["célula-satélite"],
                        "en-us": ["satellite cell"],
                        "es": ["célula satélite"]
                    },
                    "weight": 0.6,
                    "context": ["cell biology", "physiology"]
                },
                {
                    "term": "macrófagos",
                    "translations": {
                        "pt-br": ["macrófagos"],
                        "en-us": ["macrophages"],
                        "es": ["macrófagos"]
                    },
                    "weight": 0.5,
                    "context": ["immunology", "cell biology"]
                }
            ],
            "context": ["bodybuilding", "sports science", "exercise physiology"],
            "sources": {
                "en-us": ["PubMed", "Google Scholar"],
                "es": ["Scielo", "Redalyc"],
                "pt-br": ["Scielo", "BVS"]
            }
        },
        "diabetes": {
            "variants": [
                "diabete", "diabetes mellitus", "diabeticos", "diabetis",
                "diabetes", "diabetes mellitus", "diabetic",
                "diabetes", "diabetes mellitus", "diabético"
            ],
            "translations": {
                "pt-br": ["diabetes"],
                "en-us": ["diabetes", "diabetes mellitus"],
                "es": ["diabetes", "diabetes mellitus"]
            },
            "related_terms": [
                {
                    "term": "hiperglicemia",
                    "translations": {
                        "pt-br": ["hiperglicemia"],
                        "en-us": ["hyperglycemia"],
                        "es": ["hiperglucemia"]
                    },
                    "weight": 0.9,
                    "context": ["endocrinology", "metabolism"]
                },
                {
                    "term": "resistência à insulina",
                    "translations": {
                        "pt-br": ["resistência à insulina"],
                        "en-us": ["insulin resistance"],
                        "es": ["resistencia a la insulina"]
                    },
                    "weight": 0.8,
                    "context": ["endocrinology", "metabolism"]
                },
                {
                    "term": "células beta",
                    "translations": {
                        "pt-br": ["células beta"],
                        "en-us": ["beta cells"],
                        "es": ["células beta"]
                    },
                    "weight": 0.7,
                    "context": ["cell biology", "endocrinology"]
                },
                {
                    "term": "inflamação crônica",
                    "translations": {
                        "pt-br": ["inflamação crônica"],
                        "en-us": ["chronic inflammation"],
                        "es": ["inflamación crónica"]
                    },
                    "weight": 0.6,
                    "context": ["immunology", "metabolism"]
                }
            ],
            "context": ["endocrinology", "clinical medicine", "public health"],
            "sources": {
                "en-us": ["PubMed", "Google Scholar"],
                "es": ["Scielo", "Redalyc"],
                "pt-br": ["Scielo", "BVS"]
            }
        },
        "obesidade": {
            "variants": [
                "obesidades", "obeso", "obesitdade",
                "obesity", "obese",
                "obesidad", "obeso"
            ],
            "translations": {
                "pt-br": ["obesidade"],
                "en-us": ["obesity"],
                "es": ["obesidad"]
            },
            "related_terms": [
                {
                    "term": "acúmulo de gordura",
                    "translations": {
                        "pt-br": ["acúmulo de gordura"],
                        "en-us": ["fat accumulation"],
                        "es": ["acumulación de grasa"]
                    },
                    "weight": 0.9,
                    "context": ["nutrition", "metabolism"]
                },
                {
                    "term": "tecido adiposo",
                    "translations": {
                        "pt-br": ["tecido adiposo"],
                        "en-us": ["adipose tissue"],
                        "es": ["tejido adiposo"]
                    },
                    "weight": 0.8,
                    "context": ["physiology", "metabolism"]
                },
                {
                    "term": "adipócitos",
                    "translations": {
                        "pt-br": ["adipócitos"],
                        "en-us": ["adipocytes"],
                        "es": ["adipocitos"]
                    },
                    "weight": 0.7,
                    "context": ["cell biology", "metabolism"]
                },
                {
                    "term": "inflamação metabólica",
                    "translations": {
                        "pt-br": ["inflamação metabólica"],
                        "en-us": ["metabolic inflammation"],
                        "es": ["inflamación metabólica"]
                    },
                    "weight": 0.6,
                    "context": ["immunology", "metabolism"]
                }
            ],
            "context": ["nutrition", "endocrinology", "public health"],
            "sources": {
                "en-us": ["PubMed", "Google Scholar"],
                "es": ["Scielo", "Redalyc"],
                "pt-br": ["Scielo", "BVS"]
            }
        }
    }
}

# Índice para buscas rápidas
TERM_INDEX = defaultdict(list)
for category, terms in SCIENTIFIC_TERMS.items():
    for term, data in terms.items():
        for lang in SUPPORTED_LANGUAGES:
            norm_term = normalize_term(term, language=lang)
            TERM_INDEX[f"{norm_term}_{lang}"].append((category, term, lang, 0))
            for variant in data["variants"]:
                norm_variant = normalize_term(variant, language=lang)
                TERM_INDEX[f"{norm_variant}_{lang}"].append((category, term, lang, 0))
            for level, related in enumerate(data["related_terms"], 1):
                norm_related = normalize_term(related["term"], language=lang)
                TERM_INDEX[f"{norm_related}_{lang}"].append((category, term, lang, level))
                for related_lang, translations in related["translations"].items():
                    if related_lang in SUPPORTED_LANGUAGES:
                        for trans in translations:
                            norm_trans = normalize_term(trans, language=related_lang)
                            TERM_INDEX[f"{norm_trans}_{related_lang}"].append((category, term, related_lang, level))

# Função para corrigir erros de digitação - Simplificada
def correct_typo(term: str, valid_terms: list, language: str, threshold: int = 80) -> Tuple[str, float, int]:
    if language not in SUPPORTED_LANGUAGES:
        language = "pt-br"
    
    # Filtrando termos do mesmo idioma
    lang_terms = [t for t in valid_terms if t.endswith(f"_{language}")]
    if not lang_terms:
        return term, 1.0, 0
    
    # Como não temos fuzzywuzzy, simplesmente retornamos o termo original
    # ou a primeira correspondência exata se encontrada
    for valid_term in lang_terms:
        if valid_term.startswith(f"{term}_"):
            corrected = valid_term.rsplit("_", 1)[0]
            matches = TERM_INDEX[valid_term]
            level = min(m[3] for m in matches)
            return corrected, 1.0, level
    
    return term, 1.0, 0

# Função para construir query otimizada
def build_query(term: str, input_language: str = "pt-br", target_language: str = "en-us", max_depth: int = 3, max_terms: int = 5) -> Dict[str, Any]:
    if input_language not in SUPPORTED_LANGUAGES:
        input_language = "pt-br"
    if target_language not in SUPPORTED_LANGUAGES:
        target_language = "en-us"

    normalized_term = normalize_term(term, language=input_language)
    # Como não temos mais a correção de typo sofisticada, usamos apenas o termo normalizado
    corrected_term, confidence, level = term, 1.0, 0

    # Montar tradução simples
    if input_language != "en-us" and target_language == "en-us":
        # Caso especial: tradução simples para inglês de alguns termos conhecidos
        translations = {
            "hipertrofia": "hypertrophy",
            "diabetes": "diabetes",
            "obesidade": "obesity"
        }
        query = translations.get(term.lower(), term)
    else:
        query = term
        
    return {
        "query": query,
        "sources": [],
        "context": [],
        "output_language": input_language,
        "confidence": confidence,
        "level": level
    }

# Decorator para timeout
def with_timeout(timeout_seconds, default_return=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = default_return
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    result = future.result(timeout=timeout_seconds)
                except concurrent.futures.TimeoutError:
                    print(f"⏱️ Timeout de {timeout_seconds} segundos atingido para {func.__name__}")
                    return default_return
            return result
        return wrapper
    return decorator

# Função otimizada para buscar no PubMed
def search_pubmed(query: str, max_results: int = MAX_RESULTS_DEFAULT, input_language: str = "pt-br") -> List[Dict[str, str]]:
    if input_language not in SUPPORTED_LANGUAGES:
        input_language = "pt-br"
    
    cache_key = f"{query}_{max_results}_{input_language}"
    if CACHE_ENABLED and cache_key in pubmed_cache:
        print(f"🔍 PubMed: Usando resultados em cache para '{query}'")
        return pubmed_cache[cache_key]
    
    try:
        print(f"🔍 PubMed: Pesquisando: '{query}'")
        
        # Construir query otimizada
        query_data = build_query(query, input_language=input_language, target_language="en-us", max_depth=3)
        enhanced_query = query_data["query"]
        print(f"🔍 PubMed: Busca aprimorada: '{enhanced_query}'")
        
        # Etapa 1: Busca IDs
        esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": enhanced_query,
            "retmax": max_results * 3,
            "retmode": "json",
            "sort": "relevance",
            "tool": "scify",
            "email": PUBMED_EMAIL,
            "datetype": "pdat",
            "reldate": "3650"
        }
        if PUBMED_API_KEY:
            search_params["api_key"] = PUBMED_API_KEY
        
        search_response = requests.get(esearch_url, params=search_params, timeout=HTTP_TIMEOUT)
        search_response.raise_for_status()
        search_data = search_response.json()
        ids = search_data.get("esearchresult", {}).get("idlist", [])
        
        # Fallback: Tentar query original
        if not ids:
            print("🔍 PubMed: Nenhum resultado com query expandida. Tentando query original...")
            query_data = build_query(query, input_language=input_language, target_language="en-us", max_depth=0)
            search_params["term"] = query_data["query"]
            search_response = requests.get(esearch_url, params=search_params, timeout=HTTP_TIMEOUT)
            search_response.raise_for_status()
            search_data = search_response.json()
            ids = search_data.get("esearchresult", {}).get("idlist", [])
        
        # Fallback: Tentar termo simplificado
        if not ids:
            print("🔍 PubMed: Nenhum resultado com query original. Tentando termo simplificado...")
            simplified_term = query.split()[0] if " " in query else query
            search_params["term"] = simplified_term
            search_response = requests.get(esearch_url, params=search_params, timeout=HTTP_TIMEOUT)
            search_response.raise_for_status()
            search_data = search_response.json()
            ids = search_data.get("esearchresult", {}).get("idlist", [])
        
        if not ids:
            print("❌ PubMed: Nenhum resultado encontrado.")
            suggestions = []
            for concept in SCIENTIFIC_TERMS["concepts"].values():
                for trans in concept["translations"].get(input_language, []):
                    if trans.lower() != query.lower():
                        suggestions.append(trans)
            suggestion_text = f"Sugestões: {', '.join(suggestions[:3])}" if suggestions else "Tente termos mais específicos ou em inglês."
            return [{
                "title": f"Nenhum resultado encontrado para: {query}",
                "url": f"https://pubmed.ncbi.nlm.nih.gov/?term={urllib.parse.quote(query)}",
                "content": f"Não foram encontrados artigos no PubMed relacionados a '{query}'. {suggestion_text}",
                "citation": "PubMed - Sem resultados"
            }]
        
        print(f"✅ PubMed: Encontrados {len(ids)} artigos")
        ids = ids[:max_results]
        
        # Etapa 2: Buscar detalhes
        efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(ids),
            "rettype": "abstract",
            "retmode": "xml",
            "tool": "scify",
            "email": PUBMED_EMAIL
        }
        if PUBMED_API_KEY:
            fetch_params["api_key"] = PUBMED_API_KEY
        
        time.sleep(PUBMED_RATE_LIMIT)
        fetch_response = requests.get(efetch_url, params=fetch_params, timeout=HTTP_TIMEOUT)
        fetch_response.raise_for_status()
        xml_content = fetch_response.text
        
        results = []
        root = ET.fromstring(xml_content)
        articles = root.findall(".//PubmedArticle")
        
        for article in articles:
            try:
                pmid_elem = article.find(".//PMID")
                pmid = pmid_elem.text if pmid_elem is not None else "Unknown"
                title_elem = article.find(".//ArticleTitle")
                title = title_elem.text if title_elem is not None else f"Artigo PubMed {pmid}"
                title = title or f"Artigo PubMed {pmid}"
                
                abstract_parts = []
                for abstract_elem in article.findall(".//AbstractText"):
                    if abstract_elem is not None and abstract_elem.text is not None:
                        abstract_parts.append(abstract_elem.text)
                abstract = " ".join(abstract_parts) if abstract_parts else ""
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
                if not citation:
                    citation = "Fonte: PubMed"
                
                content = f"Título: {title}\n\nAutores: {authors_text}\n\n"
                if journal and year:
                    content += f"Publicado em: {journal}, {year}\n\n"
                content += f"Resumo: {abstract if abstract else 'Resumo não disponível.'}"
                
                results.append({
                    "title": title[:100] + "..." if title and len(title) > 100 else title,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "content": content,
                    "citation": citation
                })
                print(f"📄 PubMed: Processado artigo {pmid}")
            
            except Exception as e:
                print(f"⚠️ Erro processando artigo: {e}")
        
        if CACHE_ENABLED and results:
            pubmed_cache[cache_key] = results
        return results
    
    except Exception as e:
        print(f"❌ Erro na busca PubMed: {e}")
        return [{
            "title": "Erro na busca PubMed",
            "url": "",
            "content": f"Ocorreu um erro: {str(e)}",
            "citation": "Erro"
        }]

@with_timeout(SEARCH_TIMEOUT, default_return=[{"title": "Tempo limite excedido", "url": "", "content": "A busca no PubMed excedeu o tempo limite."}])
def search_pubmed_with_timeout(query: str, max_results: int = MAX_RESULTS_DEFAULT, input_language: str = "pt-br") -> List[Dict[str, str]]:
    return search_pubmed(query, max_results, input_language)

# Função para buscar no Crossref
def search_crossref(query: str, max_results: int = MAX_RESULTS_DEFAULT, input_language: str = "pt-br") -> List[Dict[str, str]]:
    if input_language not in SUPPORTED_LANGUAGES:
        input_language = "pt-br"
    
    cache_key = f"{query}_{max_results}_{input_language}"
    if CACHE_ENABLED and cache_key in crossref_cache:
        print(f"🔍 Crossref: Usando resultados em cache para '{query}'")
        return crossref_cache[cache_key]
    
    try:
        print(f"🔍 Crossref: Pesquisando: '{query}'")
        
        query_data = build_query(query, input_language=input_language, target_language="en-us", max_depth=3)
        enhanced_query = query_data["query"]
        print(f"🔍 Crossref: Busca aprimorada: '{enhanced_query}'")
        
        search_params = {
            "query": enhanced_query,
            "rows": max_results * 2,
            "sort": "relevance",
            "order": "desc",
            "mailto": CROSSREF_EMAIL,
            "filter": "has-abstract:true,has-full-text:true,type:journal-article",
            "select": "DOI,title,author,abstract,URL,published-print,container-title"
        }
        
        crossref_url = "https://api.crossref.org/works"
        response = requests.get(crossref_url, params=search_params, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        
        data = response.json()
        items = data.get("message", {}).get("items", [])
        
        # Fallback: Tentar query original
        if not items:
            print("🔍 Crossref: Nenhum resultado com query expandida. Tentando query original...")
            query_data = build_query(query, input_language=input_language, target_language="en-us", max_depth=0)
            search_params["query"] = query_data["query"]
            response = requests.get(crossref_url, params=search_params, timeout=HTTP_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            items = data.get("message", {}).get("items", [])
        
        # Fallback: Tentar termo simplificado
        if not items:
            print("🔍 Crossref: Nenhum resultado com query original. Tentando termo simplificado...")
            simplified_term = query.split()[0] if " " in query else query
            search_params["query"] = simplified_term
            response = requests.get(crossref_url, params=search_params, timeout=HTTP_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            items = data.get("message", {}).get("items", [])
        
        if not items:
            print("❌ Crossref: Nenhum resultado encontrado.")
            suggestions = []
            for concept in SCIENTIFIC_TERMS["concepts"].values():
                for trans in concept["translations"].get(input_language, []):
                    if trans.lower() != query.lower():
                        suggestions.append(trans)
            suggestion_text = f"Sugestões: {', '.join(suggestions[:3])}" if suggestions else "Tente termos mais específicos."
            return [{
                "title": f"Nenhum resultado encontrado no Crossref para: {query}",
                "url": f"https://www.crossref.org/",
                "content": f"Não foram encontrados artigos no Crossref relacionados a '{query}'. {suggestion_text}",
                "citation": "Crossref - Sem resultados"
            }]
        
        print(f"✅ Crossref: Encontrados {len(items)} artigos")
        
        results = []
        for item in items[:max_results]:
            try:
                doi = item.get("DOI", "")
                title = " ".join(item.get("title", ["Artigo sem título"]))
                abstract = item.get("abstract", "Resumo não disponível.")
                if abstract:
                    abstract = re.sub(r'<[^>]+>', '', abstract)
                    if len(abstract) > MAX_ABSTRACT_LENGTH:
                        abstract = abstract[:MAX_ABSTRACT_LENGTH] + "..."
                
                authors = []
                for author in item.get("author", []):
                    given = author.get("given", "")
                    family = author.get("family", "")
                    if family:
                        authors.append(f"{family} {given[:1]}" if given else family)
                
                authors_text = ", ".join(authors[:3])
                if len(authors) > 3:
                    authors_text += " et al."
                
                year = ""
                if "published-print" in item and "date-parts" in item["published-print"]:
                    year = str(item["published-print"]["date-parts"][0][0])
                
                journal = item.get("container-title", [""])[0] if item.get("container-title") else ""
                
                citation = ""
                if authors_text:
                    citation = authors_text
                if year:
                    citation += f" ({year})"
                if journal:
                    citation += f". {journal}"
                if doi:
                    citation += f". DOI: {doi}"
                if not citation:
                    citation = f"Fonte: Crossref. DOI: {doi}"
                
                url = f"https://doi.org/{doi}" if doi else ""
                content = f"Título: {title}\n\nAutores: {authors_text}\n\n"
                if journal and year:
                    content += f"Publicado em: {journal}, {year}\n\n"
                if doi:
                    content += f"DOI: {doi}\n\n"
                content += f"Resumo: {abstract}"
                
                results.append({
                    "title": title[:100] + "..." if len(title) > 100 else title,
                    "url": url,
                    "content": content,
                    "citation": citation
                })
                print(f"📄 Crossref: Processado artigo com DOI {doi}")
            
            except Exception as e:
                print(f"⚠️ Erro processando artigo Crossref: {e}")
        
        if CACHE_ENABLED and results:
            crossref_cache[cache_key] = results
        return results
    
    except Exception as e:
        print(f"❌ Erro na busca Crossref: {e}")
        return [{
            "title": "Erro na busca Crossref",
            "url": "",
            "content": f"Ocorreu um erro: {str(e)}",
            "citation": "Erro - Crossref"
        }]

@with_timeout(SEARCH_TIMEOUT, default_return=[{"title": "Tempo limite excedido", "url": "", "content": "A busca no Crossref excedeu o tempo limite."}])
def search_crossref_with_timeout(query: str, max_results: int = MAX_RESULTS_DEFAULT, input_language: str = "pt-br") -> List[Dict[str, str]]:
    return search_crossref(query, max_results, input_language)

# Função para buscar no ArXiv
def search_arxiv(query: str, max_results: int = MAX_RESULTS_DEFAULT, input_language: str = "pt-br") -> List[Dict[str, str]]:
    if input_language not in SUPPORTED_LANGUAGES:
        input_language = "pt-br"
    
    cache_key = f"arxiv_{query}_{max_results}_{input_language}"
    if CACHE_ENABLED and cache_key in arxiv_cache:
        print(f"🔍 ArXiv: Usando resultados em cache para '{query}'")
        return arxiv_cache[cache_key]
    
    try:
        print(f"🔍 ArXiv: Pesquisando: '{query}'")
        
        query_data = build_query(query, input_language=input_language, target_language="en-us", max_depth=3)
        enhanced_query = query_data["query"]
        print(f"🔍 ArXiv: Busca aprimorada: '{enhanced_query}'")
        
        arxiv_url = "http://export.arxiv.org/api/query"
        search_params = {
            "search_query": f"all:{enhanced_query}",
            "start": 0,
            "max_results": max_results * 2,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        
        response = requests.get(arxiv_url, params=search_params, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        
        root = ET.fromstring(response.text)
        ns = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
        entries = root.findall('.//atom:entry', ns)
        
        # Fallback: Tentar query original
        if not entries:
            print("🔍 ArXiv: Nenhum resultado com query expandida. Tentando query original...")
            query_data = build_query(query, input_language=input_language, target_language="en-us", max_depth=0)
            search_params["search_query"] = f"all:{query_data['query']}"
            response = requests.get(arxiv_url, params=search_params, timeout=HTTP_TIMEOUT)
            response.raise_for_status()
            root = ET.fromstring(response.text)
            entries = root.findall('.//atom:entry', ns)
        
        # Fallback: Tentar termo simplificado
        if not entries:
            print("🔍 ArXiv: Nenhum resultado com query original. Tentando termo simplificado...")
            simplified_term = query.split()[0] if " " in query else query
            search_params["search_query"] = f"all:{simplified_term}"
            response = requests.get(arxiv_url, params=search_params, timeout=HTTP_TIMEOUT)
            response.raise_for_status()
            root = ET.fromstring(response.text)
            entries = root.findall('.//atom:entry', ns)
        
        if not entries:
            print("❌ ArXiv: Nenhum resultado encontrado.")
            suggestions = []
            for concept in SCIENTIFIC_TERMS["concepts"].values():
                for trans in concept["translations"].get(input_language, []):
                    if trans.lower() != query.lower():
                        suggestions.append(trans)
            suggestion_text = f"Sugestões: {', '.join(suggestions[:3])}" if suggestions else "Tente termos mais acadêmicos."
            return [{
                "title": f"Nenhum resultado encontrado no ArXiv para: {query}",
                "url": f"https://arxiv.org/search/?query={urllib.parse.quote(query)}",
                "content": f"Não foram encontrados artigos no ArXiv relacionados a '{query}'. {suggestion_text}",
                "citation": "ArXiv - Sem resultados"
            }]
        
        print(f"✅ ArXiv: Encontrados {len(entries)} artigos")
        
        results = []
        for entry in entries[:max_results]:
            try:
                id_element = entry.find('./atom:id', ns)
                if id_element is None or id_element.text is None:
                    continue
                id_url = id_element.text
                arxiv_id = id_url.split('/abs/')[-1] if '/abs/' in id_url else id_url
                
                title_element = entry.find('./atom:title', ns)
                title = title_element.text.strip() if title_element is not None and title_element.text is not None else "Sem título"
                if title:
                    title = re.sub(r'\s+', ' ', title)
                
                summary_element = entry.find('./atom:summary', ns)
                summary = summary_element.text if summary_element is not None and summary_element.text is not None else "Resumo não disponível"
                if summary:
                    summary = re.sub(r'\s+', ' ', summary).strip()
                    if len(summary) > MAX_ABSTRACT_LENGTH:
                        summary = summary[:MAX_ABSTRACT_LENGTH] + "..."
                
                authors = []
                for author in entry.findall('./atom:author/atom:name', ns):
                    if author.text:
                        name_parts = author.text.strip().split()
                        if name_parts:
                            if len(name_parts) > 1:
                                last_name = name_parts[-1]
                                initials = ''.join([n[0] for n in name_parts[:-1]])
                                authors.append(f"{last_name} {initials}")
                            else:
                                authors.append(name_parts[0])
                
                authors_text = ", ".join(authors[:3])
                if len(authors) > 3:
                    authors_text += " et al."
                
                published_element = entry.find('./atom:published', ns)
                published = published_element.text if published_element is not None and published_element.text is not None else ""
                year = published.split('-')[0] if published and '-' in published else ""
                
                categories = []
                for category in entry.findall('./atom:category', ns):
                    if 'term' in category.attrib:
                        categories.append(category.attrib['term'])
                
                categories_text = ", ".join(categories[:3])
                if len(categories) > 3:
                    categories_text += ", ..."
                
                url = f"https://arxiv.org/abs/{arxiv_id}"
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                
                citation = ""
                if authors_text:
                    citation = authors_text
                if year:
                    citation += f" ({year})"
                citation += f". arXiv:{arxiv_id}"
                if categories_text:
                    citation += f" [{categories_text}]"
                
                content = f"Título: {title}\n\nAutores: {authors_text}\n\n"
                if year:
                    content += f"Publicado em: {year}\n\n"
                if categories_text:
                    content += f"Categorias: {categories_text}\n\n"
                content += f"ID: arXiv:{arxiv_id}\n\n"
                content += f"Resumo: {summary}\n\n"
                content += f"PDF: {pdf_url}"
                
                results.append({
                    "title": title[:100] + "..." if len(title) > 100 else title,
                    "url": url,
                    "content": content,
                    "citation": citation
                })
                print(f"📄 ArXiv: Processado artigo com ID {arxiv_id}")
            
            except Exception as e:
                print(f"⚠️ Erro processando artigo ArXiv: {e}")
        
        if CACHE_ENABLED and results:
            arxiv_cache[cache_key] = results
        return results
    
    except Exception as e:
        print(f"❌ Erro na busca ArXiv: {e}")
        return [{
            "title": "Erro na busca ArXiv",
            "url": f"https://arxiv.org/search/?query={urllib.parse.quote(query)}",
            "content": f"Ocorreu um erro: {str(e)}",
            "citation": "Erro - ArXiv"
        }]

@with_timeout(SEARCH_TIMEOUT, default_return=[{"title": "Tempo limite excedido", "url": "", "content": "A busca no ArXiv excedeu o tempo limite."}])
def search_arxiv_with_timeout(query: str, max_results: int = MAX_RESULTS_DEFAULT, input_language: str = "pt-br") -> List[Dict[str, str]]:
    return search_arxiv(query, max_results, input_language)

# Função para buscar no Semantic Scholar
def search_semantic_scholar(query: str, max_results: int = MAX_RESULTS_DEFAULT, input_language: str = "pt-br") -> List[Dict[str, str]]:
    if input_language not in SUPPORTED_LANGUAGES:
        input_language = "pt-br"
    
    cache_key = f"semantic_{query}_{max_results}_{input_language}"
    if CACHE_ENABLED and cache_key in semantic_cache:
        print(f"🔍 Semantic Scholar: Usando resultados em cache para '{query}'")
        return semantic_cache[cache_key]
    
    try:
        print(f"🔍 Semantic Scholar: Pesquisando: '{query}'")
        
        query_data = build_query(query, input_language=input_language, target_language="en-us", max_depth=3)
        enhanced_query = query_data["query"]
        print(f"🔍 Semantic Scholar: Busca aprimorada: '{enhanced_query}'")
        
        semantic_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        fields = "paperId,title,abstract,url,year,authors,venue,publicationDate,publicationTypes,openAccessPdf"
        search_params = {
            "query": enhanced_query,
            "limit": max_results * 2,
            "fields": fields
        }
        headers = {"User-Agent": "Scify Health Search/0.1.0 (contact: liongabr@gmail.com)"}
        
        response = requests.get(semantic_url, params=search_params, headers=headers, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        
        data = response.json()
        papers = data.get("data", [])
        
        # Fallback: Tentar query original
        if not papers:
            print("🔍 Semantic Scholar: Nenhum resultado com query expandida. Tentando query original...")
            query_data = build_query(query, input_language=input_language, target_language="en-us", max_depth=0)
            search_params["query"] = query_data["query"]
            response = requests.get(semantic_url, params=search_params, headers=headers, timeout=HTTP_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            papers = data.get("data", [])
        
        # Fallback: Tentar termo simplificado
        if not papers:
            print("🔍 Semantic Scholar: Nenhum resultado com query original. Tentando termo simplificado...")
            simplified_term = query.split()[0] if " " in query else query
            search_params["query"] = simplified_term
            response = requests.get(semantic_url, params=search_params, headers=headers, timeout=HTTP_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            papers = data.get("data", [])
        
        if not papers:
            print("❌ Semantic Scholar: Nenhum resultado encontrado.")
            suggestions = []
            for concept in SCIENTIFIC_TERMS["concepts"].values():
                for trans in concept["translations"].get(input_language, []):
                    if trans.lower() != query.lower():
                        suggestions.append(trans)
            suggestion_text = f"Sugestões: {', '.join(suggestions[:3])}" if suggestions else "Tente termos mais específicos."
            return [{
                "title": f"Nenhum resultado encontrado no Semantic Scholar para: {query}",
                "url": f"https://www.semanticscholar.org/search?q={urllib.parse.quote(query)}",
                "content": f"Não foram encontrados artigos no Semantic Scholar relacionados a '{query}'. {suggestion_text}",
                "citation": "Semantic Scholar - Sem resultados"
            }]
        
        print(f"✅ Semantic Scholar: Encontrados {len(papers)} artigos")
        
        results = []
        for paper in papers[:max_results]:
            try:
                paper_id = paper.get("paperId", "")
                title = paper.get("title", "Artigo sem título")
                abstract = paper.get("abstract", "Resumo não disponível.")
                if abstract and len(abstract) > MAX_ABSTRACT_LENGTH:
                    abstract = abstract[:MAX_ABSTRACT_LENGTH] + "..."
                
                url = paper.get("url") or f"https://www.semanticscholar.org/paper/{paper_id}"
                year = paper.get("year", "")
                venue = paper.get("venue", "")
                
                authors = []
                for author in paper.get("authors", []):
                    name = author.get("name", "")
                    if name:
                        authors.append(name)
                
                authors_text = ", ".join(authors[:3])
                if len(authors) > 3:
                    authors_text += " et al."
                
                pdf_url = ""
                if "openAccessPdf" in paper and paper["openAccessPdf"]:
                    pdf_url = paper["openAccessPdf"].get("url", "")
                
                citation = ""
                if authors_text:
                    citation = authors_text
                if year:
                    citation += f" ({year})"
                if venue:
                    citation += f". {venue}"
                if paper_id:
                    citation += f". S2 ID: {paper_id}"
                if not citation:
                    citation = f"Fonte: Semantic Scholar. ID: {paper_id}"
                
                content = f"Título: {title}\n\nAutores: {authors_text}\n\n"
                if venue and year:
                    content += f"Publicado em: {venue}, {year}\n\n"
                content += f"Resumo: {abstract}"
                if pdf_url:
                    content += f"\n\nPDF: {pdf_url}"
                
                results.append({
                    "title": title[:100] + "..." if len(title) > 100 else title,
                    "url": url,
                    "content": content,
                    "citation": citation
                })
                print(f"📄 Semantic Scholar: Processado artigo com ID {paper_id}")
            
            except Exception as e:
                print(f"⚠️ Erro processando artigo Semantic Scholar: {e}")
        
        if CACHE_ENABLED and results:
            semantic_cache[cache_key] = results
        return results
    
    except Exception as e:
        print(f"❌ Erro na busca Semantic Scholar: {e}")
        return [{
            "title": "Erro na busca Semantic Scholar",
            "url": f"https://www.semanticscholar.org/search?q={urllib.parse.quote(query)}",
            "content": f"Ocorreu um erro: {str(e)}",
            "citation": "Erro - Semantic Scholar"
        }]

@with_timeout(SEARCH_TIMEOUT, default_return=[{"title": "Tempo limite excedido", "url": "", "content": "A busca no Semantic Scholar excedeu o tempo limite."}])
def search_semantic_scholar_with_timeout(query: str, max_results: int = MAX_RESULTS_DEFAULT, input_language: str = "pt-br") -> List[Dict[str, str]]:
    return search_semantic_scholar(query, max_results, input_language)

# Função para processar resumos
def process_summaries_parallel(results: List[Dict[str, str]], user_input: str) -> List[SearchResult]:
    def process_one(result):
        try:
            title = result["title"]
            url = result["url"]
            content = result["content"]
            citation = result.get("citation", "Fonte: PubMed")
            
            if SKIP_SUMMARY:
                print(f"⚡ Modo rápido: Usando conteúdo direto para {title[:30]}...")
                summary = ""
                if "Resumo:" in content:
                    summary = content.split("Resumo:", 1)[1].strip()
                else:
                    summary = content
                if len(summary) > MAX_ABSTRACT_LENGTH:
                    summary = summary[:MAX_ABSTRACT_LENGTH] + "..."
                if len(title) > 100:
                    title = title[:100] + "..."
                if not url or not url.startswith("http"):
                    url = f"https://pubmed.ncbi.nlm.nih.gov/?term={urllib.parse.quote(title)}"
                
                return SearchResult(
                    title=title,
                    url=url,
                    summary=summary,
                    citation=citation
                )
            
            summary_prompt_text = search_summary_prompt.format(
                user_input=user_input,
                content=content
            )
            
            @with_timeout(SUMMARY_TIMEOUT, default_return="Tempo limite excedido ao resumir.")
            def get_summary(prompt):
                return summary_llm.invoke(prompt).content
            
            summary = get_summary(summary_prompt_text)
            return SearchResult(
                title=title,
                url=url,
                summary=str(summary) if summary else "Sem resumo disponível",
                citation=citation
            )
        
        except Exception as e:
            print(f"❌ Erro ao resumir: {e}")
            return SearchResult(
                title=title,
                url=url,
                summary=f"Erro ao resumir: {str(e)}",
                citation=citation
            )
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        search_results = list(executor.map(process_one, results))
    return search_results

# Função de busca principal
def single_search(state: ReportState) -> Dict[str, List[SearchResult]]:
    query_text = state.user_input
    search_sources = state.search_sources
    max_results = state.max_results_per_source
    input_language = getattr(state, "language", "pt-br")
    if input_language not in SUPPORTED_LANGUAGES:
        input_language = "pt-br"
    
    results_pubmed = []
    results_crossref = []
    results_arxiv = []
    results_semantic = []
    
    start_time = time.time()
    state.timestamps["search_start"] = start_time
    
    try:
        normalized_query = normalize_term(query_text, language=input_language)
        corrected_query, confidence, level = correct_typo(normalized_query, list(TERM_INDEX.keys()), input_language)
        query_data = build_query(corrected_query, input_language=input_language, target_language="en-us")
        enhanced_query = query_data["query"]
        print(f"🔍 Query processada: '{query_text}' -> '{enhanced_query}' (confiança: {confidence}, nível: {level})")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            if search_sources.get("pubmed", True):
                futures["pubmed"] = executor.submit(search_pubmed_with_timeout, corrected_query, max_results, input_language)
            if search_sources.get("crossref", True):
                futures["crossref"] = executor.submit(search_crossref_with_timeout, corrected_query, max_results, input_language)
            if search_sources.get("arxiv", True):
                futures["arxiv"] = executor.submit(search_arxiv_with_timeout, corrected_query, max_results, input_language)
            if search_sources.get("semantic", True):
                futures["semantic"] = executor.submit(search_semantic_scholar_with_timeout, corrected_query, max_results, input_language)
            
            if "pubmed" in futures:
                pubmed_start = time.time()
                raw_results_pubmed = futures["pubmed"].result()
                pubmed_time = time.time() - pubmed_start
                api_response_times["pubmed"].append(pubmed_time)
                if raw_results_pubmed:
                    results_pubmed = process_summaries_parallel(raw_results_pubmed, query_text)
                    for result in results_pubmed:
                        result.source = "PubMed"
            
            if "crossref" in futures:
                crossref_start = time.time()
                raw_results_crossref = futures["crossref"].result()
                crossref_time = time.time() - crossref_start
                api_response_times["crossref"].append(crossref_time)
                if raw_results_crossref:
                    results_crossref = process_summaries_parallel(raw_results_crossref, query_text)
                    for result in results_crossref:
                        result.source = "Crossref"
            
            if "arxiv" in futures:
                arxiv_start = time.time()
                raw_results_arxiv = futures["arxiv"].result()
                arxiv_time = time.time() - arxiv_start
                api_response_times["arxiv"].append(arxiv_time)
                if raw_results_arxiv:
                    results_arxiv = process_summaries_parallel(raw_results_arxiv, query_text)
                    for result in results_arxiv:
                        result.source = "ArXiv"
            
            if "semantic" in futures:
                semantic_start = time.time()
                raw_results_semantic = futures["semantic"].result()
                semantic_time = time.time() - semantic_start
                api_response_times["semantic"].append(semantic_time)
                if raw_results_semantic:
                    results_semantic = process_summaries_parallel(raw_results_semantic, query_text)
                    for result in results_semantic:
                        result.source = "Semantic Scholar"
        
        combined_results = []
        max_len = max(
            len(results_pubmed) if results_pubmed else 0,
            len(results_crossref) if results_crossref else 0,
            len(results_arxiv) if results_arxiv else 0,
            len(results_semantic) if results_semantic else 0
        )
        for i in range(max_len):
            if results_pubmed and i < len(results_pubmed):
                combined_results.append(results_pubmed[i])
            if results_crossref and i < len(results_crossref):
                combined_results.append(results_crossref[i])
            if results_arxiv and i < len(results_arxiv):
                combined_results.append(results_arxiv[i])
            if results_semantic and i < len(results_semantic):
                combined_results.append(results_semantic[i])
        
        combined_results = combined_results[:max_results * 4]
        
        end_time = time.time()
        total_search_time = end_time - start_time
        state.timestamps["search_end"] = end_time
        state.timestamps["search_duration"] = total_search_time
        
        print(f"✅ Busca concluída em {total_search_time:.2f}s. Total de resultados: {len(combined_results)}")
        return {"search_results": combined_results}
    
    except Exception as e:
        print(f"❌ Erro na pesquisa: {str(e)}")
        return {"search_results": []}

# Função de revisão
@with_timeout(REVIEW_TIMEOUT, {"final_response": "Tempo limite excedido ao gerar o relatório final."})
def reviewer(state: ReportState) -> Dict[str, str]:
    try:
        if not state.search_results:
            return {"final_response": f"<div class='result-summary'>Não foram encontrados resultados para a sua pesquisa: <strong>{state.user_input}</strong>.</div>"}
        
        if len(state.search_results) == 1 and "Nenhum resultado encontrado para:" in state.search_results[0].title:
            no_results_response = f"""# Pesquisa Científica: {state.user_input}

<div class="result-summary">
<h3>Não foram encontrados estudos específicos</h3>
<p>Sua busca por "<strong>{state.user_input}</strong>" não retornou resultados diretos.</p>
<ul>
  <li>Use termos em inglês (ex.: "muscle growth" em vez de "crescimento muscular")</li>
  <li>Adicione contexto (ex.: "hypertrophy training")</li>
  <li>Combine termos relacionados (ex.: "weight loss exercise diet")</li>
</ul>
<p>Buscar diretamente em:</p>
<ul>
  <li><a href="https://pubmed.ncbi.nlm.nih.gov/?term={urllib.parse.quote(state.user_input)}">PubMed</a></li>
  <li><a href="https://arxiv.org/search/?query={urllib.parse.quote(state.user_input)}">ArXiv</a></li>
  <li><a href="https://www.semanticscholar.org/search?q={urllib.parse.quote(state.user_input)}">Semantic Scholar</a></li>
</ul>
</div>
"""
            return {"final_response": no_results_response}
        
        results_by_source = {}
        for result in state.search_results:
            source = result.source or "Outros"
            if source not in results_by_source:
                results_by_source[source] = []
            results_by_source[source].append(result)
        
        header = f"""# Pesquisa Científica: {state.user_input}

### Resultados mais relevantes encontrados em bases científicas

---

"""
        response_times = ""
        for api, times in api_response_times.items():
            if times:
                avg_time = sum(times) / len(times)
                response_times += f"- {api.title()}: {avg_time:.2f}s em média\n"
        
        if response_times:
            header += f"""<div class="stats-box" style="font-size: 0.8em; color: #666; margin-bottom: 20px;">
<strong>Estatísticas:</strong>
<ul style="margin: 5px 0;">
<li>Total de fontes: {len(results_by_source)}</li>
<li>Total de artigos: {len(state.search_results)}</li>
</ul>
</div>

"""
        
        all_results_formatted = []
        source_order = ["PubMed", "Crossref", "ArXiv", "Semantic Scholar", "Outros"]
        
        for source in source_order:
            if source in results_by_source and results_by_source[source]:
                source_results = results_by_source[source]
                source_header = f"""<div class="source-header" style="background-color: #f5f5f5; padding: 10px; margin: 20px 0 10px 0; border-left: 5px solid #007bff;">
<h2 style="margin: 0;">{source} ({len(source_results)})</h2>
</div>"""
                all_results_formatted.append(source_header)
                
                for i, result in enumerate(source_results):
                    title = result.title
                    url = result.url
                    summary = result.summary
                    citation = result.citation or f"Fonte: {source}"
                    
                    article_formatted = f"""<div class="article" style="margin-bottom: 20px; padding-bottom: 20px; border-bottom: 1px solid #eee;">
<h3>{i+1}. {title}</h3>
<p><em>{citation}</em></p>
<div class="summary">{summary}</div>
<p style="margin-top: 10px;">
<a href="{url}" target="_blank" style="background-color: #007bff; color: white; padding: 5px 10px; text-decoration: none; border-radius: 3px;">Ver artigo completo</a>
</p>
</div>

"""
                    all_results_formatted.append(article_formatted)
        
        results_text = "\n".join(all_results_formatted)
        footer = """<div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee; font-size: 0.8em; color: #666;">
<p>Resultados obtidos de múltiplas bases científicas usando busca avançada.</p>
<p>Para mais informações, clique nos links para ler os artigos completos.</p>
</div>
"""
        
        final_text = header + results_text + footer
        return {"final_response": final_text}
    
    except Exception as e:
        print(f"❌ Erro ao revisar: {e}")
        emergency_response = f"<div class='result-summary'>Ocorreu um erro ao formatar os resultados: {str(e)}</div>"
        if hasattr(state, 'search_results') and state.search_results:
            emergency_response += "<ul>"
            for result in state.search_results:
                emergency_response += f"<li><a href='{result.url}' target='_blank'>{result.title}</a></li>"
            emergency_response += "</ul>"
        return {"final_response": emergency_response}

# Construindo o grafo
builder = StateGraph(ReportState)
builder.add_node("single_search", single_search)
builder.add_node("reviewer", reviewer)
builder.add_edge(START, "single_search")
builder.add_edge("single_search", "reviewer")
builder.add_edge("reviewer", END)

print("🔄 Compilando o grafo de processamento...")
graph = builder.compile()