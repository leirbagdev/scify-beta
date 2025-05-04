from langgraph.graph import StateGraph
from langgraph.constants import START, END
from langgraph.types import Send
from schemas import ReportState, ReportStructure, SearchResult, Section, Query
from prompts import planner_prompt, search_summary_prompt, reviewer_prompt
from langchain_ollama import ChatOllama
from tavily import TavilyClient
from Bio import Entrez
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple, TypedDict, Optional, Union, cast
import os
import json
import re

# Carregar variáveis de ambiente
load_dotenv()

# Configuração das LLMs
model_name = "mistral:7b-instruct-v0.2-q4_0"  # Modelo mais leve e eficiente
planner_llm = ChatOllama(model=model_name, base_url=os.getenv("OLLAMA_HOST"))
summary_llm = ChatOllama(model=model_name, base_url=os.getenv("OLLAMA_HOST"))
writer_llm = ChatOllama(model=model_name, base_url=os.getenv("OLLAMA_HOST"))

# Configuração do Tavily e PubMed
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
Entrez.email = "liongabr@gmail.com"  # Substitua pelo seu email
Entrez.api_key = os.getenv("PUBMED_API_KEY")

# Função para criar uma estrutura padrão de relatório
def create_default_structure(query: str) -> ReportStructure:
    """Cria uma estrutura padrão para o relatório"""
    return ReportStructure(
        introduction=f"Relatório científico sobre: {query}",
        sections=[
            Section(
                title="Introdução",
                description="Introdução ao tema de pesquisa",
                queries=[Query(query=query)]
            )
        ]
    )

# Função para planejar o relatório científico
def report_planner(state: ReportState) -> ReportState:
    """Planeja a estrutura do relatório científico."""
    try:
        prompt = planner_prompt.format(
            user_input=state.user_input
        )
        planner_response = planner_llm.invoke(prompt)
        response_text = str(planner_response.content)
        
        # Extrair dados estruturados da resposta
        try:
            # Tenta encontrar um bloco JSON na resposta
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text)
            data = {}
            
            if json_match:
                # Se encontrou um bloco JSON, tenta parsear
                json_text = json_match.group(1).strip()
                data = json.loads(json_text)
            else:
                # Caso contrário, tenta parsear a resposta completa
                data = json.loads(response_text)
                
            # Criar a estrutura do relatório a partir dos dados obtidos
            sections = []
            for section_data in data.get("sections", []):
                queries = []
                for query_data in section_data.get("queries", []):
                    query_text = ""
                    if isinstance(query_data, dict):
                        query_text = query_data.get("query", state.user_input)
                    else:
                        query_text = str(query_data)
                    queries.append(Query(query=query_text))
                
                sections.append(Section(
                    title=section_data.get("title", "Seção"),
                    description=section_data.get("description", "Sem descrição"),
                    queries=queries or [Query(query=state.user_input)]
                ))
            
            if not sections:
                sections = [Section(
                    title="Pesquisa Geral",
                    description="Informações gerais sobre o tema",
                    queries=[Query(query=state.user_input)]
                )]
            
            report_structure = ReportStructure(
                introduction=data.get("introduction", f"Relatório sobre: {state.user_input}"),
                sections=sections
            )
            
        except Exception as e:
            print(f"Erro ao parsear resposta: {e}")
            # Em caso de erro, usa uma estrutura padrão
            report_structure = create_default_structure(state.user_input)
        
        # Atualiza o estado
        state_dict = state.model_dump()
        state_dict["report_structure"] = report_structure
        return ReportState.model_validate(state_dict)
        
    except Exception as e:
        print(f"Erro geral no planner: {e}")
        # Em caso de erro mais grave, retorna estado com estrutura padrão
        state_dict = state.model_dump()
        state_dict["report_structure"] = create_default_structure(state.user_input)
        return ReportState.model_validate(state_dict)

# Função auxiliar para buscar no PubMed
def search_pubmed(query: str, max_results: int = 1) -> List[Dict[str, str]]:
    """Busca artigos no PubMed."""
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        record = Entrez.read(handle)
        handle.close()
        
        # Extrai os IDs de forma segura
        ids = []
        if "IdList" in record and record["IdList"]:
            ids = record["IdList"]
        
        if not ids:
            return []
            
        handle = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="text")
        abstracts = handle.read()
        handle.close()
        
        results = []
        for id in ids:
            results.append({
                "title": f"PubMed ID {id}",
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{id}/",
                "content": abstracts
            })
        return results
        
    except Exception as e:
        print(f"Erro na busca PubMed: {e}")
        return [{
            "title": "Erro na busca PubMed",
            "url": "",
            "content": f"Ocorreu um erro: {str(e)}"
        }]

# Função auxiliar para buscar no Tavily
def search_tavily(query: str, max_results: int = 1) -> List[Dict[str, str]]:
    """Busca informações usando a API Tavily."""
    try:
        response = tavily_client.search(query=query, max_results=max_results)
        results = []
        
        for r in response.get("results", [])[:max_results]:
            results.append({
                "title": r.get("title", "Sem título"),
                "url": r.get("url", ""),
                "content": r.get("raw_content", r.get("content", "Sem conteúdo"))
            })
        
        return results
        
    except Exception as e:
        print(f"Erro na busca Tavily: {e}")
        return [{
            "title": "Erro na busca Tavily",
            "url": "",
            "content": f"Ocorreu um erro: {str(e)}"
        }]

# Função para realizar uma única busca
def single_search(state: tuple[str, str, str]) -> Dict[str, List[SearchResult]]:
    """Realiza uma única busca com base na query fornecida.
    
    Args:
        state: Tupla contendo (query, user_input, source)
    """
    query, user_input, source = state  # Descompacta a tupla
    
    try:
        print(f"Iniciando busca para: {query} (fonte: {source})")
        
        # Decide qual serviço usar com base na fonte selecionada
        if source == "pubmed" or "medicina" in query.lower() or "saúde" in query.lower():
            print(f"Usando PubMed para buscar: {query}")
            results = search_pubmed(query)
        else:
            print(f"Usando Tavily para buscar: {query}")
            results = search_tavily(query)
        
        if not results:
            print(f"Nenhum resultado encontrado para: {query}")
            return {"search_results": []}
        
        search_results = []
        for result in results:
            try:
                # Tenta resumir o conteúdo
                print(f"Resumindo: {result['title']}")
                summary_prompt = search_summary_prompt.format(
                    user_input=user_input,
                    content=result["content"]
                )
                summary = summary_llm.invoke(summary_prompt).content
                
                search_results.append(SearchResult(
                    title=result["title"],
                    url=result["url"],
                    summary=str(summary) if summary else "Sem resumo disponível"
                ))
            except Exception as e:
                print(f"Erro ao resumir: {e}")
                # Em caso de erro, adiciona um resultado com o erro
                search_results.append(SearchResult(
                    title=result["title"],
                    url=result["url"],
                    summary=f"Erro ao resumir: {str(e)}"
                ))
        
        return {"search_results": search_results}
        
    except Exception as e:
        print(f"Erro na busca: {e}")
        return {"search_results": [
            SearchResult(
                title="Erro na pesquisa",
                url="",
                summary=f"Ocorreu um erro: {str(e)}"
            )
        ]}

# Função para distribuir buscas
def spawn_searches(state: ReportState):
    """Gera buscas paralelas com base na estrutura do relatório."""
    if not state.report_structure:
        return [Send("single_search", (state.user_input, state.user_input, state.source))]
    
    queries = []
    for section in state.report_structure.sections:
        for query in section.queries:
            # Inclui a query, o input original do usuário e a fonte de pesquisa
            queries.append((query.query, state.user_input, state.source))
    
    if not queries:
        queries.append((state.user_input, state.user_input, state.source))
    
    # Retorna uma lista de Send para buscas em paralelo
    return [Send("single_search", query) for query in queries]

# Função para revisar e finalizar o relatório
def reviewer(state: ReportState) -> Dict[str, str]:
    """Revisa os resultados de busca e gera o relatório final."""
    try:
        # Debug: imprimir informações sobre o estado
        print(f"Estado recebido no revisor: {state}")
        print(f"Número de resultados: {len(state.search_results) if state.search_results else 'Nenhum'}")
        
        # Verifica se temos resultados
        if not state.search_results:
            print("Sem resultados de busca. Gerando resposta simples.")
            return {"final_response": "Não foram encontrados resultados para a sua pesquisa."}
        
        # Obtém os resumos de todas as buscas
        summaries = "\n\n".join([f"### {result.title}\n{result.summary}" 
                                for result in state.search_results])
        
        print(f"Resumos concatenados: {summaries[:100]}...")
        
        # Resposta simples para teste
        if not summaries:
            return {"final_response": "Não foram encontrados resultados para a sua pesquisa."}
            
        # OPÇÃO PARA CONTORNAR PROBLEMAS COM O MODELO
        # Em caso de problemas, descomente esta linha para retornar uma resposta direta
        # return {"final_response": f"Resultados encontrados:\n\n{summaries}"}
        
        # Gera o relatório final
        print("Chamando o modelo para gerar o relatório final...")
        prompt = reviewer_prompt.format(
            user_input=state.user_input,
            summaries=summaries
        )
        
        response = writer_llm.invoke(prompt)
        print("Resposta do modelo recebida.")
        
        content = str(response.content) if response.content else "Não foi possível gerar um relatório"
        return {"final_response": content}
        
    except Exception as e:
        print(f"Erro ao revisar: {e}")
        import traceback
        traceback.print_exc()
        return {"final_response": f"Erro ao gerar o relatório final: {str(e)}"}

# Construindo o grafo
builder = StateGraph(ReportState)
builder.add_node("report_planner", report_planner)
builder.add_node("single_search", single_search)
builder.add_node("reviewer", reviewer)

# Conexões (edges)
builder.add_edge(START, "report_planner")
builder.add_conditional_edges(
    "report_planner",
    spawn_searches,
    {
        "single_search": "single_search"
    }
)
builder.add_edge("single_search", "reviewer")
builder.add_edge("reviewer", END)

graph = builder.compile()