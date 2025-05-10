from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import operator
from typing import Annotated

# Estrutura de uma query de busca
class Query(BaseModel):
    query: str = Field(description="Termo de busca gerado pelo planejador")

# Estrutura de uma seção do relatório
class Section(BaseModel):
    title: str = Field(description="Título da seção")
    description: str = Field(description="Descrição da seção")
    queries: List[Query] = Field(description="Lista de queries para a seção")

# Estrutura do relatório gerado pelo Report Planner
class ReportStructure(BaseModel):
    introduction: str = Field(description="Introdução do relatório")
    sections: List[Section] = Field(description="Lista de seções do relatório")

# Resultado de uma busca (Single Search)
class SearchResult(BaseModel):
    title: str = Field(description="Título da página")
    url: str = Field(description="URL da página")
    summary: str = Field(description="Resumo do conteúdo")
    citation: str = Field(default="", description="Citação do artigo")
    source: str = Field(default="", description="Fonte da busca (PubMed, Crossref, etc)")

# Estado do grafo (ReportState)
class ReportState(BaseModel):
    user_input: str = Field(default="", description="Pergunta do usuário")
    report_structure: Optional[ReportStructure] = Field(default=None, description="Estrutura do relatório")
    search_results: Annotated[List[SearchResult], operator.add] = Field(
        default_factory=list, description="Resultados das buscas (acumulados)"
    )
    final_response: str = Field(default="", description="Resposta final")
    source: str = Field(default="pubmed", description="Fonte de pesquisa (pubmed ou tavily)")
    language: str = Field(default="pt-br", description="Idioma da busca")
    max_results_per_source: int = Field(default=5, description="Número máximo de resultados por fonte")
    search_sources: Dict[str, bool] = Field(
        default_factory=lambda: {"pubmed": True, "crossref": True, "arxiv": True, "semantic_scholar": True, "europe_pmc": True},
        description="Fontes de pesquisa ativas"
    )
    timestamps: Dict[str, float] = Field(
        default_factory=dict, description="Registros de tempo para análise de performance"
    )