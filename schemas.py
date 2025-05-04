from pydantic import BaseModel, Field
from typing import List, Optional
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

# Estado do grafo (ReportState)
class ReportState(BaseModel):
    user_input: str = Field(default="", description="Pergunta do usuário")
    report_structure: Optional[ReportStructure] = Field(default=None, description="Estrutura do relatório")
    search_results: Annotated[List[SearchResult], operator.add] = Field(
        default_factory=list, description="Resultados das buscas (acumulados)"
    )
    final_response: str = Field(default="", description="Resposta final")
    source: str = Field(default="pubmed", description="Fonte de pesquisa (pubmed ou tavily)")