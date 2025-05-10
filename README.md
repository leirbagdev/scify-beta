# SciFy v2 - Pesquisa Científica Inteligente

SciFy é uma aplicação para pesquisa de artigos científicos que facilita o acesso a informações de saúde e bem-estar baseadas em evidências.

## Funcionalidades

- Busca de artigos científicos em múltiplas fontes:
  - PubMed (literatura biomédica)
  - Semantic Scholar (artigos científicos com análise semântica)
  - Europe PMC (literatura biomédica europeia)
  - CrossRef (metadados de artigos acadêmicos)
- Pesquisa simultânea em múltiplas fontes
- Exibição de resumos e metadados dos artigos encontrados
- Filtro de resultados por fonte
- Tradução simplificada de termos técnicos
- Exportação de resultados em formato CSV
- Interface amigável construída com Streamlit

## Requisitos

- Python 3.7+
- Bibliotecas listadas em `requirements.txt`

## Instalação

1. Clone o repositório:
```bash
git clone <seu-repositorio>
cd scifyv2
```

2. Crie um ambiente virtual e instale as dependências:
```bash
python -m venv .venv_py311
source .venv_py311/bin/activate  # No Windows: .venv_py311\Scripts\activate
pip install -r requirements.txt
```

3. Execute a aplicação:
```bash
streamlit run app.py
```

4. Acesse a interface web no navegador (geralmente em http://localhost:8501).

## Configuração

Você pode configurar as credenciais de API criando um arquivo `.env` com:

```
PUBMED_EMAIL=seu.email@exemplo.com
PUBMED_API_KEY=sua_chave_api
```

## Como usar

1. Digite seu termo de busca no campo de texto
2. Selecione as fontes de dados que deseja pesquisar no menu lateral
3. Ajuste o número máximo de resultados por fonte
4. Clique em "Pesquisar"
5. Use o filtro de fontes para refinar os resultados, se necessário
6. Navegue pelos resultados com informações de título, autores, resumo e link para o artigo completo
7. Exporte os resultados em CSV se desejar

## APIs utilizadas

- PubMed E-utilities: https://www.ncbi.nlm.nih.gov/books/NBK25501/
- Semantic Scholar API: https://www.semanticscholar.org/product/api
- Europe PMC REST API: https://europepmc.org/RestfulWebService
- CrossRef REST API: https://www.crossref.org/documentation/retrieve-metadata/rest-api/

## Aviso

Esta aplicação é apenas para fins educacionais e de pesquisa. Sempre consulte profissionais de saúde para orientações médicas.

#### SciFyV2

#### Um projeto Python que utiliza langchain-ollama. 


#### deepseek-r1:14b
#### llama3.1:8b-instruct-q4_K_S

#### Chief Information Security Officers (CISO) e os líderes de segurança comunicam o risco cibernético ao conselho de administração.

#### A primeira pesquisa acadêmica sobre governança, risco e compliance definiu o GRC  como ####"o conjunto integrado de capacidades que permitem que uma organização atinja de ####maneira confiável os objetivos, aborde as incertezas e aja com integridade". 


