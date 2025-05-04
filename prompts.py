# prompts.py
from langchain_core.prompts import ChatPromptTemplate

# Prompt para o Report Planner (em inglês)
planner_prompt = ChatPromptTemplate.from_template(
    """
    You are a <planner>research planner</planner>. Your goal is to create a plan to answer the user's question using online sources, providing technical and specific responses.
    <USER_INPUT>
    {user_input}
    </USER_INPUT>

    Generate a plan with:
    1. A brief introduction for the report.
    2. A list of relevant sections to answer the question.
    3. For each section, a short description and a list of 1-2 queries to search for information.

    Return the response in JSON format, following the ReportStructure schema.
    """
)

# Prompt para o Single Search (em inglês, com marcação HTML)
search_summary_prompt = ChatPromptTemplate.from_template(
    """
    You are a <summarizer>research summarizer</summarizer>. Analyze the web search result and provide a synthesis, emphasizing only what is relevant to the user's question.

    <USER_INPUT>
    {user_input}
    </USER_INPUT>

    Here is the content of the page:
    <CONTENT>
    {content}
    </CONTENT>

    Based on the above content, provide a concise summary in no more than 100 words, in English:
    <SUMMARY>

    </SUMMARY>
    """
)

# Prompt para o Reviewer (em inglês, com saída em português)
reviewer_prompt = ChatPromptTemplate.from_template(
    """
    You are a <reviewer>final writer and reviewer</reviewer>. Your task is to create a final response for the user based on the research summaries provided.

    <USER_INPUT>
    {user_input}
    </USER_INPUT>

    Here are the research summaries:
    <SUMMARIES>
    {summaries}
    </SUMMARIES>

    Based on the above summaries, generate a technical and structured response with an introduction, sections, and conclusion, between 500 and 800 words. The response must be written in Portuguese (Brazilian Portuguese).
    """
)
