import os
from dotenv import load_dotenv
load_dotenv()

import requests
from newspaper import Article

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
}

article_url = "https://www.artificialintelligence-news.com/2022/01/25/meta-claims-new-ai-supercomputer-will-set-records/"

session = requests.Session()

try:
    response = session.get(article_url, headers=headers, timeout=10)
    if response.status_code == 200:
        article = Article(article_url)
        article.download()
        article.parse()
        print(f"Title: {article.title}")
        print(f"Text: {article.text}")
    else:
        print(f"Failed to fetch article at {article_url}")
except Exception as e:
    print(f"Error occurred while fetching article at {article_url}: {e}")

article_title = article.title
article_text = article.text

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import field_validator, BaseModel, Field
from typing import List

# create output parser class
class ArticleSummary(BaseModel):
    title: str = Field(description="Title of the article")
    summary: List[str] = Field(description="Bulleted list summary of the article")

    @field_validator('summary')
    @classmethod
    def has_three_or_more_lines(cls, list_of_lines):
        if len(list_of_lines) < 3:
            raise ValueError("Generated summary has less than three bullet points!")
        return list_of_lines

# set up output parser
parser = PydanticOutputParser(pydantic_object=ArticleSummary)

from langchain_core.prompts import PromptTemplate

# create prompt template
template = """
You are a very good assistant that summarizes online articles.
Here's the article you want to summarize.
==================
Title: {article_title}
{article_text}
==================
{format_instructions}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["article_title", "article_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

formatted_prompt = prompt.format_prompt(article_title=article_title, article_text=article_text)

from langchain_groq import ChatGroq

chat = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

output = chat.invoke(formatted_prompt.to_string())

parsed_output = parser.parse(output.content)
print(parsed_output)
