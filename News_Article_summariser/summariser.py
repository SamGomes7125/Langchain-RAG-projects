

import requests
from newspaper import Article

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
}

article_urls = "https://www.artificialintelligence-news.com/2022/01/25/meta-claims-new-ai-supercomputer-will-set-records/"

session = requests.Session()

try:
    response = session.get(article_urls, headers=headers, timeout=10)

    if response.status_code == 200:
        article = Article(article_urls)
        article.download()
        article.parse()

        print(f"Title: {article.title}")
        print(f"Text: {article.text}")

    else:
        print(f"Failed to fetch article at {article_urls}")
except Exception as e:
    print(f"Error occurred while fetching article at {article_urls}: {e}")
    
from langchain_core.messages import HumanMessage

# we get the article data from the scraping part
article_title = article.title
article_text = article.text

import os
from dotenv import load_dotenv

# This loads the variables from the .env file into the environment
load_dotenv()


# prepare template for prompt
template = """You are a very good assistant that summarizes online articles.

Here's the article you want to summarize.

==================
Title: {article_title}

{article_text}
==================

Write a summary of the previous article.
"""

prompt = template.format(article_title=article.title, article_text=article.text)

messages = [HumanMessage(content=prompt)]

from langchain_groq import ChatGroq

# load the model
chat = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

# generate summary
summary = chat.invoke(messages)
print(summary.content)