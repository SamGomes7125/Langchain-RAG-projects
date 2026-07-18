import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import DeepLake
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import re

load_dotenv()

my_activeloop_org_id = "swarnabhaghosh2005"
my_activeloop_dataset_name = "langchain_course_jarvis_assistant"
dataset_path = f'hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}'

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_documentation_urls():
    return [
        '/docs/huggingface_hub/guides/overview',
        '/docs/huggingface_hub/guides/download',
        '/docs/huggingface_hub/guides/upload',
        '/docs/huggingface_hub/guides/hf_file_system',
        '/docs/huggingface_hub/guides/repository',
        '/docs/huggingface_hub/guides/search',
    ]

def construct_full_url(base_url, relative_url):
    return base_url + relative_url

def scrape_page_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.body.text.strip()
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\xff]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def scrape_all_content(base_url, relative_urls, filename):
    content = []
    for relative_url in relative_urls:
        full_url = construct_full_url(base_url, relative_url)
        scraped_content = scrape_page_content(full_url)
        content.append(scraped_content.rstrip('\n'))
    with open(filename, 'w', encoding='utf-8') as file:
        for item in content:
            file.write("%s\n" % item)
    return content

def load_docs(root_dir, filename):
    docs = []
    try:
        loader = TextLoader(os.path.join(root_dir, filename), encoding='utf-8')
        docs.extend(loader.load_and_split())
    except Exception:
        pass
    return docs

def split_docs(docs):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(docs)

def main():
    base_url = 'https://huggingface.co'
    filename = 'content.txt'
    root_dir = './'
    relative_urls = get_documentation_urls()
    scrape_all_content(base_url, relative_urls, filename)
    docs = load_docs(root_dir, filename)
    texts = split_docs(docs)
    db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)
    db.add_documents(texts)
    os.remove(filename)

if __name__ == '__main__':
    main()