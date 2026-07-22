import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import DeepLake

load_dotenv()

# ACTIVELOOP_TOKEN and GROQ_API_KEY are read automatically from .env

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
import os
from langchain_community.document_loaders import TextLoader

root_dir = './the-algorithm'
docs = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    for file in filenames:
        try:
            loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
            docs.extend(loader.load_and_split())
        except Exception as e:
            pass

print(f"Loaded {len(docs)} documents")

from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)

print(f"Split into {len(texts)} chunks")

username = "swarnabhaghosh2005" 
db = DeepLake(dataset_path="./deeplake_data/twitter-algorithm", embedding_function=embeddings)
db.add_documents(texts)


