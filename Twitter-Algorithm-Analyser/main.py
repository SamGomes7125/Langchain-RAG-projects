import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import DeepLake

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = DeepLake(dataset_path="./deeplake_data/twitter-algorithm", read_only=True, embedding_function=embeddings)

retriever = db.as_retriever()
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['fetch_k'] = 100
retriever.search_kwargs['k'] = 10

def filter(x):
    if 'com.google' in x['text'].data()['value']:
        return False
    metadata = x['metadata'].data()['value']
    return 'scala' in metadata['source'] or 'py' in metadata['source']

retriever.search_kwargs['filter'] = filter

from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationalRetrievalChain

model = ChatGroq(model_name='llama-3.3-70b-versatile', temperature=0)
qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)

import time

questions = [
    "What does favCountParams do?",
    "is it Likes + Bookmarks, or not clear from the code?",
    "What are the major negative modifiers that lower your linear ranking parameters?",
    "How do you get assigned to SimClusters?",
    "What is needed to migrate from one SimClusters to another SimClusters?",
    "How much do I get boosted within my cluster?",
    "How does Heavy ranker work. what are it's main inputs?",
    "How can one influence Heavy ranker?",
    "why threads and long tweets do so well on the platform?",
    "Are thread and long tweet creators building a following that reacts to only threads?",
    "Do you need to follow different strategies to get most followers vs to get most likes and bookmarks per tweet?",
    "Content meta data and how it impacts virality (e.g. ALT in images).",
    "What are some unexpected fingerprints for spam factors?",
    "Is there any difference between company verified checkmarks and blue verified individual checkmarks?",
]

MAX_HISTORY_TURNS = 3  # only keep the last few Q&A pairs as context
MAX_RETRIES = 3

chat_history = []

for question in questions:
    trimmed_history = chat_history[-MAX_HISTORY_TURNS:]

    for attempt in range(MAX_RETRIES):
        try:
            result = qa.invoke({"question": question, "chat_history": trimmed_history})
            break
        except Exception as e:
            wait_time = 2 ** attempt  # 1s, 2s, 4s backoff
            print(f"Error on question '{question}': {e}")
            print(f"Retrying in {wait_time}s... (attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(wait_time)
    else:
        print(f"Failed to get answer for: {question} after {MAX_RETRIES} attempts. Skipping.")
        continue

    chat_history.append((question, result['answer']))
    print(f"-> **Question**: {question} \n")
    print(f"**Answer**: {result['answer']} \n")

    time.sleep(1)  # small pause between calls to stay under rate limits